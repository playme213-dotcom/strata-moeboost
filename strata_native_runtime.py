#!/usr/bin/env python3
"""
Strata-MoEBoost Native C++ Engine Driver (Zero llama-cpp-python).
Communicates with `build-cuda12/strata --serve` over stdin/stdout FIFO pipes.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator, Any

ROOT = Path(__file__).resolve().parent

class NativeStrataEngine:
    """Zero llama-cpp-python native C++ engine controller."""
    def __init__(self,
                 bin_path: str = str(ROOT / "build-cuda12" / "strata"),
                 pack_dir: str = str(ROOT / "pack" / "unpacked"),
                 shard1_path: str = "",
                 shard2_path: str = "",
                 resident_budget_gib: float = 10.0,
                 kv_format: str = "q4_0",
                 port: int = 4133):
        self.bin_path = bin_path
        self.pack_dir = pack_dir
        self.shard1_path = shard1_path
        self.shard2_path = shard2_path
        self.resident_budget_gib = resident_budget_gib
        self.kv_format = kv_format
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.max_context: int = 4096
        self.ready: bool = False
        self.lock = threading.Lock()

    def start(self, timeout: float = 60.0):
        if self.proc and self.proc.poll() is None:
            return

        cmd = [
            self.bin_path,
            "--serve",
            "--pack", self.pack_dir,
            "--kv", self.kv_format,
            "--mmap-experts",
            "--resident-budget-gib", str(self.resident_budget_gib),
            "--ple-io", "direct",
            "--pool-workers", "4"
        ]
        if self.shard1_path:
            cmd.extend(["--native", self.shard1_path])
        if self.shard2_path:
            cmd.extend(["--ple-gguf", self.shard2_path])

        print(f"[NativeStrataEngine] Spawning: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def _reader():
            for line in self.proc.stdout:
                line_str = line.strip()
                self.lines.put(line_str)
                if line_str.startswith("READY"):
                    parts = line_str.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        self.max_context = int(parts[1])
                    self.ready = True

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        # Wait for READY
        t_deadline = time.time() + timeout
        while time.time() < t_deadline:
            if self.ready:
                print(f"[NativeStrataEngine] Engine ready! Max context: {self.max_context}")
                return
            if self.proc.poll() is not None:
                err = self.proc.stderr.read()
                raise RuntimeError(f"Native strata engine exited prematurely (code {self.proc.returncode}): {err}")
            time.sleep(0.1)

        raise TimeoutError(f"Native strata engine failed to report READY within {timeout}s.")

    def stream_generate(self, token_ids: list[int], max_new: int = 512, temperature: float = 0.7) -> Generator[int, None, None]:
        """Sends GEN line to native engine and streams token IDs as they decode."""
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("Engine not running")

        with self.lock:
            # Protocol: GEN <max_new> temperature=<F> <token_ids...>
            ids_str = " ".join(str(tid) for tid in token_ids)
            gen_cmd = f"GEN {max_new} temperature={temperature:.2f} {ids_str}\n"
            self.proc.stdin.write(gen_cmd)
            self.proc.stdin.flush()

            while True:
                line = self.lines.get()
                if line.startswith("T "):
                    # T <token_id>
                    parts = line.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        yield int(parts[1])
                elif line.startswith("DONE") or line.startswith("ERR"):
                    break

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.ready = False
