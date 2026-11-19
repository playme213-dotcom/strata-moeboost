#!/usr/bin/env python3
"""
Strata-MoEBoost Hybrid Unpacker & Runtime Manager
Unpacks GGUF models into dense weight blobs, tokenizers, and expert layouts
for high-speed zero-overhead execution by the native Strata-HotRod C++ engine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def unpack_gguf(gguf_path: str, pack_dir: str, compat_bf16: bool = True, skip_experts: bool = True) -> dict:
    """
    Unpacks GGUF model into pack directory for native strata execution.
    Creates:
      - dense.bin
      - native_experts.txt
      - tokenizer/vocab.json
    """
    p_gguf = Path(gguf_path)
    if not p_gguf.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    p_out = Path(pack_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    dense_bin = p_out / "dense.bin"
    vocab_json = p_out / "tokenizer" / "vocab.json"
    native_exp = p_out / "native_experts.txt"

    if dense_bin.exists() and vocab_json.exists():
        return {
            "status": "ready",
            "pack_dir": str(p_out),
            "dense_bin": str(dense_bin),
            "vocab_json": str(vocab_json),
            "reused": True
        }

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "iq_pack.py"),
        "--gguf", str(p_gguf),
        "--out", str(p_out)
    ]
    if compat_bf16:
        cmd.append("--compat-bf16")
    if skip_experts:
        cmd.append("--skip-experts")

    print(f"[Strata-Unpacker] Unpacking {p_gguf.name} -> {p_out}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Unpack failed: {res.stderr or res.stdout}")

    return {
        "status": "created",
        "pack_dir": str(p_out),
        "dense_bin": str(dense_bin),
        "vocab_json": str(vocab_json),
        "reused": False
    }

def main():
    parser = argparse.ArgumentParser(description="Unpack GGUF for native Strata-HotRod engine")
    parser.add_argument("--gguf", required=True, help="Input GGUF model path")
    parser.add_argument("--out", default="pack/unpacked", help="Output pack directory")
    parser.add_argument("--with-experts", action="store_true", help="Also extract full experts.bin")
    args = parser.parse_args()

    info = unpack_gguf(args.gguf, args.out, skip_experts=not args.with_experts)
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
