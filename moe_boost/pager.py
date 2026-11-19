"""Python wrapper for the optional C++ positional reader."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

try:
    from moe_pager_cpp import Pager as _CppPager
except ImportError:  # Allows API development before the extension is built.
    _CppPager = None


class Pager:
    def __init__(self) -> None:
        self._native = _CppPager() if _CppPager else None
        self._files: dict[int, tuple[int, int]] = {}
        self._next_handle = 1
        self._metrics_lock = threading.Lock()
        self._bytes_read = 0
        self._read_ops = 0
        self._read_ms = 0.0
        self._expert_heat: dict[str, int] = {}

    def measure_ram_bandwidth(self, data) -> float:
        """ms to read len(data) bytes at multi-threaded AVX2 DDR read speed; 0.0 if unavailable."""
        if self._native and hasattr(self._native, "measure_ram_bandwidth"):
            try:
                return float(self._native.measure_ram_bandwidth(data))
            except Exception:
                return 0.0
        return 0.0

    def calibrate_ram_bandwidth(self, thread_candidates: list[int] | None = None) -> dict:
        """Benchmark candidate thread counts in C++ AVX2 and lock in the fastest DDR4 read speed."""
        candidates = [int(t) for t in (thread_candidates or [2, 4, 8]) if int(t) > 0]
        if self._native and hasattr(self._native, "calibrate_ram_bandwidth"):
            try:
                return dict(self._native.calibrate_ram_bandwidth(candidates))
            except Exception:
                pass
        return {"best_threads": 0, "bandwidth_mbs": 0.0, "bandwidth_gbs": 0.0, "candidates": []}

    def advise_ram_buffer(self, data) -> bool:
        """Apply MADV_WILLNEED and MADV_HUGEPAGE (>=2MB) to a resident host buffer."""
        if self._native and hasattr(self._native, "advise_ram_buffer"):
            try:
                return bool(self._native.advise_ram_buffer(data))
            except Exception:
                return False
        return False

    def advise_file_sequential(self, handle: int) -> bool:
        """Apply POSIX_FADV_SEQUENTIAL and POSIX_FADV_WILLNEED on an open pager handle."""
        if self._native and hasattr(self._native, "advise_file_sequential"):
            try:
                return bool(self._native.advise_file_sequential(int(handle)))
            except Exception:
                return False
        return False

    @property
    def native(self) -> bool:
        return self._native is not None

    def open(self, path: str) -> int:
        if self._native:
            return int(self._native.open(os.fspath(path)))
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        size = os.fstat(fd).st_size
        with self._metrics_lock:
            handle = self._next_handle
            self._next_handle += 1
            self._files[handle] = (fd, size)
        return handle

    def close(self, handle: int) -> None:
        if self._native:
            self._native.close(handle)
            return
        with self._metrics_lock:
            item = self._files.pop(handle, None)
        if item is None:
            raise ValueError("unknown or already closed file handle")
        os.close(item[0])

    def file_size(self, handle: int) -> int:
        if self._native:
            return int(self._native.file_size(handle))
        with self._metrics_lock:
            item = self._files.get(handle)
        if item is None:
            raise ValueError("unknown or already closed file handle")
        return item[1]

    def pread(self, handle: int, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        if self._native:
            return self._native.pread(handle, offset, length)
        with self._metrics_lock:
            item = self._files.get(handle)
        if item is None:
            raise ValueError("unknown or already closed file handle")
        fd, size = item
        if offset > size or length > size - offset:
            raise ValueError("requested byte range exceeds file bounds")
        started = time.perf_counter()
        data = os.pread(fd, length, offset)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(data) != length:
            raise OSError("unexpected end of file during pread")
        with self._metrics_lock:
            self._bytes_read += len(data)
            self._read_ms += elapsed_ms
            self._read_ops += 1
        return data

    def pread_array(self, handle: int, offset: int, length: int) -> np.ndarray:
        """Return a uint8 view backed by a NumPy allocation filled by positional read."""
        if length < 0:
            raise ValueError("length must be non-negative")
        output = np.empty(length, dtype=np.uint8)
        if self._native:
            self._native.pread_into(handle, offset, output)
            return output
        data = self.pread(handle, offset, length)
        try:
            view = np.frombuffer(data, dtype=np.uint8, count=length, copy=False)
        except TypeError:  # NumPy before the `copy` keyword; frombuffer is still a view.
            view = np.frombuffer(data, dtype=np.uint8, count=length)
        output[:] = view
        return output

    def pread_into(self, handle: int, offset: int, output: np.ndarray) -> None:
        if output.dtype != np.uint8 or output.ndim != 1 or not output.flags.c_contiguous:
            raise ValueError("output must be a contiguous one-dimensional uint8 NumPy array")
        if self._native:
            self._native.pread_into(handle, offset, output)
            return
        data = self.pread(handle, offset, output.size)
        try:
            view = np.frombuffer(data, dtype=np.uint8, count=output.size, copy=False)
        except TypeError:  # NumPy before the `copy` keyword.
            view = np.frombuffer(data, dtype=np.uint8, count=output.size)
        output[:] = view

    def pread_segments_into(self, handle: int, segments: list[tuple[int, int]], output: np.ndarray) -> None:
        if output.dtype != np.uint8 or output.ndim != 1 or not output.flags.c_contiguous:
            raise ValueError("output must be a contiguous one-dimensional uint8 NumPy array")
        if any(offset < 0 or length < 0 for offset, length in segments):
            raise ValueError("segment offsets and lengths must be non-negative")
        if self._native and hasattr(self._native, "pread_segments_into"):
            self._native.pread_segments_into(handle, [(int(o), int(n)) for o, n in segments], output)
            return
        total = sum(int(length) for _, length in segments)
        if total != output.size:
            raise ValueError("sum of segment lengths does not match output buffer size")
        cursor = 0
        for offset, length in segments:
            self.pread_into(handle, int(offset), output[cursor:cursor + int(length)])
            cursor += int(length)

    def record_expert(self, expert_id: str) -> None:
        if self._native:
            self._native.record_expert(expert_id)
            return
        with self._metrics_lock:
            self._expert_heat[expert_id] = self._expert_heat.get(expert_id, 0) + 1

    def metrics(self) -> dict:
        if self._native:
            return dict(self._native.metrics())
        with self._metrics_lock:
            throughput = ((self._bytes_read / (1024 * 1024)) / (self._read_ms / 1000.0)) if self._read_ms else 0.0
            return {"time_used_for_reading_ms": self._read_ms, "bytes_read": self._bytes_read,
                    "read_ops": self._read_ops,
                    "actual_ssd_throughput_mbs": throughput, "expert_heat": dict(self._expert_heat)}


pager = Pager()
