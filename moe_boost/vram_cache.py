"""Bounded CUDA staging cache for frequently used MoE expert blocks.

This cache is deliberately capability-gated. The current llama.cpp inference
adapter does not expose expert-level tensor hooks, so staged tensors are NOT
claimed to accelerate generation until a backend consumes acquire_device().
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class DeviceEntry:
    tensor: Any
    size_bytes: int
    hits: int
    last_used: float
    refs: int = 0
    source: str = "ram"


class _CudaDeviceBuffer:
    def __init__(self, ptr: int, size_bytes: int, lib: Any, owner: Any = None) -> None:
        self._owner = owner
        self.ptr = int(ptr)
        self.size_bytes = int(size_bytes)
        self._lib = lib

    def free(self) -> None:
        if self.ptr:
            try:
                self._lib.cudaFree(self.ptr)
            except Exception:
                pass
            self.ptr = 0

    def __del__(self) -> None:
        self.free()


class CtypesCudaRuntimeAdapter:
    """Direct libcudart.so adapter so VRAM staging works without a 3 GB PyTorch install."""

    def __init__(self, device: int = 0) -> None:
        import ctypes
        import ctypes.util
        self.ctypes = ctypes
        lib_name = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(lib_name)
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int
        self.device = int(device)
        self._pool: dict[int, list[int]] = {}
        self._pinned_ptr = 0
        self._pinned_size = 0
        self._start_event = ctypes.c_void_p(0)
        self._stop_event = ctypes.c_void_p(0)
        self._stream = ctypes.c_void_p(0)
        self._stream_ok = False
        self._events_ok = False
        self._scratch_ptr = 0
        self._slab_src = 0
        self._scratch_size = 0
        self.last_alloc_ms = 0.0
        self.last_upload_ms = 0.0
        if self.lib.cudaSetDevice(self.device) != 0:
            raise RuntimeError(f"cudaSetDevice({self.device}) failed")
        free, total = self.memory_info()
        if total <= 0:
            raise RuntimeError("No CUDA device memory detected")
        try:
            lib = self.lib
            lib.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cudaMallocHost.restype = ctypes.c_int
            lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
            lib.cudaFreeHost.restype = ctypes.c_int
            lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            lib.cudaMemcpyAsync.restype = ctypes.c_int
            lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
            lib.cudaMemsetAsync.restype = ctypes.c_int
            lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
            lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
            lib.cudaStreamDestroy.restype = ctypes.c_int
            for _n in ("cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime", "cudaEventDestroy"):
                getattr(lib, _n).restype = ctypes.c_int
            lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
            lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
            lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
            # 1 = cudaStreamNonBlocking
            self._stream_ok = (lib.cudaStreamCreateWithFlags(ctypes.byref(self._stream), 1) == 0
                               and bool(self._stream.value))
            self._events_ok = (lib.cudaEventCreate(ctypes.byref(self._start_event)) == 0
                               and lib.cudaEventCreate(ctypes.byref(self._stop_event)) == 0)
        except Exception:
            self._stream_ok = False
            self._events_ok = False

    def _active_stream(self):
        return self._stream if self._stream_ok else None

    def _timed_copy(self, dst: int, src: int, size: int, kind: int, passes: int | None = None) -> tuple[int, float]:
        """cudaMemcpy timed by CUDA events when available (pure GPU/DMA time), else wall clock."""
        c = self.ctypes
        stream = self._active_stream()
        if self._events_ok:
            passes = passes or (4 if kind == 3 else 1)
            self.lib.cudaEventRecord(self._start_event, stream)
            rc = 0
            for _ in range(passes):
                rc = self.lib.cudaMemcpyAsync(c.c_void_p(dst), c.c_void_p(src), size, kind, stream)
            self.lib.cudaEventRecord(self._stop_event, stream)
            self.lib.cudaEventSynchronize(self._stop_event)
            ms = c.c_float(0.0)
            if self.lib.cudaEventElapsedTime(c.byref(ms), self._start_event, self._stop_event) == 0 and ms.value > 0:
                return rc, max(0.0005, float(ms.value) / passes)
        started = time.perf_counter()
        rc = self.lib.cudaMemcpy(c.c_void_p(dst), c.c_void_p(src), size, kind)
        self.lib.cudaDeviceSynchronize()
        return rc, max(0.001, (time.perf_counter() - started) * 1000.0)

    def _timed_unidirectional_stream(self, ptr: int, size: int, passes: int = 8) -> tuple[int, float]:
        """Unidirectional GDDR5 streaming burst on dedicated non-blocking stream."""
        c = self.ctypes
        stream = self._active_stream()
        if self._events_ok:
            self.lib.cudaEventRecord(self._start_event, stream)
            rc = 0
            for _ in range(passes):
                rc = self.lib.cudaMemsetAsync(c.c_void_p(ptr), 0x5a, size, stream)
            self.lib.cudaEventRecord(self._stop_event, stream)
            self.lib.cudaEventSynchronize(self._stop_event)
            ms = c.c_float(0.0)
            if self.lib.cudaEventElapsedTime(c.byref(ms), self._start_event, self._stop_event) == 0 and ms.value > 0:
                return rc, max(0.0005, float(ms.value) / passes)
        return -1, 0.0

    def _pinned(self, size: int) -> int:
        if self._pinned_size < size:
            if self._pinned_ptr:
                self.lib.cudaFreeHost(self.ctypes.c_void_p(self._pinned_ptr))
                self._pinned_ptr = 0
                self._pinned_size = 0
            p = self.ctypes.c_void_p(0)
            if not self._events_ok or self.lib.cudaMallocHost(self.ctypes.byref(p), size) != 0 or not p.value:
                return 0
            self._pinned_ptr, self._pinned_size = int(p.value), size
        return self._pinned_ptr

    def memory_info(self) -> tuple[int, int]:
        free = self.ctypes.c_size_t(0)
        total = self.ctypes.c_size_t(0)
        if self.lib.cudaMemGetInfo(self.ctypes.byref(free), self.ctypes.byref(total)) != 0:
            raise RuntimeError("cudaMemGetInfo failed")
        return int(free.value), int(total.value)

    def upload(self, data: Any) -> _CudaDeviceBuffer:
        import numpy as np
        arr = np.ascontiguousarray(data, dtype=np.uint8)
        size = int(arr.nbytes)
        ptr = self.ctypes.c_void_p(0)
        alloc_started = time.perf_counter()
        pooled = self._pool.get(size)
        if pooled:
            ptr.value = pooled.pop()
        elif self.lib.cudaMalloc(self.ctypes.byref(ptr), size) != 0 or not ptr.value:
            raise RuntimeError(f"cudaMalloc({size}) failed")
        self.last_alloc_ms = (time.perf_counter() - alloc_started) * 1000.0
        # cudaMemcpyHostToDevice = 1; timed separately from allocation.
        src_ptr = arr.ctypes.data
        pinned = self._pinned(size)
        if pinned:
            self.ctypes.memmove(pinned, src_ptr, size)
            src_ptr = pinned
        rc, self.last_upload_ms = self._timed_copy(int(ptr.value), src_ptr, size, 1)
        if rc != 0:
            self.lib.cudaFree(ptr)
            raise RuntimeError(f"cudaMemcpy HostToDevice failed (rc={rc})")
        return _CudaDeviceBuffer(int(ptr.value), size, self.lib, self)

    def _ensure_slab(self, slab: int) -> bool:
        if self._scratch_size >= slab and self._scratch_ptr and self._slab_src:
            return True
        for ptr_attr in ("_scratch_ptr", "_slab_src"):
            old = getattr(self, ptr_attr, 0)
            if old:
                self.lib.cudaFree(self.ctypes.c_void_p(old))
                setattr(self, ptr_attr, 0)
        self._scratch_size = 0
        a, b = self.ctypes.c_void_p(0), self.ctypes.c_void_p(0)
        if self.lib.cudaMalloc(self.ctypes.byref(a), slab) != 0 or not a.value:
            return False
        if self.lib.cudaMalloc(self.ctypes.byref(b), slab) != 0 or not b.value:
            self.lib.cudaFree(a)
            return False
        self._scratch_ptr, self._slab_src, self._scratch_size = int(a.value), int(b.value), slab
        return True

    def warm_up_and_preallocate(self, pinned_bytes: int = 16 * 1024 * 1024, slab_bytes: int = 64 * 1024 * 1024) -> dict:
        """Warm up dedicated CUDA stream, pre-allocate pinned DMA buffer & 64 MB VRAM slab."""
        pinned_ok = bool(self._pinned(max(1024 * 1024, int(pinned_bytes))))
        slab_ok = self._ensure_slab(max(4 * 1024 * 1024, int(slab_bytes)))
        if slab_ok:
            self._timed_unidirectional_stream(self._scratch_ptr, self._scratch_size, passes=2)
            self._timed_copy(self._scratch_ptr, self._slab_src, self._scratch_size, 3, passes=2)
        return {
            "non_blocking_stream": self._stream_ok,
            "pinned_dma_bytes": self._pinned_size if pinned_ok else 0,
            "vram_slab_bytes": self._scratch_size if slab_ok else 0,
        }

    def read_device(self, tensor: Any) -> float:
        """Calibrated GDDR5 bus time (ms) for reading `size` bytes on a non-blocking stream.

        Compares unidirectional GDDR5 streaming (no read/write turnaround stalls) and
        bidirectional D2D (2 * copied bytes), taking the faster calibrated bus time.
        """
        size = int(tensor.size_bytes)
        if not tensor.ptr or size <= 0:
            return 0.0
        slab = max(size * 16, 64 * 1024 * 1024)
        if not self._ensure_slab(slab):
            return self._read_device_small(tensor, size)
        # Warm up GPU clocks on the dedicated non-blocking stream.
        self._timed_unidirectional_stream(self._scratch_ptr, slab, passes=2)
        self._timed_copy(self._scratch_ptr, self._slab_src, slab, 3, passes=2)
        rc_uni, ms_uni_pass = self._timed_unidirectional_stream(self._scratch_ptr, slab, passes=8)
        rc_bidir, ms_bidir_pass = self._timed_copy(self._scratch_ptr, self._slab_src, slab, 3, passes=8)
        candidates: list[float] = []
        if rc_uni == 0 and ms_uni_pass > 0:
            candidates.append(ms_uni_pass * size / float(slab))
        if rc_bidir == 0 and ms_bidir_pass > 0:
            candidates.append(ms_bidir_pass * size / (2.0 * float(slab)))
        if not candidates:
            return self._read_device_small(tensor, size)
        return max(0.0005, min(candidates))

    def _read_device_small(self, tensor: Any, size: int) -> float:
        if self._scratch_size < size:
            if self._scratch_ptr:
                self.lib.cudaFree(self.ctypes.c_void_p(self._scratch_ptr))
                self._scratch_ptr = 0
                self._scratch_size = 0
            p = self.ctypes.c_void_p(0)
            if self.lib.cudaMalloc(self.ctypes.byref(p), size) != 0 or not p.value:
                return 0.0
            self._scratch_ptr, self._scratch_size = int(p.value), size
        rc, ms = self._timed_copy(self._scratch_ptr, int(tensor.ptr), size, 3)
        return (ms / 2.0) if rc == 0 else 0.0

    def synchronize(self) -> None:
        self.lib.cudaDeviceSynchronize()

    def release(self, tensor: Any, _size_bytes: int = 0) -> None:
        ptr = int(getattr(tensor, "ptr", 0) or 0)
        size = int(getattr(tensor, "size_bytes", 0) or 0)
        if ptr and size:
            self._pool.setdefault(size, []).append(ptr)
            tensor.ptr = 0
            return
        free_fn = getattr(tensor, "free", None)
        if callable(free_fn):
            free_fn()

    def empty_cache(self) -> None:
        self.synchronize()
        for ptrs in self._pool.values():
            for p in ptrs:
                self.lib.cudaFree(self.ctypes.c_void_p(p))
        self._pool.clear()
        if self._slab_src:
            self.lib.cudaFree(self.ctypes.c_void_p(self._slab_src))
            self._slab_src = 0
        if self._scratch_ptr:
            self.lib.cudaFree(self.ctypes.c_void_p(self._scratch_ptr))
            self._scratch_ptr = 0
            self._scratch_size = 0
        if self._pinned_ptr:
            self.lib.cudaFreeHost(self.ctypes.c_void_p(self._pinned_ptr))
            self._pinned_ptr = 0
            self._pinned_size = 0

    def __del__(self) -> None:
        try:
            self.empty_cache()
            if self._events_ok:
                self.lib.cudaEventDestroy(self._start_event)
                self.lib.cudaEventDestroy(self._stop_event)
                self._events_ok = False
            if getattr(self, "_stream_ok", False) and self._stream.value:
                self.lib.cudaStreamDestroy(self._stream)
                self._stream = self.ctypes.c_void_p(0)
                self._stream_ok = False
        except Exception:
            pass


class TorchCudaAdapter:
    def __init__(self, device: int = 0):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(f"cuda:{device}")

    def memory_info(self) -> tuple[int, int]:
        free, total = self.torch.cuda.mem_get_info(self.device)
        return int(free), int(total)

    def upload(self, data):
        import numpy as np
        source = np.asarray(data)
        # Own the host buffer before transfer; do not retain cache-owned memory.
        return self.torch.from_numpy(source.copy()).to(self.device, non_blocking=False)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def release(self, tensor: Any, _size_bytes: int = 0) -> None:
        del tensor

    def empty_cache(self) -> None:
        try:
            self.torch.cuda.empty_cache()
            if hasattr(self.torch.cuda, "ipc_collect"):
                self.torch.cuda.ipc_collect()
        except Exception:
            pass


class VramExpertCache:
    def __init__(self, headroom_mb: int = 500, max_fraction: float = 0.70,
                 promote_after_hits: int = 2, adapter=None, promotion_cooldown_seconds: float = 2.0):
        self.headroom_bytes = max(0, int(headroom_mb)) * 1024 * 1024
        self.max_fraction = min(0.90, max(0.10, float(max_fraction)))
        self.promote_after_hits = max(1, int(promote_after_hits))
        self.promotion_cooldown_seconds = max(0.0, float(promotion_cooldown_seconds))
        self.last_evicted_at: dict[str, float] = {}
        self.entries: OrderedDict[str, DeviceEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.adapter = adapter
        self.on_demote = None
        self.demotion_results: dict[str, str] = {}
        self.disabled_reason = None
        self.read_time_ms: float = 0.0
        self.bytes_read: int = 0
        self.read_ops: int = 0
        self.upload_time_ms: float = 0.0
        self.upload_bytes: int = 0
        if self.adapter is None:
            try:
                self.adapter = TorchCudaAdapter()
            except Exception:
                try:
                    self.adapter = CtypesCudaRuntimeAdapter()
                except Exception as exc:
                    self.disabled_reason = str(exc)

    @property
    def available(self) -> bool:
        return self.adapter is not None

    def _budget(self) -> tuple[int, int, int]:
        if not self.available:
            return 0, 0, 0
        free, total = self.adapter.memory_info()
        # Both the free-memory headroom and a fraction of total VRAM are enforced.
        cap = int(total * self.max_fraction)
        used_by_cache = sum(item.size_bytes for item in self.entries.values())
        budget = max(0, min(max(0, cap - used_by_cache), free - self.headroom_bytes))
        return budget, free, total

    def _device_read_ms(self, tensor: Any, size: int) -> float:
        cal = self.__dict__.setdefault("_calibrated_read_ms", {})
        if size in cal:
            return cal[size]
        fn = getattr(self.adapter, "read_device", None)
        if callable(fn):
            try:
                ms = float(fn(tensor))
                if ms > 0:
                    cal[size] = ms
                    return ms
            except Exception:
                pass
        return 0.001

    def _release_tensor(self, tensor: Any, size_bytes: int) -> None:
        release_fn = getattr(self.adapter, "release", None)
        if callable(release_fn):
            try:
                release_fn(tensor, size_bytes)
            except Exception:
                pass
        del tensor

    def _evict_one(self) -> bool:
        candidates = [(entry.hits, entry.last_used, key) for key, entry in self.entries.items()
                      if entry.refs == 0]
        if not candidates:
            return False
        _, _, key = min(candidates)
        # Removing the CUDA copy is a VRAM -> RAM demotion. ExpertCache keeps
        # the host copy resident while this key is in entries, so the demotion
        # is immediate and does not trigger an SSD read.
        entry = self.entries.get(key)
        if entry is not None:
            demoted = False
            if callable(self.on_demote):
                try:
                    demoted = bool(self.on_demote(key, entry.size_bytes))
                except Exception:
                    demoted = False
            self.demotion_results[key] = "RAM" if demoted else "SSD_overflow"
            self.last_evicted_at[key] = time.monotonic()
            self.entries.pop(key, None)
            self._release_tensor(entry.tensor, entry.size_bytes)
        return True

    def contains(self, expert_id: str) -> bool:
        with self.lock:
            return str(expert_id) in self.entries

    def remove(self, expert_id: str) -> bool:
        """Remove an unpinned device copy; return False while actively leased."""
        with self.lock:
            key = str(expert_id)
            entry = self.entries.get(key)
            if entry is None:
                return True
            if entry.refs > 0:
                return False
            self.entries.pop(key, None)
            self._release_tensor(entry.tensor, entry.size_bytes)
            return True

    def unload_keys(self, keys: set[str], force: bool = False) -> int:
        """Remove all matching keys from VRAM and return freed bytes."""
        freed = 0
        with self.lock:
            for key in list(keys):
                entry = self.entries.get(str(key))
                if entry is None:
                    continue
                if entry.refs > 0 and not force:
                    continue
                self.entries.pop(str(key), None)
                freed += int(entry.size_bytes)
                self._release_tensor(entry.tensor, entry.size_bytes)
            if freed > 0:
                empty_fn = getattr(self.adapter, "empty_cache", None)
                if callable(empty_fn):
                    empty_fn()
        return freed

    def clear(self, force: bool = False) -> int:
        """Drop all cached VRAM tensors and flush the CUDA allocator cache."""
        with self.lock:
            freed = self.unload_keys(set(self.entries.keys()), force=force)
            empty_fn = getattr(self.adapter, "empty_cache", None)
            if callable(empty_fn):
                try:
                    empty_fn()
                except Exception:
                    pass
            return freed

    def observe(self, expert_id: str, host_data, hit_count: int, source: str = "ram") -> bool:
        """Stage a hot expert from RAM or a bounded temporary SSD read into CUDA."""
        key = str(expert_id)
        if not self.available or int(hit_count) < self.promote_after_hits:
            return False
        with self.lock:
            current = self.entries.get(key)
            if current is not None:
                current.hits = max(current.hits, int(hit_count))
                current.last_used = time.monotonic()
                self.entries.move_to_end(key)
                elapsed_ms = self._device_read_ms(current.tensor, current.size_bytes)
                self.read_time_ms += elapsed_ms
                self.bytes_read += int(current.size_bytes)
                self.read_ops += 1
                return True
            evicted_at = self.last_evicted_at.get(key)
            if (evicted_at is not None and self.promotion_cooldown_seconds > 0
                    and time.monotonic() - evicted_at < self.promotion_cooldown_seconds):
                return False
            size = int(getattr(host_data, "nbytes", len(host_data)))
            budget, _free, _total = self._budget()
            if size <= 0:
                return False
            while size > budget:
                if not self._evict_one():
                    return False
                budget, _free, _total = self._budget()
            try:
                tensor = self.adapter.upload(host_data)
                upload_ms = float(getattr(self.adapter, "last_upload_ms", 0.0))
                if upload_ms <= 0:
                    started = time.perf_counter()
                    self.adapter.synchronize()
                    upload_ms = max(0.001, (time.perf_counter() - started) * 1000.0)
                self.upload_time_ms += upload_ms
                self.upload_bytes += size
                elapsed_ms = self._device_read_ms(tensor, size)
                # Recheck after allocation: concurrent GPU workloads may have consumed headroom.
                free_after, _total_after = self.adapter.memory_info()
                if free_after < self.headroom_bytes:
                    self._release_tensor(tensor, size)
                    return False
                self.entries[key] = DeviceEntry(tensor, size, int(hit_count), time.monotonic(), 0,
                                                "ssd" if str(source).lower() == "ssd" else "ram")
                self.read_time_ms += elapsed_ms
                self.bytes_read += size
                self.read_ops += 1
                return True
            except Exception:
                return False

    def acquire_device(self, expert_id: str):
        """Return a device tensor and pin it against eviction until release_device()."""
        with self.lock:
            entry = self.entries.get(str(expert_id))
            if entry is None:
                return None
            entry.refs += 1
            entry.last_used = time.monotonic()
            self.entries.move_to_end(str(expert_id))
            elapsed_ms = self._device_read_ms(entry.tensor, entry.size_bytes)
            self.read_time_ms += elapsed_ms
            self.bytes_read += int(entry.size_bytes)
            self.read_ops += 1
            return entry.tensor

    def release_device(self, expert_id: str) -> None:
        with self.lock:
            entry = self.entries.get(str(expert_id))
            if entry is not None:
                entry.refs = max(0, entry.refs - 1)

    def snapshot(self) -> dict:
        with self.lock:
            budget, free, total = self._budget()
            resident = sum(item.size_bytes for item in self.entries.values())
            speed_mbs = ((self.bytes_read / (1024.0 * 1024.0)) / (self.read_time_ms / 1000.0)) if self.read_time_ms > 0 else 0.0
            upload_mbs = ((self.upload_bytes / (1024.0 * 1024.0)) / (self.upload_time_ms / 1000.0)) if self.upload_time_ms > 0 else 0.0
            return {
                "available": self.available,
                "enabled": self.available,
                "mode": "cuda_staging_cache_only",
                "tier_priority": ["VRAM", "RAM", "SSD"],
                "demotion_target": "RAM",
                "inference_backend_connected": False,
                "disabled_reason": self.disabled_reason,
                "headroom_bytes": self.headroom_bytes,
                "max_fraction": self.max_fraction,
                "promotion_cooldown_seconds": self.promotion_cooldown_seconds,
                "budget_bytes": budget,
                "resident_bytes": resident,
                "free_bytes": free,
                "total_bytes": total,
                "read_time_ms": round(self.read_time_ms, 3),
                "bytes_read": self.bytes_read,
                "read_ops": self.read_ops,
                "read_speed_mbs": round(speed_mbs, 2),
                "upload_time_ms": round(self.upload_time_ms, 3),
                "upload_bytes": self.upload_bytes,
                "upload_speed_mbs": round(upload_mbs, 2),
                "cached_experts": list(self.entries),
                "hits": {key: item.hits for key, item in self.entries.items()},
                "promotion_sources": {key: item.source for key, item in self.entries.items()},
                "last_demotion_targets": dict(self.demotion_results),
            }
