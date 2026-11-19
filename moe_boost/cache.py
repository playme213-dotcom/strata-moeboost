"""Bounded, reference-counted RAM cache for byte ranges of expert files."""
from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .pager import pager
from .queue_engine import ExpertExecutionQueue
from .vram_cache import VramExpertCache

try:
    import psutil
except ImportError:
    psutil = None


def _drop_os_page_cache(path: Path | str) -> bool:
    """Advise the Linux kernel to immediately drop clean page-cache pages for a file."""
    fadv_dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadv_dontneed is None or not hasattr(os, "posix_fadvise"):
        return False
    fd = -1
    try:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            return False
        fd = os.open(str(target), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fdatasync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, fadv_dontneed)
        return True
    except OSError:
        return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _trim_process_heap() -> bool:
    """Run Python GC and return unused glibc heap arenas back to the OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
            return True
    except Exception:
        pass
    return False


def _parse_bytes_value(val: str | int | float | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return max(0, int(val))
    s = str(val).strip()
    if not s:
        return None
    units = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$", s)
    if not m:
        try:
            return max(0, int(float(s)))
        except ValueError:
            return None
    number = float(m.group(1))
    unit = m.group(2).lower()
    mult = units.get(unit, 1)
    return max(0, int(number * mult))


def get_configured_max_ram_bytes() -> int | None:
    raw = os.getenv("MOE_MAX_RAM_BYTES") or os.getenv("MOE_MAX_RAM") or os.getenv("MOE_MAX_RAM_MB")
    if not raw:
        return None
    if raw.strip().isdigit() and not any(c in raw.lower() for c in ("g", "m", "k", "t")):
        if os.getenv("MOE_MAX_RAM_BYTES"):
            return _parse_bytes_value(raw)
        return _parse_bytes_value(f"{raw}MB")
    return _parse_bytes_value(raw)


def set_configured_max_ram(val: str | int | float | None) -> int | None:
    """Set runtime max RAM limit and update environment variable."""
    if val is None or (isinstance(val, str) and not val.strip()):
        os.environ.pop("MOE_MAX_RAM", None)
        os.environ.pop("MOE_MAX_RAM_BYTES", None)
        os.environ.pop("MOE_MAX_RAM_MB", None)
        return None
    parsed = _parse_bytes_value(val)
    if parsed is not None:
        os.environ["MOE_MAX_RAM"] = str(val).strip()
    return parsed


def safe_ram_budget() -> int:
    """Bound cache/model working-set budget while preserving a configurable OS reserve and max limit.

    Default reserve is 35% of physical RAM for stability. Override
    with MOE_RAM_RESERVE_PERCENT (5..60) when the host has a different workload.
    Can also be capped with MOE_MAX_RAM / MOE_MAX_RAM_BYTES (e.g. '16GB', '8192MB') to never exceed a maximum.
    mmap-backed GGUF pages remain reclaimable by the Linux page cache.
    """
    try:
        reserve_percent = int(os.getenv("MOE_RAM_RESERVE_PERCENT", "35"))
    except ValueError:
        reserve_percent = 35
    reserve_percent = min(60, max(5, reserve_percent))
    max_ram = get_configured_max_ram_bytes()
    if psutil:
        memory = psutil.virtual_memory()
        reserve = int(memory.total * reserve_percent / 100)
        ceiling = int(memory.total * (100 - reserve_percent) / 100)
        calc_budget = max(0, min(ceiling, int(memory.available) - reserve))
        if max_ram is not None:
            calc_budget = max(0, min(calc_budget, max_ram))
        return calc_budget
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = page_size * os.sysconf("SC_PHYS_PAGES")
    available = page_size * os.sysconf("SC_AVPHYS_PAGES")
    reserve = int(total * reserve_percent / 100)
    ceiling = int(total * (100 - reserve_percent) / 100)
    calc_budget = max(0, min(ceiling, available - reserve))
    if max_ram is not None:
        calc_budget = max(0, min(calc_budget, max_ram))
    return calc_budget


@dataclass
class Entry:
    data: object
    refs: int = 0


class ExpertCache:
    def __init__(self, manifest_path: str, telemetry_path: str, budget_bytes: int | None = None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.telemetry_path = Path(telemetry_path).expanduser().resolve()
        safe_budget = safe_ram_budget()
        self.budget_bytes = max(0, min(budget_bytes if budget_bytes is not None else safe_budget, safe_budget))
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.stream_refs: dict[str, int] = {}
        self.resident_bytes = 0
        self.lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self._handles_lock = threading.Lock()
        self._open_handles: dict[Path, tuple[int, int, int]] = {}
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}
        self.invocations: dict[str, int] = {}
        self.last_used: dict[str, float] = {}
        self.load_time_ms: dict[str, float] = {}
        self.stream_time_ms: dict[str, float] = {}
        self.stream_bytes: dict[str, int] = {}
        self.ram_read_time_ms: float = 0.0
        self.ram_bytes_read: int = 0
        self.ram_read_ops: int = 0
        self.transitions: dict[str, dict[str, int]] = {}
        self.manifest: dict[str, dict] = {}
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.manifest = raw.get("experts", raw)
        # Default to dynamic 80% VRAM allocation fraction with safety headroom
        # so MoE expert staging auto-adjusts to whatever GPU hardware is running.
        self.vram_cache = VramExpertCache(
            headroom_mb=int(os.getenv("MOE_VRAM_HEADROOM_MB", "120")),
            max_fraction=float(os.getenv("MOE_VRAM_CACHE_FRACTION", "0.80")),
            promote_after_hits=int(os.getenv("MOE_VRAM_PROMOTE_HITS", "2")),
            promotion_cooldown_seconds=float(os.getenv("MOE_VRAM_PROMOTION_COOLDOWN_SECONDS", "2")),
        )
        self.vram_cache.on_demote = self._demote_vram_expert_to_ram
        self.queue_engine = ExpertExecutionQueue(
            vram_cache=self.vram_cache,
            ram_cache=self,
            max_vram_slots=int(os.getenv("MOE_VRAM_TABLE_CAPACITY")) if os.getenv("MOE_VRAM_TABLE_CAPACITY") else None,
            max_vram_line_length=int(os.getenv("MOE_VRAM_QUEUE_DEPTH", "8")),
            hot_threshold_calls=int(os.getenv("MOE_HOT_PROMOTE_HITS", "3")),
            vram_target_fraction=0.80,
        )
        self._load_telemetry()

    def _load_telemetry(self) -> None:
        try:
            data = json.loads(self.telemetry_path.read_text(encoding="utf-8"))
            self.hits = {str(k): max(0, int(v)) for k, v in data.get("hits", {}).items()}
            self.misses = {str(k): max(0, int(v)) for k, v in data.get("misses", {}).items()}
            self.invocations = {str(k): max(0, int(v)) for k, v in data.get("invocations", {}).items()}
            self.last_used = {str(k): float(v) for k, v in data.get("last_used", {}).items()}
            self.load_time_ms = {str(k): max(0.0, float(v))
                                 for k, v in data.get("load_time_ms", {}).items()}
            self.stream_time_ms = {str(k): max(0.0, float(v))
                                   for k, v in data.get("stream_time_ms", {}).items()}
            self.stream_bytes = {str(k): max(0, int(v))
                                 for k, v in data.get("stream_bytes", {}).items()}
            self.transitions = {str(source): {str(target): max(0, int(count))
                               for target, count in targets.items()}
                               for source, targets in data.get("transitions", {}).items()}
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, json.JSONDecodeError):
            # A corrupt profile should not prevent serving requests.
            self.hits, self.misses, self.last_used = {}, {}, {}
            self.invocations = {}
            self.load_time_ms, self.stream_time_ms, self.stream_bytes = {}, {}, {}
            self.transitions = {}

    def contains(self, key: str) -> bool:
        """Check if an expert is currently resident in RAM cache."""
        with self._state_lock:
            return str(key) in self.entries

    def fetch_from_storage(self, key: str) -> bool:
        """Stage an expert directly from SSD storage into RAM workstation before work starts."""
        k = str(key)
        with self._state_lock:
            if k in self.entries:
                return True
            try:
                _, _, size = self._entry_spec(k)
                budget = self._effective_budget_locked()
                if size <= 0 or size > budget:
                    return False
                while self.resident_bytes + size > budget:
                    if not self._evict_one():
                        return False
                data = self._read_entry(k)
                if len(data) != size:
                    return False
                self.entries[k] = Entry(data)
                self.resident_bytes += size
                return True
            except Exception:
                return False

    def get_copy(self, key: str):
        """Retrieve copy/buffer of weights from RAM for computation."""
        k = str(key)
        with self._state_lock:
            entry = self.entries.get(k)
            return entry.data if entry is not None else None

    def observe_hit(self, key: str) -> None:
        """Record hit / frequency in RAM cache."""
        k = str(key)
        with self._state_lock:
            self.hits[k] = self.hits.get(k, 0) + 1
            self.last_used[k] = time.time()
            if k in self.entries:
                self.entries.move_to_end(k)

    def _entry_spec(self, expert_id: str) -> tuple[Path, int, int]:
        try:
            item = self.manifest[expert_id]
            segments = item.get("segments")
            if segments:
                first = segments[0]
                path = Path(first["path"])
                if not path.is_absolute():
                    path = self.manifest_path.parent / path
                offset = int(first["offset"])
                size = sum(int(segment["size"]) for segment in segments)
            else:
                path = Path(item["path"])
                if not path.is_absolute():
                    path = self.manifest_path.parent / path
                offset, size = int(item.get("offset", 0)), int(item["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid or missing expert manifest entry: {expert_id}") from exc
        if offset < 0 or size < 0:
            raise ValueError(f"negative offset or size for expert {expert_id}")
        return path.resolve(), offset, size

    def _entry_segments(self, expert_id: str) -> list[tuple[Path, int, int]]:
        _, _, total = self._entry_spec(expert_id)
        item = self.manifest[expert_id]
        raw_segments = item.get("segments") or [{"path": item["path"],
                                                  "offset": item.get("offset", 0), "size": item["size"]}]
        result = []
        for segment in raw_segments:
            path = Path(segment["path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            offset, size = int(segment["offset"]), int(segment["size"])
            if offset < 0 or size < 0:
                raise ValueError(f"negative offset or size for expert {expert_id}")
            result.append((path.resolve(), offset, size))
        if sum(size for _, _, size in result) != total:
            raise ValueError(f"segment sizes do not match manifest size for expert {expert_id}")
        return result

    def _get_or_open_handle(self, path: Path) -> int:
        stat = path.stat()
        with self._handles_lock:
            cached = self._open_handles.get(path)
            if cached and cached[1] == stat.st_mtime_ns and cached[2] == stat.st_size:
                return cached[0]
            if cached:
                try:
                    pager.close(cached[0])
                except Exception:
                    pass
            handle = pager.open(str(path))
            self._open_handles[path] = (handle, stat.st_mtime_ns, stat.st_size)
            return handle

    def _close_cached_handle(self, path: Path) -> None:
        with self._handles_lock:
            cached = self._open_handles.pop(path, None)
        if cached:
            try:
                pager.close(cached[0])
            except Exception:
                pass

    def _read_entry(self, expert_id: str):
        segments = self._entry_segments(expert_id)
        size = sum(length for _, _, length in segments)
        data = np.empty(size, dtype=np.uint8)
        cursor = 0
        idx = 0
        while idx < len(segments):
            path = segments[idx][0]
            batch: list[tuple[int, int]] = []
            batch_bytes = 0
            while idx < len(segments) and segments[idx][0] == path:
                _, offset, length = segments[idx]
                batch.append((offset, length))
                batch_bytes += length
                idx += 1
            handle = self._get_or_open_handle(path)
            if len(batch) == 1:
                pager.pread_into(handle, batch[0][0], data[cursor:cursor + batch_bytes])
            else:
                pager.pread_segments_into(handle, batch, data[cursor:cursor + batch_bytes])
            cursor += batch_bytes
        return data

    def _effective_budget_locked(self) -> int:
        # safe_ram_budget() is based on currently available RAM; add this cache's resident bytes
        # back so the cap stays stable as our own pages become resident, while reacting to outside pressure.
        return max(0, min(self.budget_bytes, safe_ram_budget() + self.resident_bytes))

    async def rebalance(self) -> dict:
        async with self.lock:
            with self._state_lock:
                limit = self._effective_budget_locked()
                freed = 0
                while self.resident_bytes > limit:
                    before = self.resident_bytes
                    if not self._evict_one():
                        break
                    freed += before - self.resident_bytes
                return {"budget_bytes": limit, "resident_bytes": self.resident_bytes,
                        "freed_bytes": freed, "pinned_over_budget": self.resident_bytes > limit}

    async def prefetch(self, expert_id: str) -> bool:
        """Load a predicted block into RAM without counting it as a model access."""
        key = str(expert_id)
        async with self.lock:
            with self._state_lock:
                if key in self.entries:
                    started = time.perf_counter()
                    entry = self.entries[key]
                    self.entries.move_to_end(key)
                    elapsed_ms = max(0.001, (time.perf_counter() - started) * 1000.0)
                    self.ram_read_time_ms += elapsed_ms
                    self.ram_bytes_read += len(entry.data)
                    self.ram_read_ops += 1
                    return True
                _, _, size = self._entry_spec(key)
                budget = self._effective_budget_locked()
                if size <= 0 or size > budget:
                    return False
                while self.resident_bytes + size > budget:
                    if not self._evict_one():
                        return False
            read_started = time.perf_counter()
            data = await asyncio.to_thread(self._read_entry, key)
            elapsed_ms = (time.perf_counter() - read_started) * 1000
            if len(data) != size:
                raise OSError(f"short read while prefetching expert {key}")
            with self._state_lock:
                self.load_time_ms[key] = self.load_time_ms.get(key, 0.0) + elapsed_ms
                if key in self.entries:
                    self.entries.move_to_end(key)
                    return True
                budget = self._effective_budget_locked()
                while self.resident_bytes + size > budget:
                    if not self._evict_one():
                        return False
                self.entries[key] = Entry(data)
                self.resident_bytes += size
                return True

    def _demote_vram_expert_to_ram(self, expert_id: str, size: int) -> bool:
        """Keep a displaced hot CUDA expert in RAM, evicting cold RAM blocks first."""
        key = str(expert_id)
        with self._state_lock:
            if key in self.entries:
                return True
            try:
                budget = self._effective_budget_locked()
                if size <= 0 or size > budget:
                    return False
                while self.resident_bytes + size > budget:
                    if not self._evict_one():
                        return False
                data = self._read_entry(key)
                if len(data) != size:
                    return False
                self.entries[key] = Entry(data)
                self.resident_bytes += size
                return True
            except (OSError, ValueError, MemoryError, KeyError):
                return False

    def _try_stage_ssd_hot_expert(self, expert_id: str, size: int, hit_count: int) -> bool:
        """Direct SSD -> VRAM staging via a temporary host buffer, only with safe RAM headroom."""
        if hit_count < self.vram_cache.promote_after_hits or size <= 0:
            return False
        if self.vram_cache.contains(expert_id):
            return True
        with self._state_lock:
            if expert_id in self.entries:
                return False
        if size > safe_ram_budget():
            return False
        # observe() owns VRAM eviction and rechecks free-memory headroom after
        # every eviction; its initial budget is not the whole capacity.
        temporary = None
        try:
            temporary = self._read_entry(expert_id)
            if len(temporary) != size:
                return False
            return self.vram_cache.observe(expert_id, temporary, hit_count, source="ssd")
        except (OSError, ValueError, RuntimeError, MemoryError):
            return False
        finally:
            if temporary is not None:
                del temporary

    async def record_transition(self, source: str, target: str) -> None:
        async with self.lock:
            with self._state_lock:
                row = self.transitions.setdefault(str(source), {})
                row[str(target)] = row.get(str(target), 0) + 1

    @asynccontextmanager
    async def pin_stream(self, expert_id: str):
        key = str(expert_id)
        async with self.lock:
            with self._state_lock:
                self.stream_refs[key] = self.stream_refs.get(key, 0) + 1
        try:
            yield
        finally:
            async with self.lock:
                with self._state_lock:
                    remaining = self.stream_refs.get(key, 0) - 1
                    if remaining <= 0:
                        self.stream_refs.pop(key, None)
                    else:
                        self.stream_refs[key] = remaining

    async def stream_expert(self, expert_id: str, chunk_bytes: int = 64 * 1024 * 1024):
        """Stream a large expert in bounded NumPy chunks when it cannot fit in RAM cache."""
        segments = self._entry_segments(str(expert_id))
        size = sum(length for _, _, length in segments)
        with self._state_lock:
            safe_chunk = min(max(1, int(chunk_bytes)), max(1, self._effective_budget_locked()))
            pager.record_expert(str(expert_id))
            self.invocations[str(expert_id)] = self.invocations.get(str(expert_id), 0) + 1
            invocation_count = self.invocations[str(expert_id)]
        # If the RAM tier cannot retain this expert, a hot block may still be
        # staged directly SSD -> VRAM when both VRAM and temporary host headroom fit.
        # The actual request remains SSD-paged until an expert-aware backend consumes it.
        await asyncio.to_thread(self._try_stage_ssd_hot_expert, str(expert_id), size, invocation_count)
        logical_offset = 0
        for path, offset, segment_size in segments:
            handle = await asyncio.to_thread(self._get_or_open_handle, path)
            file_size = await asyncio.to_thread(pager.file_size, handle)
            if offset > file_size or segment_size > file_size - offset:
                raise ValueError(f"expert {expert_id} range exceeds file bounds")
            for start in range(0, segment_size, safe_chunk):
                count = min(safe_chunk, segment_size - start)
                read_started = time.perf_counter()
                chunk = await asyncio.to_thread(pager.pread_array, handle, offset + start, count)
                elapsed_ms = (time.perf_counter() - read_started) * 1000
                with self._state_lock:
                    self.stream_time_ms[str(expert_id)] = self.stream_time_ms.get(str(expert_id), 0.0) + elapsed_ms
                    self.stream_bytes[str(expert_id)] = self.stream_bytes.get(str(expert_id), 0) + count
                yield logical_offset + start, chunk
            logical_offset += segment_size

    def _keys_for_file(self, model_path: Path) -> list[str]:
        matching = []
        with self._state_lock:
            all_keys = set(self.manifest) | set(self.entries) | set(self.stream_refs)
        for expert_id in all_keys:
            try:
                paths = [path for path, _, _ in self._entry_segments(expert_id)]
            except (ValueError, OSError):
                continue
            if model_path in paths:
                matching.append(expert_id)
        return matching

    async def prepare_for_model_load(self, model_path: str | Path, needed_bytes: int = 0) -> dict:
        """Evict cached experts belonging to other models so RAM is freed before model load."""
        target = Path(model_path).expanduser().resolve()
        freed_ram = 0
        freed_vram = 0
        other_paths: set[Path] = set()
        async with self.lock:
            with self._state_lock:
                keep_keys = set(self._keys_for_file(target))
                stale_keys = [key for key in list(self.entries) if key not in keep_keys]
                for key in stale_keys:
                    entry = self.entries.get(key)
                    if entry is not None and entry.refs == 0:
                        try:
                            for p, _, _ in self._entry_segments(key):
                                if p != target:
                                    other_paths.add(p)
                        except Exception:
                            pass
                        freed_vram += self.vram_cache.unload_keys({key})
                        popped = self.entries.pop(key, None)
                        if popped is not None:
                            freed_ram += len(popped.data)
                            self.resident_bytes = max(0, self.resident_bytes - len(popped.data))
                if needed_bytes > 0:
                    while self.resident_bytes > max(0, safe_ram_budget() - needed_bytes):
                        before = self.resident_bytes
                        if not self._evict_one():
                            break
                        freed_ram += max(0, before - self.resident_bytes)
        for p in other_paths:
            self._close_cached_handle(p)
            await asyncio.to_thread(_drop_os_page_cache, p)
        if freed_ram > 0 or freed_vram > 0:
            await asyncio.to_thread(_trim_process_heap)
        return {"freed_ram_bytes": freed_ram, "freed_vram_bytes": freed_vram,
                "resident_bytes": self.resident_bytes}

    async def unload_file(self, model_path: str, timeout_seconds: float = 10.0,
                          drop_page_cache: bool = True) -> dict:
        """Evict all cached blocks for a GGUF, close handles, and drop OS page cache."""
        target = Path(model_path).expanduser().resolve()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        ram_before = psutil.virtual_memory().available if psutil else None
        freed_vram = 0
        while True:
            async with self.lock:
                with self._state_lock:
                    matching = self._keys_for_file(target)
                    busy = []
                    for key in matching:
                        entry = self.entries.get(key)
                        device_entry = self.vram_cache.entries.get(key)
                        if ((entry is not None and entry.refs > 0) or self.stream_refs.get(key, 0) > 0
                                or (device_entry is not None and device_entry.refs > 0)):
                            busy.append(key)
                    if not busy:
                        freed = 0
                        freed_vram = self.vram_cache.unload_keys(set(matching), force=True)
                        for key in matching:
                            entry = self.entries.pop(key, None)
                            if entry is None:
                                continue
                            freed += len(entry.data)
                            self.resident_bytes = max(0, self.resident_bytes - len(entry.data))
                        for key in matching:
                            self.manifest.pop(key, None)
                        self._close_cached_handle(target)
                        break
            if time.monotonic() >= deadline:
                return {"unloaded": False, "model_path": str(target), "blocked_by_active_experts": busy,
                        "freed_bytes": 0, "freed_vram_bytes": 0}
            await asyncio.sleep(0.025)
        page_cache_dropped = False
        if drop_page_cache:
            page_cache_dropped = await asyncio.to_thread(_drop_os_page_cache, target)
        heap_trimmed = await asyncio.to_thread(_trim_process_heap)
        ram_after = psutil.virtual_memory().available if psutil else None
        system_ram_reclaimed = max(0, int(ram_after - ram_before)) if (ram_before is not None and ram_after is not None) else freed
        async with self.lock:
            with self._state_lock:
                remaining = [key for key in self._keys_for_file(target)
                             if key in self.entries or key in self.stream_refs or key in self.manifest]
                return {"unloaded": not remaining, "model_path": str(target), "freed_bytes": freed,
                        "freed_vram_bytes": freed_vram,
                        "system_ram_reclaimed_bytes": system_ram_reclaimed,
                        "page_cache_dropped": page_cache_dropped,
                        "heap_trimmed": heap_trimmed,
                        "ram_cache_bytes_remaining": sum(len(self.entries[key].data) for key in remaining if key in self.entries),
                        "remaining_experts": remaining, "manifest_entries_removed": len(matching)}

    async def drop_all(self, drop_page_cache: bool = True) -> dict:
        """Force-drop all RAM and VRAM expert caches, close all file handles, and flush OS page cache."""
        ram_before = psutil.virtual_memory().available if psutil else None
        known_files: set[Path] = set()
        async with self.lock:
            with self._state_lock:
                for key in list(self.manifest):
                    try:
                        for p, _, _ in self._entry_segments(key):
                            known_files.add(p)
                    except Exception:
                        pass
                with self._handles_lock:
                    known_files.update(self._open_handles.keys())
                freed_ram = sum(len(entry.data) for entry in self.entries.values())
                self.entries.clear()
                self.stream_refs.clear()
                self.resident_bytes = 0
                freed_vram = self.vram_cache.clear(force=True)
                self.close_handles()
        dropped_files = 0
        if drop_page_cache:
            for file_path in known_files:
                if await asyncio.to_thread(_drop_os_page_cache, file_path):
                    dropped_files += 1
        heap_trimmed = await asyncio.to_thread(_trim_process_heap)
        ram_after = psutil.virtual_memory().available if psutil else None
        system_ram_reclaimed = max(0, int(ram_after - ram_before)) if (ram_before is not None and ram_after is not None) else freed_ram
        return {
            "dropped": True,
            "freed_ram_cache_bytes": freed_ram,
            "freed_vram_bytes": freed_vram,
            "system_ram_reclaimed_bytes": system_ram_reclaimed,
            "page_cache_files_dropped": dropped_files,
            "heap_trimmed": heap_trimmed,
            "ram_cache_bytes_remaining": self.resident_bytes,
        }

    def _evict_one(self) -> bool:
        with self._state_lock:
            # A hot expert currently in VRAM must retain its RAM backing. If RAM is
            # pressured, demote a cold/less-useful RAM entry to SSD first. When the
            # VRAM entry is later displaced, its existing RAM copy is already ready.
            vram_resident = set(self.vram_cache.entries)
            candidates = [(self.hits.get(key, 0), self.invocations.get(key, 0),
                           self.last_used.get(key, 0.0), index, key, entry)
                          for index, (key, entry) in enumerate(self.entries.items())
                          if entry.refs == 0 and key not in vram_resident]
            if not candidates:
                return False
            # Lowest observed frequency/invocations first; oldest breaks ties.
            _hits, _invocations, _last_used, _index, key, entry = min(candidates)
            self.resident_bytes -= len(entry.data)
            del self.entries[key]
            return True

    def _measure_ram_read(self, data) -> float:
        """Calibrate DDR read time once per slice size (warm, pre-faulted); O(1) afterwards."""
        import numpy as np
        size = len(data)
        cal = self.__dict__.setdefault("_calibrated_ram_ms", {})
        if size in cal:
            return cal[size]
        native_ms = pager.measure_ram_bandwidth(data)
        if native_ms > 0:
            cal[size] = max(0.0005, native_ms)
            return cal[size]
        src = np.frombuffer(data, dtype=np.uint8)
        scratch = np.zeros(src.size, dtype=np.uint8)  # pre-faulted
        np.copyto(scratch, src)  # warm-up
        passes = 4
        started = time.perf_counter()
        for _ in range(passes):
            np.copyto(scratch, src)
        elapsed = (time.perf_counter() - started) * 1000.0 / passes
        # copy moves read+write; report the read half as the bus read time
        cal[size] = max(0.0005, elapsed / 2.0)
        return cal[size]

    @asynccontextmanager
    async def acquire(self, expert_id: str) -> AsyncIterator[bytes]:
        key = str(expert_id)
        async with self.lock:
            with self._state_lock:
                budget = self._effective_budget_locked()
                while self.resident_bytes > budget:
                    if not self._evict_one():
                        break
                entry = self.entries.get(key)
                if entry is not None:
                    hit_started = time.perf_counter()
                    self.hits[key] = self.hits.get(key, 0) + 1
                    self.entries.move_to_end(key)
                    entry.refs += 1
                    self.last_used[key] = time.time()
                    pager.record_expert(key)
                    self.invocations[key] = self.invocations.get(key, 0) + 1
                    invocation_count = self.invocations[key]
                    need_read = False
                    size = len(entry.data)
                    hit_ms = self._measure_ram_read(entry.data)
                    self.ram_read_time_ms += hit_ms
                    self.ram_bytes_read += size
                    self.ram_read_ops += 1
                else:
                    self.misses[key] = self.misses.get(key, 0) + 1
                    _, _, size = self._entry_spec(key)
                    budget = self._effective_budget_locked()
                    if size > budget:
                        raise MemoryError(f"expert {key} ({size} bytes) exceeds RAM cache budget ({budget} bytes); use SSD chunk paging")
                    while self.resident_bytes + size > budget:
                        if not self._evict_one():
                            raise MemoryError("RAM cache is full of in-use expert blocks")
                    need_read = True
            if need_read:
                read_started = time.perf_counter()
                data = await asyncio.to_thread(self._read_entry, key)
                elapsed_ms = (time.perf_counter() - read_started) * 1000
                if len(data) != size:
                    raise OSError(f"short read for expert {key}: expected {size}, got {len(data)}")
                with self._state_lock:
                    self.load_time_ms[key] = self.load_time_ms.get(key, 0.0) + elapsed_ms
                    while self.resident_bytes + size > self._effective_budget_locked():
                        if not self._evict_one():
                            raise MemoryError("RAM cache is full of in-use expert blocks")
                    entry = Entry(data, refs=1)
                    self.entries[key] = entry
                    self.resident_bytes += len(data)
                    self.last_used[key] = time.time()
                    pager.record_expert(key)
                    self.invocations[key] = self.invocations.get(key, 0) + 1
                    invocation_count = self.invocations[key]
                    self.ram_read_time_ms += self._measure_ram_read(data)
                    self.ram_bytes_read += len(data)
                    self.ram_read_ops += 1
        # GPU promotion is best-effort and never blocks or breaks the RAM/SSD path.
        await asyncio.to_thread(self.vram_cache.observe, key, entry.data, invocation_count)
        try:
            yield entry.data
        finally:
            async with self.lock:
                with self._state_lock:
                    entry.refs -= 1

    def close_handles(self) -> None:
        with self._handles_lock:
            items = list(self._open_handles.values())
            self._open_handles.clear()
        for handle, _, _ in items:
            try:
                pager.close(handle)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close_handles()
        except Exception:
            pass

    async def save_telemetry(self) -> None:
        async with self.lock:
            with self._state_lock:
                self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"version": 2, "updated_at": time.time(), "hits": dict(self.hits),
                           "misses": dict(self.misses), "invocations": dict(self.invocations),
                           "last_used": dict(self.last_used),
                           "load_time_ms": dict(self.load_time_ms), "stream_time_ms": dict(self.stream_time_ms),
                           "stream_bytes": dict(self.stream_bytes),
                           "transitions": {k: dict(v) for k, v in self.transitions.items()}}
            fd, temp_name = tempfile.mkstemp(prefix=".expert_telemetry.", dir=self.telemetry_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, self.telemetry_path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    async def import_benchmark(self, payload: dict) -> None:
        """Merge a benchmark profile containing hit counts into the live profile."""
        raw_hits = payload.get("hits", payload.get("expert_hits", {}))
        if isinstance(raw_hits, list):
            counts = {}
            for row in raw_hits:
                if not isinstance(row, dict) or "expert_id" not in row:
                    raise ValueError("benchmark hit rows need expert_id and optional hit_count")
                key = str(row["expert_id"])
                counts[key] = counts.get(key, 0) + max(0, int(row.get("hit_count", row.get("count", 1))))
        elif isinstance(raw_hits, dict):
            counts = {str(key): max(0, int(value)) for key, value in raw_hits.items()}
        else:
            raise ValueError("benchmark must contain an object or array of expert hit counts")
        async with self.lock:
            with self._state_lock:
                for key, value in counts.items():
                    self.hits[key] = self.hits.get(key, 0) + value
                    self.last_used[key] = time.time()
                raw_transitions = payload.get("transitions", {})
                if isinstance(raw_transitions, dict):
                    for source, targets in raw_transitions.items():
                        if isinstance(targets, dict):
                            row = self.transitions.setdefault(str(source), {})
                            for target, value in targets.items():
                                row[str(target)] = row.get(str(target), 0) + max(0, int(value))
        await self.save_telemetry()

    async def snapshot(self) -> dict:
        async with self.lock:
            with self._state_lock:
                vram = self.vram_cache.snapshot()
                ssd_metrics = pager.metrics()
                vram_ids = set(vram.get("cached_experts", []))
                ram_ids = set(self.entries)
                active_ids = ({key for key, value in self.entries.items() if value.refs > 0} |
                              {key for key, refs in self.stream_refs.items() if refs > 0})
                known_ids = set(self.manifest) | ram_ids | vram_ids
                hot_ids = {key for key in known_ids
                           if self.invocations.get(key, 0) >= self.vram_cache.promote_after_hits}
                ram_speed_mbs = ((self.ram_bytes_read / (1024.0 * 1024.0)) / (self.ram_read_time_ms / 1000.0)) if self.ram_read_time_ms > 0 else 0.0
                try:
                    reserve_pct = min(60, max(5, int(os.getenv("MOE_RAM_RESERVE_PERCENT", "35"))))
                except ValueError:
                    reserve_pct = 10
                max_ram_val = get_configured_max_ram_bytes()
                return {"ram_budget_bytes": self._effective_budget_locked(), "resident_bytes": self.resident_bytes,
                        "max_ram_bytes": max_ram_val,
                        "cached_experts": list(self.entries), "hits": dict(self.hits),
                        "invocations": dict(self.invocations),
                        "load_time_ms": dict(self.load_time_ms),
                        "stream_time_ms": dict(self.stream_time_ms),
                        "stream_bytes": dict(self.stream_bytes),
                        "ram_read_time_ms": round(self.ram_read_time_ms, 3),
                        "ram_bytes_read": self.ram_bytes_read,
                        "ram_read_ops": self.ram_read_ops,
                        "ram_read_speed_mbs": round(ram_speed_mbs, 2),
                        "active_experts": sorted(active_ids),
                        "misses": dict(self.misses), "native_pager": pager.native,
                        "tiers": {"vram": sorted(vram_ids), "ram": sorted(ram_ids - vram_ids),
                                  "ssd": sorted(known_ids - ram_ids - vram_ids),
                                  "hot_experts": sorted(hot_ids),
                                  "hot_on_ssd": sorted(hot_ids - ram_ids - vram_ids),
                                  "hot_vram_backed_by_ram": sorted(hot_ids & vram_ids & ram_ids),
                                  "active_pinned": sorted(active_ids)},
                        "tier_io": {
                            "vram": {
                                "read_time_ms": float(vram.get("read_time_ms") or 0.0),
                                "read_bytes": int(vram.get("bytes_read") or 0),
                                "read_ops": int(vram.get("read_ops") or 0),
                                "read_speed_mbs": float(vram.get("read_speed_mbs") or 0.0),
                            },
                            "ram": {
                                "read_time_ms": round(self.ram_read_time_ms, 3),
                                "read_bytes": self.ram_bytes_read,
                                "read_ops": self.ram_read_ops,
                                "read_speed_mbs": round(ram_speed_mbs, 2),
                            },
                            "ssd": {
                                "read_time_ms": round(float(ssd_metrics.get("time_used_for_reading_ms") or 0.0), 3),
                                "read_bytes": int(ssd_metrics.get("bytes_read") or 0),
                                "read_ops": int(ssd_metrics.get("read_ops") or 0),
                                "read_speed_mbs": round(float(ssd_metrics.get("actual_ssd_throughput_mbs") or 0.0), 2),
                            },
                        },
                        "policy": {"priority": ["VRAM", "RAM", "SSD"],
                                   "protect_vram_hot_in_ram": True,
                                   "ram_reserve_percent": reserve_pct,
                                   "max_ram_bytes": max_ram_val},
                        "vram": vram}
