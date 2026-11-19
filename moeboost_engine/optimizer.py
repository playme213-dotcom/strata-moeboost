"""Memory-aware candidate generation, ranking, and VRAM/RAM/SSD tier optimization."""
from __future__ import annotations

import asyncio
import ctypes
import gc
import os
import time
from pathlib import Path
from typing import Any


def _format_duration_ms(ms: float | int | None) -> str:
    if ms is None:
        return "—"
    value_ms = max(0.0, float(ms))
    if value_ms == 0.0:
        return "0.00s"
    total_seconds = value_ms / 1000.0
    if total_seconds >= 3600.0:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60.0
        return f"{hours}h {minutes}m {seconds:.1f}s"
    if total_seconds >= 60.0:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60.0
        return f"{minutes}m {seconds:.1f}s"
    if total_seconds >= 1.0:
        return f"{total_seconds:.2f}s"
    return f"{total_seconds:.2f}s ({value_ms:.1f} ms)"


def _format_size_gb_mb(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "—"
    value = max(0.0, float(num_bytes))
    if value >= 1024.0 ** 3:
        return f"{value / (1024.0 ** 3):.2f} GB"
    if value == 0.0:
        return "0.00 MB"
    mb = value / (1024.0 ** 2)
    return f"{mb:.4f} MB" if mb < 0.01 else f"{mb:.2f} MB"


def _format_speed_mbs(mbs: float | int | None) -> str:
    if mbs is None:
        return "0.00 MB/s"
    value = max(0.0, float(mbs))
    return f"{value / 1024.0:.2f} GB/s" if value >= 1024.0 else f"{value:.2f} MB/s"


def snapshot_tier_io(cache: Any, pager_obj: Any) -> dict[str, dict[str, Any]]:
    """Collect a consistent tier_io snapshot across VRAM, RAM, and SSD."""
    vram_data = cache.vram_cache.snapshot()
    ssd_data = pager_obj.metrics()

    vram_ms = float(vram_data.get("read_time_ms") or 0.0)
    vram_bytes = int(vram_data.get("bytes_read") or 0)
    vram_ops = int(vram_data.get("read_ops") or 0)
    vram_speed = float(vram_data.get("read_speed_mbs") or 0.0)

    ram_ms = float(getattr(cache, "ram_read_time_ms", 0.0) or 0.0)
    ram_bytes = int(getattr(cache, "ram_bytes_read", 0) or 0)
    ram_ops = int(getattr(cache, "ram_read_ops", 0) or 0)
    ram_speed = ((ram_bytes / (1024.0 * 1024.0)) / (ram_ms / 1000.0)) if ram_ms > 0 else 0.0

    ssd_ms = float(ssd_data.get("time_used_for_reading_ms") or 0.0)
    ssd_bytes = int(ssd_data.get("bytes_read") or 0)
    ssd_ops = int(ssd_data.get("read_ops") or 0)
    ssd_speed = float(ssd_data.get("actual_ssd_throughput_mbs") or 0.0)

    return {
        "vram": {
            "read_time_ms": round(vram_ms, 3),
            "read_time_seconds": round(vram_ms / 1000.0, 4),
            "read_time_formatted": _format_duration_ms(vram_ms),
            "read_bytes": vram_bytes,
            "read_size_formatted": _format_size_gb_mb(vram_bytes),
            "read_ops": vram_ops,
            "read_speed_mbs": round(vram_speed, 2),
            "read_speed_formatted": _format_speed_mbs(vram_speed),
        },
        "ram": {
            "read_time_ms": round(ram_ms, 3),
            "read_time_seconds": round(ram_ms / 1000.0, 4),
            "read_time_formatted": _format_duration_ms(ram_ms),
            "read_bytes": ram_bytes,
            "read_size_formatted": _format_size_gb_mb(ram_bytes),
            "read_ops": ram_ops,
            "read_speed_mbs": round(ram_speed, 2),
            "read_speed_formatted": _format_speed_mbs(ram_speed),
        },
        "ssd": {
            "read_time_ms": round(ssd_ms, 3),
            "read_time_seconds": round(ssd_ms / 1000.0, 4),
            "read_time_formatted": _format_duration_ms(ssd_ms),
            "read_bytes": ssd_bytes,
            "read_size_formatted": _format_size_gb_mb(ssd_bytes),
            "read_ops": ssd_ops,
            "read_speed_mbs": round(ssd_speed, 2),
            "read_speed_formatted": _format_speed_mbs(ssd_speed),
        },
    }


def _posix_advise_path(path: Path) -> bool:
    fadv_seq = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
    fadv_will = getattr(os, "POSIX_FADV_WILLNEED", None)
    if not hasattr(os, "posix_fadvise") or fadv_seq is None or fadv_will is None:
        return False
    fd = -1
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            return False
        fd = os.open(str(resolved), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        os.posix_fadvise(fd, 0, 0, fadv_seq)
        os.posix_fadvise(fd, 0, 0, fadv_will)
        return True
    except OSError:
        return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _trim_heap() -> bool:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
            return True
    except Exception:
        pass
    return False


async def optimize_storage_tiers(
    cache: Any,
    pager_obj: Any = None,
    model_paths: list[str | Path] | None = None,
    max_prefetch_experts: int = 32,
    max_prefetch_bytes: int = 256 * 1024 * 1024,
    max_vram_experts: int | None = None,
) -> dict[str, Any]:
    """Optimize VRAM, RAM, and SSD tiers end-to-end and return before/after tier telemetry."""
    import numpy as np
    if pager_obj is None:
        from moe_boost.pager import pager as default_pager
        pager_obj = default_pager

    started = time.perf_counter()
    before = snapshot_tier_io(cache, pager_obj)
    optimizations_applied: list[str] = []

    # 1. SSD optimization: apply POSIX_FADV_SEQUENTIAL + POSIX_FADV_WILLNEED and sort warm queue by (path, offset)
    active_paths: set[Path] = set()
    for p in model_paths or []:
        try:
            rp = Path(p).expanduser().resolve()
            if rp.is_file():
                active_paths.add(rp)
        except OSError:
            pass
    manifest_rows: list[tuple[str, int, int, str, int]] = []
    with cache._state_lock:
        for key in list(cache.manifest):
            try:
                segs = cache._entry_segments(key)
                if not segs:
                    continue
                first_path, first_offset, _ = segs[0]
                total_size = sum(sz for _, _, sz in segs)
                active_paths.add(first_path)
                inv = int(cache.invocations.get(key, 0))
                hits = int(cache.hits.get(key, 0))
                manifest_rows.append((str(first_path), int(first_offset), total_size, key, inv + hits))
            except (ValueError, OSError):
                continue

    advised_fds = 0
    for p in sorted(active_paths):
        try:
            handle = await asyncio.to_thread(cache._get_or_open_handle, p)
            if await asyncio.to_thread(pager_obj.advise_file_sequential, handle):
                advised_fds += 1
            elif await asyncio.to_thread(_posix_advise_path, p):
                advised_fds += 1
        except OSError:
            if await asyncio.to_thread(_posix_advise_path, p):
                advised_fds += 1
    if advised_fds:
        optimizations_applied.append(f"ssd_fadvise_sequential_willneed({advised_fds}_files)")

    # Sort candidate hot experts by physical (path, offset) to maximize sequential NVMe/SSD read throughput.
    manifest_rows.sort(key=lambda row: (row[0], row[1]))
    hot_candidates = [row for row in manifest_rows if row[4] > 0]
    if not hot_candidates and manifest_rows:
        hot_candidates = manifest_rows[:max_prefetch_experts]
    # Keep top candidates by heat, then order them physically by (path, offset) for sequential prefetch.
    if len(hot_candidates) > max_prefetch_experts:
        hot_candidates = sorted(hot_candidates, key=lambda row: row[4], reverse=True)[:max_prefetch_experts]
        hot_candidates.sort(key=lambda row: (row[0], row[1]))

    prefetched_count = 0
    prefetched_bytes = 0
    for _path_str, _offset, sz, key, _score in hot_candidates:
        if prefetched_count >= max_prefetch_experts or prefetched_bytes + sz > max_prefetch_bytes:
            continue
        if key in cache.entries:
            continue
        try:
            if await cache.prefetch(key):
                prefetched_count += 1
                prefetched_bytes += sz
        except Exception:
            continue
    if prefetched_count:
        optimizations_applied.append(f"ssd_sequential_prefetch({prefetched_count}_experts)")

    # 2. RAM optimization: madvise(MADV_HUGEPAGE + MADV_WILLNEED), C++ AVX2 thread sweep, rebalance & malloc_trim
    advised_buffers = 0
    with cache._state_lock:
        resident_items = [(key, entry.data) for key, entry in cache.entries.items()]
    for _key, data in resident_items:
        if await asyncio.to_thread(pager_obj.advise_ram_buffer, data):
            advised_buffers += 1
    if advised_buffers:
        optimizations_applied.append(f"ram_madvise_hugepage_willneed({advised_buffers}_buffers)")

    ram_cal = await asyncio.to_thread(pager_obj.calibrate_ram_bandwidth, [2, 4, 8])
    if "__dict__" in dir(cache):
        cache.__dict__["_calibrated_ram_ms"] = {}
    ram_probe_bytes = 4 * 1024 * 1024
    if resident_items:
        sample_buf = resident_items[0][1]
        await asyncio.to_thread(cache._measure_ram_read, sample_buf)
    else:
        probe = np.empty(ram_probe_bytes, dtype=np.uint8)
        await asyncio.to_thread(pager_obj.measure_ram_bandwidth, probe)
        del probe
    optimizations_applied.append(
        f"ram_avx2_calibrated(threads={ram_cal.get('best_threads', 0)},speed={_format_speed_mbs(ram_cal.get('bandwidth_mbs', 0.0))})"
    )

    rebalance_result = await cache.rebalance()
    heap_trimmed = await asyncio.to_thread(_trim_heap)
    if heap_trimmed:
        optimizations_applied.append("ram_malloc_trim")

    # 3. VRAM optimization: warm up non-blocking stream, pinned DMA + 64MB slab, recalibrate & promote hot experts
    vram_warmup: dict[str, Any] = {}
    vram_calibrated_mbs = 0.0
    vram_promoted = 0
    vram_cache = cache.vram_cache
    if vram_cache.available:
        adapter = vram_cache.adapter
        warm_fn = getattr(adapter, "warm_up_and_preallocate", None)
        if callable(warm_fn):
            vram_warmup = await asyncio.to_thread(warm_fn)
            optimizations_applied.append("vram_nonblocking_stream_pinned_dma_64mb_slab")
        vram_cache.__dict__["_calibrated_read_ms"] = {}

        # Promote hottest resident RAM experts into VRAM
        with cache._state_lock:
            ram_ranked = sorted(
                ((cache.invocations.get(k, 0), cache.hits.get(k, 0), cache.last_used.get(k, 0.0), k, e.data)
                 for k, e in cache.entries.items()),
                reverse=True,
            )
        # Limit VRAM promotions to the top hot experts (defaulting to 1024 so up to ~2.5 GB
        # of GDDR5 holds ~1,000 hot experts, with remaining experts resident in DDR RAM).
        if max_vram_experts is not None:
            max_vram_promoted = max(0, int(max_vram_experts))
        else:
            max_vram_promoted = int(os.getenv("MOE_MAX_VRAM_EXPERTS", "1024"))
        for inv, hits, _lu, key, data in ram_ranked:
            if vram_promoted >= max_vram_promoted:
                break
            effective_hits = max(inv, hits, vram_cache.promote_after_hits)
            if await asyncio.to_thread(vram_cache.observe, key, data, effective_hits, "ram"):
                vram_promoted += 1

        # Recalibrate resident VRAM entries (or a temporary probe if empty) at peak GDDR5 speed
        with vram_cache.lock:
            existing = [(k, e.tensor, e.size_bytes) for k, e in vram_cache.entries.items()]
        if existing:
            _k0, t0, s0 = existing[0]
            cal_ms = await asyncio.to_thread(vram_cache._device_read_ms, t0, s0)
            if cal_ms > 0:
                vram_calibrated_mbs = (s0 / (1024.0 * 1024.0)) / (cal_ms / 1000.0)
        else:
            read_dev_fn = getattr(adapter, "read_device", None)
            if callable(read_dev_fn):
                probe_size = 4 * 1024 * 1024
                try:
                    probe_tensor = await asyncio.to_thread(adapter.upload, np.zeros(probe_size, dtype=np.uint8))
                    cal_ms = await asyncio.to_thread(read_dev_fn, probe_tensor)
                    await asyncio.to_thread(adapter.release, probe_tensor, probe_size)
                    if cal_ms > 0:
                        vram_calibrated_mbs = (probe_size / (1024.0 * 1024.0)) / (cal_ms / 1000.0)
                except Exception:
                    pass
        if vram_calibrated_mbs > 0:
            optimizations_applied.append(f"vram_gddr_calibrated({ _format_speed_mbs(vram_calibrated_mbs) })")
        if vram_promoted:
            optimizations_applied.append(f"vram_hot_promoted({vram_promoted}_experts)")

    after = snapshot_tier_io(cache, pager_obj)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return {
        "status": "optimized",
        "elapsed_ms": elapsed_ms,
        "before": {"tier_io": before},
        "after": {"tier_io": after},
        "optimizations_applied": optimizations_applied,
        "active_settings": {
            "vram": {
                "available": bool(vram_cache.available),
                "non_blocking_stream": bool(vram_warmup.get("non_blocking_stream", False)),
                "pinned_dma_bytes": int(vram_warmup.get("pinned_dma_bytes", 0)),
                "vram_slab_bytes": int(vram_warmup.get("vram_slab_bytes", 0)),
                "calibrated_read_speed_mbs": round(vram_calibrated_mbs, 2),
                "calibrated_read_speed_formatted": _format_speed_mbs(vram_calibrated_mbs),
                "promoted_experts": vram_promoted,
            },
            "ram": {
                "best_threads": int(ram_cal.get("best_threads") or 0),
                "calibrated_read_speed_mbs": round(float(ram_cal.get("bandwidth_mbs") or 0.0), 2),
                "calibrated_read_speed_formatted": _format_speed_mbs(ram_cal.get("bandwidth_mbs") or 0.0),
                "thread_candidates": ram_cal.get("candidates", []),
                "advised_buffers": advised_buffers,
                "heap_trimmed": heap_trimmed,
                "rebalance": rebalance_result,
            },
            "ssd": {
                "advised_files": advised_fds,
                "prefetched_experts": prefetched_count,
                "prefetched_bytes": prefetched_bytes,
                "prefetched_size_formatted": _format_size_gb_mb(prefetched_bytes),
                "manifest_sorted_by_offset": bool(manifest_rows),
            },
        },
    }


def build_candidates(current: dict, max_context: int = 4096) -> list[dict]:
    """Small, bounded sweep; avoids combinatorial benchmark explosions."""
    import os
    logical_cpus = os.cpu_count() or 8
    try:
        import psutil
        physical_cpus = psutil.cpu_count(logical=False) or max(1, logical_cpus // 2)
    except Exception:
        physical_cpus = max(1, logical_cpus // 2)
    base_ctx = min(max_context, max(512, int(current.get("context_size", 2048))))
    base_batch = max(32, int(current.get("batch_size", 512)))
    phys_threads = max(1, min(logical_cpus, int(physical_cpus)))
    logical_threads = max(phys_threads, int(logical_cpus))
    base_threads = max(1, min(logical_cpus, int(current.get("threads", phys_threads))))
    high_threads = max(logical_threads, min(logical_cpus * 2, int(current.get("threads_batch", logical_threads))))
    high_ctx = max(512, min(max_context, 4096))
    mid_ctx = max(512, min(max_context, 2048))
    raw = [
        {"context_size": mid_ctx, "batch_size": 512,
         "threads": phys_threads, "threads_batch": high_threads},
        {"context_size": mid_ctx, "batch_size": 512,
         "threads": logical_threads, "threads_batch": high_threads},
        {"context_size": mid_ctx, "batch_size": 1024,
         "threads": phys_threads, "threads_batch": high_threads},
        {"context_size": mid_ctx, "batch_size": 1024,
         "threads": logical_threads, "threads_batch": high_threads},
        {"context_size": base_ctx, "batch_size": base_batch,
         "threads": base_threads, "threads_batch": high_threads},
        {"context_size": high_ctx, "batch_size": 512,
         "threads": phys_threads, "threads_batch": high_threads},
        {"context_size": high_ctx, "batch_size": 1024,
         "threads": logical_threads, "threads_batch": high_threads},
    ]
    unique = []
    for candidate in raw:
        if candidate not in unique:
            unique.append(candidate)
    return unique[:6]


def score_result(generation_tps: float, prefill_proxy_tps: float) -> float:
    """Favor decode speed while retaining a smaller prompt-processing weight."""
    return round(max(0.0, generation_tps) * 0.70 + max(0.0, prefill_proxy_tps) * 0.30, 4)
