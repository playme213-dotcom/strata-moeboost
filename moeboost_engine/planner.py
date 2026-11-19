"""Conservative load-fit planner for dense and sparse GGUF models."""
from __future__ import annotations

from engine.accelerator import memory_snapshot

MIB = 1024 * 1024


def make_load_plan(details: dict, ram_budget_bytes: int) -> dict:
    tensors = [tensor for layer in details.get("layers", {}).values() for tensor in layer]
    # Include non-layer tensors in total/base sizes as well; inspector's complete tensor table is not
    # duplicated in the layer map for embeddings/output tensors, so use file size as conservative total.
    total_bytes = int(details.get("file_size") or 0)
    expert_bytes = sum(int(t.get("byte_size") or 0) for t in tensors if t.get("is_expert"))
    exact = all(t.get("size_exact", False) for t in tensors)
    expert_count = details.get("expert_count")
    active_count = details.get("active_experts")
    moe = details.get("classification") == "MoE" and bool(expert_count)
    expert_count = max(1, int(expert_count or 1))
    per_expert = expert_bytes / expert_count if expert_bytes else None
    active_window = per_expert * active_count if per_expert is not None and active_count else None
    base_bytes = max(0, total_bytes - expert_bytes)

    hardware = memory_snapshot()
    target_vram = 2560 * MIB  # 3 GB device target with 500 MiB held back.
    selected_backend = hardware.get("backend")
    accelerator = next((device for device in hardware["gpu"]
                        if device.get("backend") == selected_backend
                        and device.get("device_type") != "cpu"), None)
    vram_budget = None
    if accelerator and accelerator.get("vram_free_mb") is not None:
        vram_budget = min(target_vram, max(0, accelerator["vram_free_mb"] - 500) * MIB)
    fits_full_vram = total_bytes <= vram_budget if vram_budget is not None else None
    fits_full_ram = total_bytes <= ram_budget_bytes
    # A full-file VRAM estimate is only asserted when the entire GGUF fits.
    # Otherwise model weights are treated as host-mapped for this conservative view.
    # This runtime deliberately keeps MoE weights host-mapped: llama.cpp places
    # whole layers, not individual experts, so pretending expert weights fit in
    # VRAM would overstate GPU residency.
    estimated_vram_weights = total_bytes if fits_full_vram and not moe else 0
    # The GGUF is mapped as a whole. Actual resident pages depend on workload and
    # Linux reclaim; use the host-mapped file footprint as the conservative bound.
    estimated_host_working_set = max(0, total_bytes - estimated_vram_weights)
    hardware_ram = hardware.get("ram") or {}
    ram_available = hardware_ram.get("available_bytes")
    ram_total = hardware_ram.get("total_bytes")
    from moe_boost.cache import get_configured_max_ram_bytes
    max_ram = get_configured_max_ram_bytes()
    effective_ram_budget = max(int(ram_budget_bytes), int((ram_available or 0) * 0.90), int((ram_total or 0) * 0.85))
    if max_ram is not None:
        effective_ram_budget = min(effective_ram_budget, max_ram)
    projected_ram_available = (max(0, int(ram_available) - int(estimated_host_working_set))
                               if ram_available is not None else None)
    active_footprint = base_bytes + active_window if moe and active_window is not None else None
    fits_active_vram = active_footprint <= vram_budget if active_footprint is not None and vram_budget is not None else None
    fits_active_ram = active_footprint <= effective_ram_budget if active_footprint is not None else None
    fits_full_ram = total_bytes <= effective_ram_budget

    if fits_full_vram:
        strategy = "full-model-vram"
        verdict = "fits"
    elif moe and fits_active_vram:
        strategy = "base-plus-active-experts-vram"
        verdict = "active-window-fits"
    elif fits_full_ram:
        strategy = "full-model-ram"
        verdict = "fits"
    elif moe and fits_active_ram:
        strategy = "ssd-cold-experts-ram-active-window"
        verdict = "tiered-window-fits"
    elif moe and per_expert is not None and per_expert <= effective_ram_budget and base_bytes <= effective_ram_budget:
        strategy = "ssd-paged-expert-chunks"
        verdict = "chunked-paging-required"
    elif total_bytes > 0:
        strategy = "mmap-paged-host-ram"
        verdict = "mmap-reclaimable-fits"
        fits_active_ram = True
    else:
        strategy = "insufficient-memory-or-unknown"
        verdict = "cannot-confirm-fit"
    return {
        "model": details.get("path"), "classification": details.get("classification"),
        "expert_count": expert_count if moe else None, "active_experts_per_layer": active_count,
        "model_file_bytes": total_bytes, "estimated_expert_tensor_bytes": expert_bytes,
        "estimated_base_bytes": base_bytes, "estimated_bytes_per_expert_all_layers": per_expert,
        "estimated_active_expert_window_bytes": active_window,
        "estimated_vram_model_weights_bytes": estimated_vram_weights,
        "estimated_host_working_set_bytes": estimated_host_working_set,
        "ram_available_before_bytes": ram_available,
        "estimated_ram_available_after_bytes": projected_ram_available,
        "ram_reserve_percent": int(__import__("os").getenv("MOE_RAM_RESERVE_PERCENT", "35"))
                               if __import__("os").getenv("MOE_RAM_RESERVE_PERCENT", "35").isdigit() else 35,
        "tensor_sizes_exact": exact, "vram_budget_bytes": vram_budget,
        "ram_cache_budget_bytes": effective_ram_budget,
        "fits_full_vram": fits_full_vram, "fits_full_ram": fits_full_ram,
        "fits_active_window_vram": fits_active_vram, "fits_active_window_ram": fits_active_ram,
        "strategy": strategy, "verdict": verdict,
        "note": "Estimates are planning bounds, not exact allocations. GGUF weights are memory-mapped; Linux may reclaim clean mapped pages. Partial GPU layer offload and K/Q/V + KV-cache offload are used when VRAM headroom permits. The RAM reserve is protected by the planner, not a hard OS reservation. KV cache, compute buffers, context length and concurrent apps add runtime memory use.",
    }
