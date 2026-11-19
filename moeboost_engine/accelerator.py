"""Conservative local accelerator discovery and backend tuning suggestions."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass


@dataclass
class AcceleratorConfig:
    backend: str = "auto"
    gpu_layers: int = -1
    vk_queues: int = 1
    cuda_streams: int = 1
    vram_headroom_mb: int = 400


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


_HARDWARE_CACHE: dict = {"at": 0.0, "value": None}
_HARDWARE_LOCK = threading.Lock()


def discover_hardware() -> dict:
    with _HARDWARE_LOCK:
        now = time.monotonic()
        if _HARDWARE_CACHE["value"] is not None and now - _HARDWARE_CACHE["at"] < 10:
            return {"selected_backend": _HARDWARE_CACHE["value"]["selected_backend"],
                    "devices": [dict(device) for device in _HARDWARE_CACHE["value"]["devices"]]}
        result = _discover_hardware()
        _HARDWARE_CACHE.update(at=now, value=result)
        return {"selected_backend": result["selected_backend"],
                "devices": [dict(device) for device in result["devices"]]}


def _discover_hardware() -> dict:
    devices = []
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        output = _run([nvidia, "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                       "--format=csv,noheader,nounits"])
        if output:
            for line in output.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 5:
                    continue
                try:
                    devices.append({"backend": "cuda", "name": fields[0], "vram_total_mb": int(fields[1]),
                                    "vram_used_mb": int(fields[2]), "vram_free_mb": int(fields[3]),
                                    "utilization_percent": int(fields[4])})
                except ValueError:
                    continue
    vulkan = shutil.which("vulkaninfo")
    if vulkan:
        output = _run([vulkan, "--summary"], timeout=5.0)
        if output:
            for section in re.split(r"(?=GPU\d+:)", output):
                name_match = re.search(r"deviceName\s*=\s*(.+)", section)
                index_match = re.match(r"GPU(\d+):", section)
                if not name_match:
                    continue
                name = name_match.group(1).strip()
                vulkan_device = next((device for device in devices
                                      if device.get("backend") == "vulkan" and device.get("name") == name), None)
                if vulkan_device is None:
                    devices.append({"backend": "vulkan", "name": name,
                                    "vram_total_mb": None, "vram_used_mb": None,
                                    "vram_free_mb": None, "utilization_percent": None,
                                    "vulkan_index": int(index_match.group(1)) if index_match else None})
        # Vulkan heap probing is also needed when selecting Vulkan on a host
        # that exposes the same device through CUDA.
        if output:
            detailed = _run([vulkan, "--text"], timeout=8.0)
            if detailed:
                sections = re.split(r"(?=GPU\d+:)", detailed)
                for section in sections:
                    name_match = re.search(r"deviceName\s*=\s*(.+)", section)
                    if not name_match:
                        continue
                    name = name_match.group(1).strip()
                    device = next((item for item in devices
                                   if item["backend"] == "vulkan" and item["name"] == name), None)
                    if not device:
                        continue
                    index_match = re.match(r"GPU(\d+):", section)
                    if index_match:
                        device["vulkan_index"] = int(index_match.group(1))
                    device_type = re.search(r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)", section)
                    if device_type:
                        device["device_type"] = device_type.group(1).lower()
                    local_heaps = []
                    heaps = re.findall(r"memoryHeaps\[\d+\]:\s*(.*?)(?=memoryHeaps\[\d+\]:|memoryTypes:)",
                                       section, re.S)
                    for heap in heaps:
                        if "MEMORY_HEAP_DEVICE_LOCAL_BIT" not in heap:
                            continue
                        size = re.search(r"^\s*size\s*=\s*(\d+)", heap, re.M)
                        budget = re.search(r"^\s*budget\s*=\s*(\d+)", heap, re.M)
                        usage = re.search(r"^\s*usage\s*=\s*(\d+)", heap, re.M)
                        size_bytes = int(size.group(1)) if size else 0
                        local_heaps.append((size_bytes, int(budget.group(1)) if budget else size_bytes,
                                            int(usage.group(1)) if usage else 0))
                    if local_heaps:
                        total, budget_total, used = max(local_heaps, key=lambda heap: heap[0])
                        device["vram_total_mb"] = total // (1024 * 1024)
                        device["vram_used_mb"] = used // (1024 * 1024)
                        device["vram_free_mb"] = max(0, budget_total - used) // (1024 * 1024)
    selected = "cuda" if any(d["backend"] == "cuda" for d in devices) else (
        "vulkan" if any(d["backend"] == "vulkan" and d.get("device_type") != "cpu"
                        for d in devices) else "cpu")
    return {"selected_backend": selected, "devices": devices}


def resolve_config(requested: AcceleratorConfig | None = None, layer_count: int | None = None,
                   layer_sizes: dict[int, int | None] | None = None) -> dict:
    requested = requested or AcceleratorConfig()
    if requested.backend not in {"auto", "cuda", "vulkan", "cpu"}:
        raise ValueError("backend must be auto, cuda, vulkan, or cpu")
    if requested.vram_headroom_mb < 0 or requested.cuda_streams < 1 or requested.vk_queues < 1:
        raise ValueError("VRAM headroom must be non-negative and queue/stream counts must be positive")
    hardware = discover_hardware()
    backend = hardware["selected_backend"] if requested.backend == "auto" else requested.backend
    candidates = [d for d in hardware["devices"] if d["backend"] == backend
                  and (backend != "vulkan" or d.get("device_type") != "cpu")]
    if backend == "vulkan":
        candidates.sort(key=lambda d: (d.get("device_type") == "discrete_gpu",
                                       d.get("vram_free_mb") is not None,
                                       d.get("vram_free_mb") or 0), reverse=True)
    device = candidates[0] if candidates else None
    gpu_layers = requested.gpu_layers
    usable_vram_mb = None
    if device and device.get("vram_free_mb") is not None:
        usable_vram_mb = max(0, device["vram_free_mb"] - requested.vram_headroom_mb)
    if gpu_layers < 0:
        # If layer sizes are known, calculate exact fit; otherwise if CUDA is available, offload to capacity
        gpu_layers = 0
        if backend in {"cuda", "vulkan"} and usable_vram_mb is not None:
            if layer_sizes and all(size is not None for size in layer_sizes.values()):
                remaining = usable_vram_mb * 1024 * 1024
                for layer in sorted(layer_sizes, reverse=True):
                    if layer_sizes[layer] > remaining:
                        break
                    remaining -= layer_sizes[layer]
                    gpu_layers += 1
            elif layer_count is not None and layer_count > 0:
                # Sharded or split models with weights across files: enable GPU layer offloading
                gpu_layers = layer_count
    if layer_count is not None:
        gpu_layers = max(0, min(gpu_layers, layer_count))
    if requested.gpu_layers >= 0 and gpu_layers > 0 and backend in {"cuda", "vulkan"}:
        if usable_vram_mb is None or not layer_sizes or any(size is None for size in layer_sizes.values()):
            gpu_layers = 0
        else:
            affordable = 0
            remaining = usable_vram_mb * 1024 * 1024
            for layer in sorted(layer_sizes, reverse=True):
                if layer_sizes[layer] > remaining or affordable >= gpu_layers:
                    break
                remaining -= layer_sizes[layer]
                affordable += 1
            gpu_layers = affordable
    if backend == "cpu":
        gpu_layers = 0
    # GTX 1050 3GB target: retain requested 500MB headroom and avoid aggressive concurrency.
    if device and "gtx 1050" in device["name"].lower():
        gpu_layers = min(gpu_layers, max(0, layer_count or 1))
        streams = min(requested.cuda_streams, 2)
    else:
        streams = requested.cuda_streams
    return {"requested": asdict(requested), "backend": backend, "gpu_layers": gpu_layers,
            "cuda_streams": streams, "vk_queues": requested.vk_queues,
            "vram_headroom_mb": requested.vram_headroom_mb,
            "selected_device": device, "hardware": hardware,
            "usable_vram_mb": usable_vram_mb,
            "note": "The runtime applies this layer limit when loading a model; changing accelerator settings does not alter an already loaded model."}


def memory_snapshot() -> dict:
    gpu = discover_hardware()
    from moe_boost.cache import get_configured_max_ram_bytes
    max_ram = get_configured_max_ram_bytes()
    try:
        import psutil
        vm = psutil.virtual_memory()
        try:
            reserve_percent = min(60, max(5, int(os.getenv("MOE_RAM_RESERVE_PERCENT", "35"))))
        except ValueError:
            reserve_percent = 10
        ceiling = int(vm.total * (100 - reserve_percent) / 100)
        if max_ram is not None:
            ceiling = min(ceiling, max_ram)
        ram = {"total_bytes": vm.total, "used_bytes": vm.used, "available_bytes": vm.available,
               "allocatable_ceiling_bytes": ceiling,
               "reserve_target_bytes": int(vm.total * reserve_percent / 100),
               "reserve_percent": reserve_percent,
               "max_ram_bytes": max_ram}
    except ImportError:
        try:
            reserve_percent = min(60, max(5, int(os.getenv("MOE_RAM_RESERVE_PERCENT", "35"))))
        except ValueError:
            reserve_percent = 10
        ceiling = max_ram if max_ram is not None else None
        ram = {"total_bytes": None, "used_bytes": None, "available_bytes": None,
               "allocatable_ceiling_bytes": ceiling, "reserve_target_bytes": None,
               "reserve_percent": reserve_percent,
               "max_ram_bytes": max_ram}
    return {"ram": ram, "gpu": gpu["devices"], "backend": gpu["selected_backend"]}
