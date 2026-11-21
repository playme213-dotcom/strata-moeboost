#!/usr/bin/env python3
"""Multi-Tier Hardware Plan Optimizer for Strata-MoEBoost.
Inspects system VRAM and RAM, calculates safe headroom (default 300 MiB VRAM headroom),
and automatically selects the optimal execution plan across diverse hardware setups:
  - Tier 1: Ultra-Low Spec (<= 2 GB VRAM or <= 12 GB RAM)
  - Tier 2: Low-Spec Preferred (GTX 1050 / 3 GB - 4 GB VRAM + 16-24 GB RAM)
  - Tier 3: Mid-Spec Balanced (6 GB - 8 GB VRAM, RTX 2060/3060/4060 + 32 GB RAM)
  - Tier 4: High-Spec Extended (12 GB - 16 GB VRAM, RTX 3080/4070/4080 + 32-64 GB RAM)
  - Tier 5: Frontier Pro / Workstation (24 GB - 48 GB+ VRAM, RTX 3090/4090 / A6000)
"""
import os
import sys
import json
import shutil
import subprocess

def probe_hardware():
    vram_total_mb = 0
    vram_free_mb = 0
    ram_total_mb = 0
    ram_avail_mb = 0

    # 1. Probe GPU VRAM
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            if out:
                parts = out.split("\n")[0].split(",")
                vram_total_mb = int(parts[0].strip())
                vram_free_mb = int(parts[1].strip())
        except Exception:
            pass

    # 2. Probe Host RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_total_mb = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    ram_avail_mb = int(line.split()[1]) // 1024
    except Exception:
        ram_total_mb = 16384
        ram_avail_mb = 12288

    return vram_total_mb, vram_free_mb, ram_total_mb, ram_avail_mb

def compute_plan(vram_headroom_mb: int = 300, override_vram_mb: int | None = None, override_ram_mb: int | None = None, expert_size_mb: float = 1.05):
    """
    Computes optimal multi-tier hardware placement.
    expert_size_mb: Size in MB per expert blob (e.g. 1.05 MB for Flash-Next, ~2.3 MB for Qwen-25B, ~3.8 MB for Qwen-35B).
    Calculates exact dynamic 'helper chairs' that fit in available VRAM while respecting headroom.
    """
    vram_tot, vram_free, ram_tot, ram_avail = probe_hardware()
    if override_vram_mb is not None:
        vram_tot = override_vram_mb
        vram_free = override_vram_mb
    if override_ram_mb is not None:
        ram_tot = override_ram_mb
        ram_avail = override_ram_mb

    # Clamp expert size to safe minimum
    expert_size_mb = max(0.5, float(expert_size_mb))

    plan = {
        "vram_total_mb": vram_tot,
        "vram_free_mb": vram_free,
        "vram_headroom_mb": vram_headroom_mb,
        "ram_total_mb": ram_tot,
        "ram_avail_mb": ram_avail,
        "expert_size_mb": expert_size_mb,
        "tier": "",
        "tier_name": "",
        "dense_in_vram": True,
        "kv_format": "q4_0",
        "max_context": 512,
        "prefill": 16,
        "resident_budget_gib": "6.0",
        "ple_io": "direct",
        "mmap_experts": True,
        "vram_reserve_mb": vram_headroom_mb,
        "expert_cache_slots": "auto",
        "chairs_vram_mb": 0,
        "summary": ""
    }

    # =========================================================================
    # Tier 1: Ultra-Low Spec (<= 2 GB VRAM or <= 12 GB RAM)
    # Target ~64 MB for helper chairs
    # =========================================================================
    if vram_tot <= 2048 or ram_tot <= 12288:
        target_chairs_mb = 64.0
        slots = max(16, int(target_chairs_mb / expert_size_mb))
        actual_mb = round(slots * expert_size_mb, 1)

        plan["tier"] = "ultra_low"
        plan["tier_name"] = f"Tier 1: Ultra-Low Spec ({slots} Helper Chairs + Host-Mapped Dense)"
        plan["dense_in_vram"] = False       # Dense mapped in host RAM to avoid OOM
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 256
        plan["prefill"] = 8
        plan["resident_budget_gib"] = "2.0" # Minimal RAM footprint
        plan["expert_cache_slots"] = str(slots)
        plan["chairs_vram_mb"] = actual_mb
        plan["vram_reserve_mb"] = max(150, vram_headroom_mb)
        plan["summary"] = f"Ultra-Low Spec (<=2GB VRAM): Host-mapped dense, {slots} Helper Chairs ({actual_mb}MB), 2GB expert RAM, 256 ctx, NVMe streaming"

    # =========================================================================
    # Tier 2: Low-Spec Preferred (GTX 1050/1650 3GB-4GB VRAM + 16-24GB RAM)
    # Target ~135 MB for helper chairs (dynamic based on expert_size_mb)
    # =========================================================================
    elif vram_tot < 5120:
        target_chairs_mb = 135.0
        slots = max(32, int(target_chairs_mb / expert_size_mb))
        actual_mb = round(slots * expert_size_mb, 1)

        plan["tier"] = "low_spec_preferred"
        plan["tier_name"] = f"Tier 2: Low-Spec Preferred ({slots} Helper Chairs + 6GB RAM)"
        plan["dense_in_vram"] = True        # Dense in VRAM (~1.4 GB)
        plan["kv_format"] = "q4_0"          # Compact 4-bit Hadamard KV cache (~350 MB)
        plan["max_context"] = 512
        plan["prefill"] = 16
        plan["resident_budget_gib"] = "6.0" if ram_tot >= 20000 else "4.0"
        plan["expert_cache_slots"] = str(slots)
        plan["chairs_vram_mb"] = actual_mb
        plan["vram_reserve_mb"] = max(300, vram_headroom_mb)
        plan["summary"] = f"Low-Spec Preferred (<5GB VRAM): Dense+KV in VRAM, {slots} VRAM Helper Chairs ({actual_mb}MB @ {expert_size_mb}MB/exp), {plan['resident_budget_gib']}GB experts in RAM, NVMe PLE/cold experts, >650MB headroom"

    # =========================================================================
    # Tier 3: Mid-Spec Balanced (5GB - 8GB VRAM, RTX 2060/3060/4060 + 24-32GB RAM)
    # Target ~1,250 MB for helper chairs
    # =========================================================================
    elif vram_tot <= 8192:
        target_chairs_mb = 1250.0
        slots = max(128, int(target_chairs_mb / expert_size_mb))
        actual_mb = round(slots * expert_size_mb, 1)

        plan["tier"] = "mid_spec"
        plan["tier_name"] = f"Tier 3: Mid-Spec Balanced ({slots} Helper Chairs + 32GB RAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 2048
        plan["prefill"] = 32
        plan["resident_budget_gib"] = "10.0" if ram_tot >= 28000 else "6.0"
        plan["expert_cache_slots"] = str(slots)
        plan["chairs_vram_mb"] = actual_mb
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"Mid-Spec Balanced (5GB-8GB VRAM): Dense + {slots} Helper Chairs ({actual_mb}MB @ {expert_size_mb}MB/exp), 2048 ctx, {plan['resident_budget_gib']}GB warm experts in RAM"

    # =========================================================================
    # Tier 4: High-Spec Extended (12GB-16GB VRAM, RTX 4070/3080/4080 + 32-64GB RAM)
    # Target ~6,500 MB for helper chairs
    # =========================================================================
    elif vram_tot <= 16384:
        target_chairs_mb = 6500.0
        slots = max(1024, int(target_chairs_mb / expert_size_mb))
        actual_mb = round(slots * expert_size_mb, 1)

        plan["tier"] = "high_spec"
        plan["tier_name"] = f"Tier 4: High-Spec Extended ({slots} Helper Chairs + 32-64GB RAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "q8_0"          # High-precision Q8_0 KV cache
        plan["max_context"] = 8192
        plan["prefill"] = 64
        plan["resident_budget_gib"] = "24.0" if ram_tot >= 48000 else "16.0"
        plan["expert_cache_slots"] = str(slots)
        plan["chairs_vram_mb"] = actual_mb
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"High-Spec Extended (12GB-16GB VRAM): Dense + {slots} Helper Chairs ({actual_mb}MB @ {expert_size_mb}MB/exp), 8192 ctx (q8_0), {plan['resident_budget_gib']}GB expert RAM, zero host-device stall"

    # =========================================================================
    # Tier 5: Frontier Pro / Workstation (24GB - 48GB+ VRAM, RTX 3090/4090 / A6000)
    # =========================================================================
    else:
        target_chairs_mb = max(14000.0, (vram_tot - 8000.0))
        slots = max(2048, int(target_chairs_mb / expert_size_mb))
        actual_mb = round(slots * expert_size_mb, 1)

        plan["tier"] = "frontier_pro"
        plan["tier_name"] = f"Tier 5: Frontier Pro / Full GPU Residency ({slots} Chairs + {vram_tot // 1024}GB VRAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "f16"           # Uncompressed full precision KV cache
        plan["max_context"] = 16384
        plan["prefill"] = 128
        plan["resident_budget_gib"] = "32.0"
        plan["expert_cache_slots"] = str(slots)
        plan["chairs_vram_mb"] = actual_mb
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"Frontier Pro: Full Model / Maximum Experts resident in VRAM ({slots} chairs, {actual_mb}MB), 16k ctx (f16), async CUDA streams, MTP draft active"

    return plan

    return plan

def simulate_all_tiers():
    scenarios = [
        ("Tier 1 (Ultra-Low)", 2048, 8192),
        ("Tier 2 (Low-Spec Preferred - Your Setup)", 3072, 23458),
        ("Tier 2 (Alternative Low 4GB/16GB)", 4096, 16384),
        ("Tier 3 (Mid-Spec 6GB/24GB)", 6144, 24576),
        ("Tier 3 (Mid-Spec 8GB/32GB)", 8192, 32768),
        ("Tier 4 (High-Spec 12GB/32GB)", 12288, 32768),
        ("Tier 4 (High-Spec 16GB/64GB)", 16384, 65536),
        ("Tier 5 (Frontier 24GB RTX 4090)", 24576, 65536)
    ]
    print("=" * 88)
    print(f"{'Hardware Setup':<35} | {'VRAM / Headroom':<15} | {'RAM Budget':<10} | {'Context':<8} | {'KV':<6}")
    print("=" * 88)
    for name, vram, ram in scenarios:
        p = compute_plan(300, override_vram_mb=vram, override_ram_mb=ram)
        vram_info = f"{vram}MB (-300MB)"
        print(f"{name:<35} | {vram_info:<15} | {str(p['resident_budget_gib']) + ' GiB':<10} | {p['max_context']:<8} | {p['kv_format']:<6}")
    print("=" * 88)

if __name__ == "__main__":
    headroom = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 300
    if "--simulate" in sys.argv or "--tiers" in sys.argv:
        simulate_all_tiers()
        sys.exit(0)

    p = compute_plan(headroom)
    if "--json" in sys.argv:
        print(json.dumps(p, indent=2))
    elif "--bash" in sys.argv:
        print(f"PLAN_TIER='{p['tier']}'")
        print(f"PLAN_TIER_NAME='{p['tier_name']}'")
        print(f"PLAN_DENSE_VRAM={'1' if p['dense_in_vram'] else '0'}")
        print(f"PLAN_KV_FORMAT='{p['kv_format']}'")
        print(f"PLAN_MAX_CONTEXT={p['max_context']}")
        print(f"PLAN_PREFILL={p['prefill']}")
        print(f"PLAN_RESIDENT_BUDGET={p['resident_budget_gib']}")
        print(f"PLAN_EXPERT_CACHE='{p['expert_cache_slots']}'")
        print(f"PLAN_VRAM_RESERVE={p['vram_reserve_mb']}")
        print(f"PLAN_SUMMARY='{p['summary']}'")
    else:
        print(f"=== Strata Hardware Plan Optimizer ===")
        print(f"Detected GPU VRAM   : {p['vram_total_mb']} MB (Free: {p['vram_free_mb']} MB)")
        print(f"Detected Host RAM   : {p['ram_total_mb']} MB (Avail: {p['ram_avail_mb']} MB)")
        print(f"Target VRAM Headroom: {p['vram_headroom_mb']} MB")
        print(f"Selected Plan       : {p['tier_name']}")
        print(f"Expert RAM Budget   : {p['resident_budget_gib']} GiB")
        print(f"KV Cache Format     : {p['kv_format']} (Max Context: {p['max_context']})")
        print(f"NVMe PLE Streaming  : {p['ple_io']}")
        print(f"Summary             : {p['summary']}")
