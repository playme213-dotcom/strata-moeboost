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

def compute_plan(vram_headroom_mb: int = 300, override_vram_mb: int | None = None, override_ram_mb: int | None = None):
    vram_tot, vram_free, ram_tot, ram_avail = probe_hardware()
    if override_vram_mb is not None:
        vram_tot = override_vram_mb
        vram_free = override_vram_mb
    if override_ram_mb is not None:
        ram_tot = override_ram_mb
        ram_avail = override_ram_mb

    plan = {
        "vram_total_mb": vram_tot,
        "vram_free_mb": vram_free,
        "vram_headroom_mb": vram_headroom_mb,
        "ram_total_mb": ram_tot,
        "ram_avail_mb": ram_avail,
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
        "summary": ""
    }

    # =========================================================================
    # Tier 1: Ultra-Low Spec (<= 2 GB VRAM or <= 12 GB RAM)
    # =========================================================================
    if vram_tot <= 2048 or ram_tot <= 12288:
        plan["tier"] = "ultra_low"
        plan["tier_name"] = "Tier 1: Ultra-Low Spec (2GB VRAM / <=12GB RAM)"
        plan["dense_in_vram"] = False       # Dense mapped in host RAM to avoid OOM
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 256
        plan["prefill"] = 8
        plan["resident_budget_gib"] = "2.0" # Minimal RAM footprint
        plan["expert_cache_slots"] = "0"
        plan["vram_reserve_mb"] = max(150, vram_headroom_mb)
        plan["summary"] = "Ultra-Low Spec: Host-mapped dense, 2GB expert RAM, 256 ctx, pure NVMe streaming"

    # =========================================================================
    # Tier 2: Low-Spec Preferred (GTX 1050/1650 3GB-4GB VRAM + 16-24GB RAM)
    # USER SPEC: "Low-Spec Preferred no expert on vram"
    # - Dense & KV cache in VRAM
    # - Zero expert slots in VRAM (keeps 100% of remaining VRAM as headroom & scratch)
    # - Hot/Warm experts in RAM (6.0 GiB)
    # - Cold experts + PLE table on SSD
    # =========================================================================
    elif vram_tot <= 4096:
        plan["tier"] = "low_spec_preferred"
        plan["tier_name"] = "Tier 2: Low-Spec Preferred (No Experts in VRAM)"
        plan["dense_in_vram"] = True        # Dense in VRAM (~1.3 GB)
        plan["kv_format"] = "q4_0"          # Compact 4-bit Hadamard KV cache (~350 MB)
        plan["max_context"] = 512
        plan["prefill"] = 16
        plan["resident_budget_gib"] = "6.0" if ram_tot >= 20000 else "4.0"
        plan["expert_cache_slots"] = "0"    # Explicit: NO experts in VRAM
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"Low-Spec Preferred: Dense+KV in VRAM, 0 experts in VRAM (pure headroom/scratch), {plan['resident_budget_gib']}GB expert RAM, NVMe PLE/cold experts"

    # =========================================================================
    # Tier 3: Mid-Spec Balanced (6GB-8GB VRAM, RTX 2060/3060/4060 + 24-32GB RAM)
    # =========================================================================
    elif vram_tot <= 8192:
        plan["tier"] = "mid_spec"
        plan["tier_name"] = "Tier 3: Mid-Spec Balanced (6GB-8GB VRAM + 32GB RAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 2048
        plan["prefill"] = 32
        plan["resident_budget_gib"] = "10.0" if ram_tot >= 28000 else "6.0"
        plan["expert_cache_slots"] = "auto"
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"Mid-Spec Balanced: Dense + ~1,200 Hot Experts in VRAM, 2048 ctx, {plan['resident_budget_gib']}GB expert RAM"

    # =========================================================================
    # Tier 4: High-Spec Extended (12GB-16GB VRAM, RTX 4070/3080/4080 + 32-64GB RAM)
    # =========================================================================
    elif vram_tot <= 16384:
        plan["tier"] = "high_spec"
        plan["tier_name"] = "Tier 4: High-Spec Extended (12GB-16GB VRAM + 32-64GB RAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "k8v4"          # Exact INT8 K + rotated Q4_0 V
        plan["max_context"] = 8192
        plan["prefill"] = 64
        plan["resident_budget_gib"] = "16.0" if ram_tot >= 48000 else "10.0"
        plan["expert_cache_slots"] = "auto"
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = f"High-Spec Extended: Dense + ~4,000 Hot Experts in VRAM, 8192 ctx (k8v4), {plan['resident_budget_gib']}GB expert RAM"

    # =========================================================================
    # Tier 5: Frontier Pro / Workstation (24GB - 48GB+ VRAM, RTX 3090/4090 / A6000)
    # =========================================================================
    else:
        plan["tier"] = "frontier_pro"
        plan["tier_name"] = "Tier 5: Frontier Pro / Full GPU Residency (24GB-48GB+ VRAM)"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "int8"
        plan["max_context"] = 16384
        plan["prefill"] = 128
        plan["resident_budget_gib"] = "auto"
        plan["expert_cache_slots"] = "auto"
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["summary"] = "Frontier Pro: Full Model / Maximum Experts resident on GPU VRAM, 16k ctx, MTP draft active"

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
