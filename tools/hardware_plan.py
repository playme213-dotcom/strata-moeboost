#!/usr/bin/env python3
"""Hardware Plan Optimizer for Strata-MoEBoost.
Inspects system VRAM and RAM, calculates safe headroom (default 300 MiB VRAM headroom),
and determines the optimal execution tiering plan:
  - Dense in VRAM vs mapped host memory
  - KV cache format & context length
  - Resident RAM expert budget
  - Hot expert slots in VRAM
  - Direct NVMe streaming for PLE and cold experts
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

def compute_plan(vram_headroom_mb: int = 300):
    vram_tot, vram_free, ram_tot, ram_avail = probe_hardware()
    
    # Defaults for low-spec reference (e.g. GTX 1050 3GB / 22GB RAM)
    dense_mb = 1350        # dense.bin + Q/K/V/O, norms, shared expert
    kv_q4_512_mb = 350     # KV cache q4_0 at 512 context
    usable_vram = max(0, vram_free - vram_headroom_mb)

    plan = {
        "vram_total_mb": vram_tot,
        "vram_free_mb": vram_free,
        "vram_headroom_mb": vram_headroom_mb,
        "ram_total_mb": ram_tot,
        "ram_avail_mb": ram_avail,
        "tier": "low_spec_preferred",
        "dense_in_vram": True,
        "kv_format": "q4_0",
        "max_context": 512,
        "prefill": 16,
        "resident_budget_gib": 6.0,
        "ple_io": "direct",
        "mmap_experts": True,
        "vram_reserve_mb": vram_headroom_mb,
        "summary": ""
    }

    # Decide VRAM & RAM tiering
    if vram_tot <= 4096:
        # Low-spec preferred plan (GTX 1050 / GTX 1650 3GB-4GB)
        plan["tier"] = "low_spec_preferred"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 512
        plan["prefill"] = 16
        plan["vram_reserve_mb"] = vram_headroom_mb
        # Expert RAM Budget: keep ~6.0 GB in RAM if system has >= 16 GB RAM, else 4.0 GB
        if ram_tot >= 20000:
            plan["resident_budget_gib"] = 6.0
        elif ram_tot >= 14000:
            plan["resident_budget_gib"] = 4.0
        else:
            plan["resident_budget_gib"] = 2.0
        plan["summary"] = f"Low-Spec Optimal Plan (VRAM: Dense+KV in 3GB with {vram_headroom_mb}MB headroom, RAM: {plan['resident_budget_gib']}GB expert cache, NVMe: PLE & cold experts direct stream)"
    elif vram_tot <= 8192:
        # Mid-tier GPU (6GB-8GB VRAM)
        plan["tier"] = "mid_spec"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "q4_0"
        plan["max_context"] = 2048
        plan["prefill"] = 32
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["resident_budget_gib"] = 8.0 if ram_tot >= 20000 else 4.0
        plan["summary"] = f"Mid-Spec Plan (VRAM: Dense + KV 2048 + Hot Experts, RAM: {plan['resident_budget_gib']}GB expert cache)"
    else:
        # High-tier GPU (12GB+ VRAM)
        plan["tier"] = "high_spec"
        plan["dense_in_vram"] = True
        plan["kv_format"] = "int8"
        plan["max_context"] = 4096
        plan["prefill"] = 64
        plan["vram_reserve_mb"] = vram_headroom_mb
        plan["resident_budget_gib"] = 12.0
        plan["summary"] = "High-Spec Plan (Maximum VRAM expert residency & INT8 KV)"

    return plan

if __name__ == "__main__":
    headroom = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    p = compute_plan(headroom)
    if "--json" in sys.argv:
        print(json.dumps(p, indent=2))
    elif "--bash" in sys.argv:
        print(f"PLAN_TIER='{p['tier']}'")
        print(f"PLAN_DENSE_VRAM={'1' if p['dense_in_vram'] else '0'}")
        print(f"PLAN_KV_FORMAT='{p['kv_format']}'")
        print(f"PLAN_MAX_CONTEXT={p['max_context']}")
        print(f"PLAN_PREFILL={p['prefill']}")
        print(f"PLAN_RESIDENT_BUDGET={p['resident_budget_gib']}")
        print(f"PLAN_VRAM_RESERVE={p['vram_reserve_mb']}")
        print(f"PLAN_SUMMARY='{p['summary']}'")
    else:
        print(f"=== Strata Hardware Plan Optimizer ===")
        print(f"Detected GPU VRAM   : {p['vram_total_mb']} MB (Free: {p['vram_free_mb']} MB)")
        print(f"Detected Host RAM   : {p['ram_total_mb']} MB (Avail: {p['ram_avail_mb']} MB)")
        print(f"Target VRAM Headroom: {p['vram_headroom_mb']} MB")
        print(f"Selected Plan Tier  : {p['tier']}")
        print(f"Expert RAM Budget   : {p['resident_budget_gib']} GiB")
        print(f"KV Cache Format     : {p['kv_format']} (Max Context: {p['max_context']})")
        print(f"NVMe PLE Streaming  : {p['ple_io']}")
        print(f"Summary             : {p['summary']}")
