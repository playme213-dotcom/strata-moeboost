#!/usr/bin/env python3
"""Unified CLI Entry Point for Strata-MoEBoost.

Commands:
  run       Launch the Strata-MoEBoost Server (with preset profiles or custom model)
  plan      Inspect host hardware and show recommended execution plan
  unpack    Extract GGUF into zero-copy unpacked layout for native engine
  optimize  Simulate or perform GGUF quantization (Q2_K, Q3_K_S, Q3_K_M)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def cmd_plan(args):
    from strata_moeboost.tuning import compute_plan, probe_hardware, simulate_all_tiers
    if args.simulate:
        simulate_all_tiers()
        return

    vram, free_vram, ram, avail_ram = probe_hardware()
    print("=" * 68)
    print(" ⚡ Strata-MoEBoost Hardware Inspection & Auto-Plan")
    print("=" * 68)
    print(f" Detected VRAM : {vram} MB (Free: {free_vram} MB)")
    print(f" Detected RAM  : {ram} MB (Available: {avail_ram} MB)")
    plan = compute_plan(vram_headroom_mb=args.headroom, expert_size_mb=args.expert_size)
    print("-" * 68)
    print(f" Selected Tier : {plan['tier_name']}")
    print(f" Dense Weight  : {'VRAM' if plan['dense_in_vram'] else 'Host RAM'}")
    print(f" Helper Chairs : {plan['expert_cache_slots']} slots ({plan['chairs_vram_mb']} MB in VRAM)")
    print(f" Expert RAM    : {plan['resident_budget_gib']} GiB")
    print(f" KV Quant      : {plan['kv_format'].upper()}")
    print(f" Max Context   : {plan['max_context']} tokens")
    print(f" Headroom      : {plan['vram_reserve_mb']} MB")
    print(f" Summary       : {plan['summary']}")
    print("=" * 68)

def cmd_unpack(args):
    from strata_moeboost.utils import unpack_gguf
    print(f"[*] Unpacking model {args.model} into {args.out or 'pack/'}...")
    res = unpack_gguf(args.model, args.out, compat_bf16=not args.no_bf16, skip_experts=args.skip_experts)
    print(f"[✓] Unpacking complete: {res}")

def cmd_optimize(args):
    from strata_moeboost.tuning import quantize_model, QUANT_PRESETS
    if args.preset not in QUANT_PRESETS:
        print(f"[-] Invalid preset: {args.preset}. Choose from: {list(QUANT_PRESETS.keys())}")
        sys.exit(1)
    print(f"[*] {'Simulating' if args.dry_run else 'Quantizing'} {args.model} with preset {args.preset}...")
    out_path = args.out or f"{Path(args.model).stem}_{args.preset}.gguf"
    res = quantize_model(args.model, out_path, preset=args.preset, threads=args.threads, dry_run=args.dry_run)
    print(f"[✓] Done: {res}")

def cmd_run(args):
    # Delegate to server launch
    cmd = [
        sys.executable, str(ROOT / "strata_moeboost_server.py"),
        "--host", args.host,
        "--port", str(args.port),
        "--mtp", args.mtp,
        "--engine-mode", args.engine_mode,
        "--kv-quant", args.kv_quant
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.low_spec:
        cmd.append("--low-spec")
    if args.gpu_layers is not None:
        cmd.extend(["--gpu-layers", str(args.gpu_layers)])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.batch is not None:
        cmd.extend(["--batch", str(args.batch)])
    if args.ctx is not None:
        cmd.extend(["--ctx", str(args.ctx)])

    os.execv(sys.executable, cmd)

def main():
    parser = argparse.ArgumentParser(description="Strata-MoEBoost Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run
    p_run = subparsers.add_parser("run", help="Run the Strata-MoEBoost inference server")
    p_run.add_argument("--model", type=str, default="", help="Path to GGUF model")
    p_run.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_run.add_argument("--port", type=int, default=4133, help="Port")
    p_run.add_argument("--low-spec", action="store_true", help="Enforce low-spec memory profile")
    p_run.add_argument("--mtp", type=str, default="auto", choices=["auto", "on", "off"], help="MTP speculative drafting")
    p_run.add_argument("--engine-mode", type=str, default="auto", choices=["auto", "unpacked", "llama"], help="Execution engine")
    p_run.add_argument("--kv-quant", type=str, default="q4_0", choices=["q4_0", "q8_0", "f16"], help="KV cache quantization")
    p_run.add_argument("--gpu-layers", type=int, default=None, help="VRAM layer offload")
    p_run.add_argument("--threads", type=int, default=None, help="CPU threads")
    p_run.add_argument("--batch", type=int, default=None, help="Micro-batch size")
    p_run.add_argument("--ctx", type=int, default=None, help="Context limit")

    # Plan
    p_plan = subparsers.add_parser("plan", help="Inspect hardware and print recommended tier plan")
    p_plan.add_argument("--headroom", type=int, default=300, help="VRAM reserve in MB")
    p_plan.add_argument("--expert-size", type=float, default=1.05, help="Average expert size in MB")
    p_plan.add_argument("--simulate", action="store_true", help="Simulate all 5 hardware tiers")

    # Unpack
    p_unpack = subparsers.add_parser("unpack", help="Unpack GGUF for native zero-copy loader")
    p_unpack.add_argument("model", type=str, help="Source GGUF model path")
    p_unpack.add_argument("--out", type=str, default=None, help="Output destination folder")
    p_unpack.add_argument("--no-bf16", action="store_true", help="Do not unpack bf16 dense tensor")
    p_unpack.add_argument("--skip-experts", action="store_true", help="Skip individual expert weight files")

    # Optimize
    p_opt = subparsers.add_parser("optimize", help="GGUF quantization speed optimizer")
    p_opt.add_argument("model", type=str, help="Source GGUF model path")
    p_opt.add_argument("--preset", type=str, default="Q3_K_S", choices=["Q2_K", "Q3_K_S", "Q3_K_M"], help="Quant preset")
    p_opt.add_argument("--out", type=str, default=None, help="Output GGUF path")
    p_opt.add_argument("--threads", type=int, default=4, help="Quantization threads")
    p_opt.add_argument("--dry-run", action="store_true", help="Simulate speed and size gain without writing")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "unpack":
        cmd_unpack(args)
    elif args.command == "optimize":
        cmd_optimize(args)

if __name__ == "__main__":
    main()
