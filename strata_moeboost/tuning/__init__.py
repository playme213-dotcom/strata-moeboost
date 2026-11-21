"""Hardware plan detection and quantization optimizer for Strata-MoEBoost."""
from __future__ import annotations

from tools.hardware_plan import probe_hardware, compute_plan, simulate_all_tiers
from strata_optimizer import analyze_hardware, quantize_model, QUANT_PRESETS

__all__ = ["probe_hardware", "compute_plan", "simulate_all_tiers", "analyze_hardware", "quantize_model", "QUANT_PRESETS"]
