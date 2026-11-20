#!/usr/bin/env bash
# ==============================================================================
# Qwen3-Coder-25B (Cerebras REAP 25B-A3B) Launcher
# Hardware Plan: GPU Prefill + RAM Expert Generation (Zero Experts in VRAM)
# GTX 1050 (3GB VRAM) + 22GB Host RAM
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build-cuda12/strata"
MODEL_FILE="${1:-/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf}"
DRAFT_FILE="${2:-/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS_draft.gguf}"
PACK_DIR="${3:-$SCRIPT_DIR/pack/qwen25b_coder}"
PORT="${4:-4133}"

echo "=================================================================="
echo " Starting Strata Engine: Qwen-25B Coder (GPU Prefill + RAM Decode)"
echo " Engine : $BIN"
echo " Model  : $MODEL_FILE"
echo " Draft  : $DRAFT_FILE"
echo " Pack   : $PACK_DIR"
echo " Port   : $PORT"
echo "=================================================================="

# Check and unpack dense projections if not yet prepared
if [[ ! -f "$PACK_DIR/dense.bin" ]]; then
    echo "Unpacking Qwen-25B Coder dense layers to $PACK_DIR..."
    mkdir -p "$PACK_DIR"
    python3 "$SCRIPT_DIR/tools/iq_pack.py" \
        --gguf "$MODEL_FILE" \
        --out "$PACK_DIR" \
        --compat-bf16 \
        --skip-experts
    python3 "$SCRIPT_DIR/tools/make_profile.py" \
        --no-base \
        --n-expert 64 \
        --out "$PACK_DIR/profile.bin"
fi

# Hardware Plan Optimizer: Auto-check VRAM/RAM (300MB VRAM headroom)
eval "$("$SCRIPT_DIR/tools/hardware_plan.py" 300 --bash)"
echo "Plan Summary: $PLAN_SUMMARY"

DRAFT_ARGS=()
if [[ -f "$DRAFT_FILE" ]]; then
    echo "Found MTP Draft Head: $DRAFT_FILE (Enabling speculative decode)"
    DRAFT_ARGS=("--draft-gguf" "$DRAFT_FILE")
fi

export STRATA_FETCH_THREADS=4
export STRATA_PAGECACHE_WARM_GIB=4

exec "$BIN" \
    --pack "$PACK_DIR" \
    --native "$MODEL_FILE" \
    --no-ple \
    "${DRAFT_ARGS[@]}" \
    --mmap-experts \
    --expert-cache 0 \
    --resident-budget-gib "$PLAN_RESIDENT_BUDGET" \
    --vram-reserve-mib "$PLAN_VRAM_RESERVE" \
    --expert-profile "$PACK_DIR/profile.bin" \
    --kv "$PLAN_KV_FORMAT" \
    --max-context 2048 \
    --prefill 16 \
    --pool-workers 4 \
    --port "$PORT"
