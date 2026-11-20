#!/usr/bin/env bash
# ==============================================================================
# Flash-Next Gyro-S Dedicated VRAM-Dense + Hot Expert Launcher
# Specifically tuned for GTX 1050 (3GB VRAM) + 22GB Host RAM + NVMe Streaming
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build-cuda12/strata"
MODEL_FILE="${1:-/media/local-ai/Model/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf}"
PLE_FILE="${2:-/media/local-ai/Model/Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf}"
PACK_DIR="${3:-$SCRIPT_DIR/pack/flash_next_gyro}"
PORT="${4:-4133}"

echo "=================================================================="
echo " Starting Strata Native Engine: Flash-Next Gyro-S (VRAM Dense Plan)"
echo " Engine      : $BIN"
echo " Model       : $MODEL_FILE"
echo " PLE (N-Gram): $PLE_FILE"
echo " Pack        : $PACK_DIR"
echo " Port        : $PORT"
echo "=================================================================="

# Check if model is unpacked
if [[ ! -f "$PACK_DIR/dense.bin" ]]; then
    echo "Dense pack not initialized. Running automated setup..."
    bash "$SCRIPT_DIR/setup_flash_next.sh"
fi

PROFILE_BIN="$PACK_DIR/profile.bin"

# Engine Parameters:
# --native: Places dense Q/K/V/O, norms, and shared expert directly onto CUDA VRAM (~1.3 GB)
# --expert-profile: Pre-allocates top hot experts into remaining GPU VRAM slots (~850 MB)
# --kv q4_0: Compresses KV cache using Hadamard rotation to ~350 MB
# --max-context 512: Safe context window for low-VRAM profile
# --prefill 16: Chunked prefill
# --ple-io direct: Async NVMe reads for PLE table with 0 RAM overhead
# --mmap-experts: Keeps cold experts on SSD/RAM without full 58GB allocation
# --pool-workers 4: Optimized for Ryzen 5 3550H CPU threads

# Auto-adjust hardware plan based on live VRAM and RAM
echo "Checking system VRAM and RAM for optimal plan (Target Headroom: 300MB VRAM)..."
eval "$("$SCRIPT_DIR/tools/hardware_plan.py" 300 --bash)"
echo "Selected Tier: $PLAN_TIER"
echo "Plan Summary : $PLAN_SUMMARY"

PLE_ARGS=()
if [[ -f "$PLE_FILE" ]]; then
    echo "Found standalone PLE Table: $PLE_FILE. Enabling --ple-io direct NVMe stream."
    PLE_ARGS=("--ple-gguf" "$PLE_FILE" "--ple-io" "direct")
else
    echo "Notice: Standalone PLE Table not present yet. Running in clean --no-ple mode."
    PLE_ARGS=("--no-ple")
fi

export STRATA_FETCH_THREADS=4
export STRATA_PAGECACHE_WARM_GIB=4

exec "$BIN" \
    --pack "$PACK_DIR" \
    --native "$MODEL_FILE" \
    "${PLE_ARGS[@]}" \
    --spec 2 \
    --mmap-experts \
    --expert-cache "${PLAN_EXPERT_CACHE:-128}" \
    --resident-budget-gib "$PLAN_RESIDENT_BUDGET" \
    --vram-reserve-mib "$PLAN_VRAM_RESERVE" \
    --expert-profile "$PROFILE_BIN" \
    --kv "$PLAN_KV_FORMAT" \
    --max-context "$PLAN_MAX_CONTEXT" \
    --prefill "$PLAN_PREFILL" \
    --pool-workers 4 \
    --port "$PORT"
