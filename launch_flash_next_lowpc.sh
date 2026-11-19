#!/usr/bin/env bash
# ==============================================================================
# Native Strata-HotRod Low-PC Launcher for Flash-Next
# Tailored for low-spec PCs (e.g. 24 GB RAM, 3-4 GB VRAM / GTX 1050 / RX 580)
# ==============================================================================
set -euo pipefail

MODEL_SHARD1="${1:-/media/local-ai/Model/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf}"
MODEL_SHARD2="${2:-/media/local-ai/Model/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf}"
PORT="${3:-4133}"
PACK_DIR="${4:-/home/local-ai/strata-moeboost/pack/flash_next_lowpc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build-cuda12/strata"

echo "=== Strata-HotRod Low-PC Flash-Next Launcher ==="
echo "Engine Binary : $BIN"
echo "Shard 1       : $MODEL_SHARD1"
echo "Shard 2 (PLE) : $MODEL_SHARD2"
echo "Pack Output   : $PACK_DIR"
echo "Port          : $PORT"
echo "================================================"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Native engine $BIN not found or not executable."
    exit 1
fi

# Step 1: Ensure dense weights and vocabulary are unpacked into pack dir
if [[ ! -f "$PACK_DIR/dense.bin" || ! -f "$PACK_DIR/tokenizer/vocab.json" ]]; then
    echo "Unpacking dense layers & tokenizer to $PACK_DIR..."
    mkdir -p "$PACK_DIR"
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tools/iq_pack.py" \
        --gguf "$MODEL_SHARD1" \
        --out "$PACK_DIR" \
        --compat-bf16 \
        --skip-experts
    echo "Pack layout ready."
else
    echo "Pack directory $PACK_DIR already prepared."
fi

# Step 2: Configure Low-PC Engine Parameters
# - --mmap-experts: avoids 34-77 GB RAM pre-allocation, reading on demand
# - --resident-budget-gib 10: keeps top active experts in host RAM (capped safely at 10 GiB out of 24 GB RAM)
# - --kv k8v4 or q4_0: cuts KV cache memory consumption by 50%
# - --ple-io direct: unbuffered NVMe reads for the 28 GB PLE table, skipping OS page cache pollution
# - --pool-workers 4: matches physical core count (e.g. 4 cores / 8 threads)

export STRATA_FETCH_THREADS=4
export STRATA_PAGECACHE_WARM_GIB=4

echo "Launching native Strata engine with Low-PC profile..."
exec "$BIN" \
    --pack "$PACK_DIR" \
    --native "$MODEL_SHARD1" \
    --ple-gguf "$MODEL_SHARD2" \
    --mmap-experts \
    --resident-budget-gib 10 \
    --kv q4_0 \
    --ple-io direct \
    --pool-workers 4 \
    --port "$PORT"
