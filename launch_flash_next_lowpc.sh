#!/usr/bin/env bash
# ==============================================================================
# Native Strata-HotRod Low-PC Launcher for Flash-Next
# Tailored for low-spec PCs (e.g. 24 GB RAM, 3-4 GB VRAM / GTX 1050 / RX 580)
# ==============================================================================
set -euo pipefail

MODEL_SHARD1="${1:-/media/local-ai/Model/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf}"
MODEL_SHARD2="${2:-/media/local-ai/Model/Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf}"
PORT="${3:-4133}"
PACK_DIR="${4:-/home/local-ai/strata-moeboost/pack/flash_next_gyro}"

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

export PYTHONPATH="$SCRIPT_DIR/build-cuda12/_deps/strata_llamacpp-src/gguf-py:${PYTHONPATH:-}"
export STRATA_GGUF_PY="$SCRIPT_DIR/build-cuda12/_deps/strata_llamacpp-src/gguf-py"
PYTHON_BIN="python3"

# Step 1: Ensure dense weights and vocabulary are unpacked into pack dir
if [[ ! -f "$PACK_DIR/dense.bin" || ! -f "$PACK_DIR/tokenizer/vocab.json" ]]; then
    echo "Unpacking dense layers & tokenizer to $PACK_DIR..."
    mkdir -p "$PACK_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/tools/iq_pack.py" \
        --gguf "$MODEL_SHARD1" \
        --out "$PACK_DIR" \
        --compat-bf16 \
        --skip-experts
    echo "Pack layout ready."
else
    echo "Pack directory $PACK_DIR already prepared."
fi

# Step 2: Ensure 512-expert profile exists for Flash-Next
PROFILE_BIN="$PACK_DIR/profile.bin"
if [[ ! -f "$PROFILE_BIN" ]]; then
    echo "Generating Flash-Next 512-expert profile ($PROFILE_BIN)..."
    "$PYTHON_BIN" "$SCRIPT_DIR/tools/make_profile.py" \
        --no-base \
        --n-expert 512 \
        --out "$PROFILE_BIN"
fi

# Step 3: Configure Low-PC Engine Parameters
# - --mmap-experts: avoids 34-77 GB RAM pre-allocation, reading on demand
# - --expert-profile: pre-ranks experts for device residency without requiring whole-model pre-allocation
# - --kv q4_0: cuts KV cache memory consumption by 50%
# - --max-context 512: safely bounds context size to prevent VRAM spikes
# - --prefill 16: chunked prefill for smooth memory consumption
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
    --expert-profile "$PROFILE_BIN" \
    --kv q4_0 \
    --max-context 512 \
    --prefill 16 \
    --ple-io direct \
    --pool-workers 4 \
    --port "$PORT"
