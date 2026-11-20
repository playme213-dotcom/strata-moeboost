#!/usr/bin/env bash
# ==============================================================================
# Automated Flash-Next Gyro-S Pack & Launch Preparation
# Configured for GTX 1050 (3GB VRAM) + 22GB RAM + NVMe Streaming
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build-cuda12/strata"
MODEL_FILE="/media/local-ai/Model/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf"
PLE_FILE="/media/local-ai/Model/Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf"
PACK_DIR="$SCRIPT_DIR/pack/flash_next_gyro"
PYTHON_BIN="python3"
export PYTHONPATH="$SCRIPT_DIR/build-cuda12/_deps/strata_llamacpp-src/gguf-py:${PYTHONPATH:-}"
export STRATA_GGUF_PY="$SCRIPT_DIR/build-cuda12/_deps/strata_llamacpp-src/gguf-py"

echo "=== Strata-MoEBoost: Flash-Next Gyro Hardware Preparation ==="
echo "Engine Binary : $BIN"
echo "Model Target  : $MODEL_FILE"
echo "Pack Directory: $PACK_DIR"
echo "============================================================"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Native engine $BIN not found. Compiling first..."
    make -C "$SCRIPT_DIR/build-cuda12" -j4 strata
fi

# Step 1: Check model file availability
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Model file $MODEL_FILE is still downloading or not present."
    echo "Please wait for aria2c download to complete."
    exit 1
fi

mkdir -p "$PACK_DIR"

# Step 2: Unpack dense weights and tokenizer
if [[ ! -f "$PACK_DIR/dense.bin" || ! -f "$PACK_DIR/tokenizer/vocab.json" ]]; then
    echo "[1/3] Unpacking dense attention/shared projections to $PACK_DIR..."
    "$PYTHON_BIN" "$SCRIPT_DIR/tools/iq_pack.py" \
        --gguf "$MODEL_FILE" \
        --out "$PACK_DIR" \
        --compat-bf16 \
        --skip-experts
    echo "Dense layers unpacked successfully."
else
    echo "[1/3] Dense layers already unpacked in $PACK_DIR."
fi

# Step 3: Build 512-expert routing profile
PROFILE_BIN="$PACK_DIR/profile.bin"
if [[ ! -f "$PROFILE_BIN" ]]; then
    echo "[2/3] Generating 512-expert static routing profile..."
    "$PYTHON_BIN" "$SCRIPT_DIR/tools/make_profile.py" \
        --no-base \
        --n-expert 512 \
        --out "$PROFILE_BIN"
    echo "Expert profile generated."
else
    echo "[2/3] Expert profile already exists at $PROFILE_BIN."
fi

echo "[3/3] Ready for launch!"
echo "Run: bash $SCRIPT_DIR/launch_flash_next_lowpc.sh"
