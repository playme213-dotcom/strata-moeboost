#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/media/local-ai/Model"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

URL_BASE="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/IQ1_M"
SHARD1="Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
SHARD2="Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"

echo "=== Starting Download of Flash-Next Coder (IQ1_M) ==="
echo "Target Destination: $DEST_DIR"
echo "Shard 1: $SHARD1 (~27.6 GB)"
echo "Shard 2: $SHARD2 (~26.8 GB)"
echo "Using multi-connection aria2c for maximum speed and auto-resume..."

aria2c -c -x 8 -s 8 -k 2M --file-allocation=none \
  "$URL_BASE/$SHARD1" \
  "$URL_BASE/$SHARD2"

echo "=== Flash-Next Download Finished ==="
ls -lh "$DEST_DIR/$SHARD1" "$DEST_DIR/$SHARD2"
