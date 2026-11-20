#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/media/local-ai/Model"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

URL="https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/resolve/main/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf"
FILENAME="Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf"

echo "=== Starting Download of Official Flash-Next (Gyro-S TQ1_0) ==="
echo "Repository : https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
echo "Target Destination: $DEST_DIR"
echo "File       : $FILENAME (~27.6 GB)"
echo "Using multi-connection aria2c for maximum speed and auto-resume..."

aria2c -c -x 8 -s 8 -k 2M --file-allocation=none \
  "$URL"

echo "=== Flash-Next Gyro Download Finished ==="
ls -lh "$DEST_DIR/$FILENAME"
