#!/usr/bin/env bash
# Download official decoupled PLE N-Gram Table for Qwen3.8-Flash-Next
set -euo pipefail

DEST_DIR="/media/local-ai/Model"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

URL="https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF/resolve/main/Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf"
OUTPUT="Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf"

echo "=== Downloading Standalone PLE Table (IQ4_NL ~26.8 GB) ==="
echo "Destination: $DEST_DIR/$OUTPUT"

if command -v aria2c >/dev/null 2>&1; then
    exec aria2c -c -x 8 -s 8 -k 2M --file-allocation=none -o "$OUTPUT" "$URL"
else
    exec curl -L -C - -o "$OUTPUT" "$URL"
fi
