#!/usr/bin/env bash
# Strata-MoEBoost by Gravitize.IGP Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON_BIN="/home/local-ai/strata-moeboost/.venv/bin/python"

MODEL="${1:-/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf}"
PORT="${2:-8080}"
GPU_LAYERS="${3:-4}"
THREADS="${4:-8}"

echo "=========================================================="
echo " Starting Strata-MoEBoost by Gravitize.IGP"
echo " Model:      $MODEL"
echo " Port:       $PORT"
echo " GPU Layers: $GPU_LAYERS (fits GTX 1050 3GB / GTX 1650 4GB)"
echo " CPU Threads: $THREADS (optimal for 24GB+ RAM Ryzen/Intel)"
echo "=========================================================="

exec "$PYTHON_BIN" "$DIR/strata_moeboost_server.py" \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS"
