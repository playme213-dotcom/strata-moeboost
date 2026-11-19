#!/usr/bin/env bash
# Strata-MoEBoost by Gravitize.IGP Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON_BIN="/home/local-ai/strata-moeboost/.venv/bin/python"

MODEL="${1:-/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf}"
PORT="${2:-8080}"
GPU_LAYERS="$3"
THREADS="$4"
BATCH="$5"
LOW_SPEC_FLAG=""

if [ "$6" == "--low-spec" ] || [ "$1" == "--low-spec" ]; then
    LOW_SPEC_FLAG="--low-spec"
fi

echo "=========================================================="
echo " Starting Strata-MoEBoost by Gravitize.IGP"
echo " Model:       $MODEL"
echo " Port:        $PORT"
echo " Low-Spec:    ${LOW_SPEC_FLAG:-auto-detecting hardware profile}"
echo "=========================================================="

ARGS=("--model" "$MODEL" "--port" "$PORT")
[ -n "$GPU_LAYERS" ] && ARGS+=("--gpu-layers" "$GPU_LAYERS")
[ -n "$THREADS" ] && ARGS+=("--threads" "$THREADS")
[ -n "$BATCH" ] && ARGS+=("--batch" "$BATCH")
[ -n "$LOW_SPEC_FLAG" ] && ARGS+=("$LOW_SPEC_FLAG")

exec "$PYTHON_BIN" "$DIR/strata_moeboost_server.py" "${ARGS[@]}"
