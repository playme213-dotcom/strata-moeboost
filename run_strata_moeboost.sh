#!/usr/bin/env bash
# Strata-MoEBoost by Gravitize.IGP Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="python"
fi

MODEL="${1:-}"
PORT="${2:-4133}"
GPU_LAYERS="$3"
THREADS="$4"
BATCH="$5"
LOW_SPEC_FLAG=""

if [ "$6" == "--low-spec" ] || [ "$1" == "--low-spec" ]; then
    LOW_SPEC_FLAG="--low-spec"
fi

echo "=========================================================="
echo " Starting Strata-MoEBoost by Gravitize.IGP"
echo " Model:       ${MODEL:-auto-detect from ~/strata-moeboost/model}"
echo " Port:        $PORT"
echo " Low-Spec:    ${LOW_SPEC_FLAG:-auto-detecting hardware profile}"
echo "=========================================================="

ARGS=("--port" "$PORT")
[ -n "$MODEL" ] && ARGS+=("--model" "$MODEL")
[ -n "$GPU_LAYERS" ] && ARGS+=("--gpu-layers" "$GPU_LAYERS")
[ -n "$THREADS" ] && ARGS+=("--threads" "$THREADS")
[ -n "$BATCH" ] && ARGS+=("--batch" "$BATCH")
[ -n "$LOW_SPEC_FLAG" ] && ARGS+=("$LOW_SPEC_FLAG")

exec "$PYTHON_BIN" "$DIR/strata_moeboost_server.py" "${ARGS[@]}"
