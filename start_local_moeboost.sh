#!/usr/bin/env bash
# ==============================================================================
# Strata-MoEBoost Server Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# Default Model and Options
MODEL_FILE="${1:-}"
PORT="${2:-4133}"
HOST="${3:-127.0.0.1}"

echo "=================================================================="
echo " Starting Strata-MoEBoost Server"
echo " Model : ${MODEL_FILE:-auto-detect from ~/strata-moeboost/model}"
echo " Host  : $HOST"
echo " Port  : $PORT"
echo " Directory: $SCRIPT_DIR"
echo "=================================================================="

# Check Python environment
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Virtual environment python not found at $PYTHON_BIN"
    if command -v python3 >/dev/null 2>&1; then
        echo "Falling back to system python3..."
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Error: Python 3 is required but not installed." >&2
        exit 1
    fi
fi

# Check required python modules and install missing if needed
echo "Checking required Python modules..."
MISSING_MODULES=$("$PYTHON_BIN" - << 'EOF'
import sys
required = [
    'uvicorn', 'fastapi', 'pydantic', 'psutil', 'requests',
    'tqdm', 'yaml', 'jinja2', 'numpy', 'llama_cpp', 'websockets', 'starlette'
]
missing = []
for mod in required:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)

if missing:
    print(" ".join(missing))
EOF
)

if [[ -n "$MISSING_MODULES" ]]; then
    echo "Installing missing modules: $MISSING_MODULES"
    "$PYTHON_BIN" -m pip install --upgrade pip || true
    "$PYTHON_BIN" -m pip install $MISSING_MODULES
else
    echo "All required modules are present."
fi

# Check model existence
if [[ -n "$MODEL_FILE" && ! -f "$MODEL_FILE" ]]; then
    echo "Warning: Model file not found at $MODEL_FILE"
    echo "Proceeding with the explicit path; verify it before loading."
fi

cd "$SCRIPT_DIR"

ARGS=(--host "$HOST" --port "$PORT" --low-spec --ctx 2048)
if [[ -n "$MODEL_FILE" ]]; then
    ARGS+=(--model "$MODEL_FILE")
fi

exec "$PYTHON_BIN" strata_moeboost_server.py "${ARGS[@]}"
