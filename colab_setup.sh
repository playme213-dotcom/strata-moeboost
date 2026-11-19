#!/usr/bin/env bash
# ==============================================================================
# colab_setup.sh - All-in-one Google Colab / Remote Cloud Setup for Strata-MoEBoost
# ==============================================================================
# Features:
# 1. Environment & GPU Setup (PyTorch, llama-cpp-python with CUDA support, FastAPI, uvicorn)
# 2. Universal Model Downloader & Selection (MoE series + HotRod Flash-Next)
# 3. Automatic Server Launch with Speculative Drafting / MTP
# 4. Optional Cloudflare / LocalTunnel URL for Web GUI
# 5. Built-in Terminal Chatbox on the last part (Download -> Server -> Load -> Chat)
# ==============================================================================

set -e

# Detect workspace root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=================================================================="
echo "   ⚡ Strata-MoEBoost Google Colab / Remote One-Click Setup"
echo "=================================================================="

# Check GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[+] GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    CUDA_AVAILABLE=1
else
    echo "[!] No NVIDIA GPU detected. Running in CPU-only fallback mode."
    CUDA_AVAILABLE=0
fi

# Step 1: Install dependencies
echo ""
echo "[Step 1/5] Checking and installing Python dependencies..."
python3 -m pip install --upgrade pip
if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install llama-cpp-python --no-cache-dir --force-reinstall || python3 -m pip install llama-cpp-python
else
    python3 -m pip install llama-cpp-python
fi
python3 -m pip install fastapi uvicorn pydantic psutil requests gguf jinja2 py-cpuinfo aria2

# Step 2: Model directory setup
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models}"
mkdir -p "$MODEL_DIR"

# Step 3: Model List & Selection
echo ""
echo "[Step 2/5] Model Selection:"
echo "------------------------------------------------------------------"
echo " 1) Cerebras Qwen3-Coder-REAP-25B-A3B IQ4_XS (12.5GB) - Recommended MoE"
echo " 2) Cerebras Qwen3-Coder-REAP-25B-A3B Q2_K (8.1GB) - Low VRAM MoE"
echo " 3) Qwen3-Coder-Next IQ2_XXS (17.9GB) - 80B MoE Next-Gen Flagship"
echo " 4) Qwen1.5-MoE-A2.7B-Chat Q2_K (5.5GB) - Ultra-Fast Lightweight MoE"
echo " 5) DeepSeek-Coder-V2-Lite-Instruct Q4_K_M (9.6GB) - Coding MoE"
echo " 6) Mixtral-8x7B-Instruct-v0.1 Q3_K_M (22.5GB) - 8x7B Standard MoE"
echo " 7) Existing local model (Specify path)"
echo "------------------------------------------------------------------"

if [ -n "$1" ]; then
    MODEL_CHOICE="$1"
else
    read -r -p "Select model to download and load [1-7] (default 1): " MODEL_CHOICE || MODEL_CHOICE="1"
fi

SELECTED_URL=""
SELECTED_FILE=""

case "$MODEL_CHOICE" in
    2)
        SELECTED_URL="https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
        SELECTED_FILE="$MODEL_DIR/cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
        ;;
    3)
        SELECTED_URL="https://huggingface.co/bartowski/Qwen_Qwen3-Coder-Next-GGUF/resolve/main/Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf"
        SELECTED_FILE="$MODEL_DIR/Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf"
        ;;
    4)
        SELECTED_URL="https://huggingface.co/tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf"
        SELECTED_FILE="$MODEL_DIR/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf"
        ;;
    5)
        SELECTED_URL="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
        SELECTED_FILE="$MODEL_DIR/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
        ;;
    6)
        SELECTED_URL="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
        SELECTED_FILE="$MODEL_DIR/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
        ;;
    7)
        read -r -p "Enter full path to your existing GGUF model: " CUSTOM_PATH
        SELECTED_FILE="$CUSTOM_PATH"
        ;;
    1|*)
        SELECTED_URL="https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
        SELECTED_FILE="$MODEL_DIR/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
        ;;
esac

# Step 4: Download model if needed
echo ""
echo "[Step 3/5] Checking / Downloading Model..."
if [ -n "$SELECTED_URL" ] && [ ! -f "$SELECTED_FILE" ]; then
    echo "[+] Downloading from: $SELECTED_URL"
    FILENAME="$(basename "$SELECTED_FILE")"
    
    # Build optional auth headers if HF_TOKEN is present
    HF_HEADER=()
    WGET_HEADER=()
    CURL_HEADER=()
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "[+] Authenticating with provided HF_TOKEN..."
        HF_HEADER=(--header="Authorization: Bearer $HF_TOKEN")
        WGET_HEADER=(--header="Authorization: Bearer $HF_TOKEN")
        CURL_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
    fi

    # Try downloading with aria2c, wget, or curl with resilient fallbacks
    DL_SUCCESS=0
    if command -v aria2c >/dev/null 2>&1; then
        echo "[+] Attempting download with aria2c..."
        aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
            --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${HF_HEADER[@]}" \
            --dir="$MODEL_DIR" --out="$FILENAME" "$SELECTED_URL" && DL_SUCCESS=1 || true
    fi

    if [ "$DL_SUCCESS" -ne 1 ] && command -v wget >/dev/null 2>&1; then
        echo "[+] Fallback: Attempting download with wget..."
        wget -c --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${WGET_HEADER[@]}" \
            -O "$SELECTED_FILE" "$SELECTED_URL" && DL_SUCCESS=1 || true
    fi

    if [ "$DL_SUCCESS" -ne 1 ]; then
        echo "[+] Fallback: Attempting download with curl..."
        curl -L -C - --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${CURL_HEADER[@]}" \
            -o "$SELECTED_FILE" "$SELECTED_URL" && DL_SUCCESS=1 || true
    fi
else
    echo "[✓] Using model file: $SELECTED_FILE"
fi

if [ ! -f "$SELECTED_FILE" ]; then
    echo "[-] Error: Model file not found at $SELECTED_FILE"
    exit 1
fi

# Step 5: Launch Strata-MoEBoost server in background
echo ""
echo "[Step 4/5] Starting Strata-MoEBoost Server (with MTP / Speculative Drafting)..."
PORT="${PORT:-4133}"
SERVER_LOG="$ROOT_DIR/server.log"

# Kill any existing server on this port
pkill -f "strata_moeboost_server.py.*$PORT" 2>/dev/null || true
sleep 1

# Launch server
python3 strata_moeboost_server.py --model "$SELECTED_FILE" --port "$PORT" --mtp auto > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[+] Server started (PID: $SERVER_PID). Waiting for model initialization..."

# Health check loop
MAX_WAIT=120
WAITED=0
SERVER_READY=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ready":true'; then
        SERVER_READY=1
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done
echo ""

if [ $SERVER_READY -ne 1 ]; then
    echo "[-] Server startup timed out or failed. Last 30 lines of log:"
    tail -n 30 "$SERVER_LOG"
    exit 1
fi

echo "[✓] Strata-MoEBoost Server is Ready on http://127.0.0.1:$PORT"

# Step 6: Interactive Terminal Chatbox on the last part
echo ""
echo "=================================================================="
echo "   ⚡ Strata-MoEBoost Interactive Chatbox"
echo "=================================================================="
echo " Model : $(basename "$SELECTED_FILE")"
echo " Server: http://127.0.0.1:$PORT"
echo " Type '/exit' or '/quit' to close chat. Type '/reset' to clear history."
echo "=================================================================="
echo ""

python3 - <<EOF
import sys
import json
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:$PORT/v1/chat/completions"
history = []

print("⚡ Chatbox ready! Enter your message below:")

while True:
    try:
        user_input = input("\nYou > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting chat.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("/exit", "/quit"):
        print("Goodbye!")
        break
    if user_input.lower() == "/reset":
        history = []
        print("[*] Conversation cleared.")
        continue

    history.append({"role": "user", "content": user_input})
    payload = {
        "messages": history,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    print("\nMoEBoost > ", end="", flush=True)
    full_response = ""

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta
                except Exception:
                    pass
        print()
        history.append({"role": "assistant", "content": full_response})
    except Exception as e:
        print(f"\n[Error communicating with server: {e}]")
EOF
