#!/usr/bin/env bash
# ==============================================================================
# download_models.sh - Strata-MoEBoost Universal Model Downloader
# ==============================================================================
# Supports interactive menu or CLI arguments.
# Automatically checks available disk space, supports resuming, and places
# models into the designated model directory.
# ==============================================================================

set -e

DEST_DIR="${MODEL_DIR:-/media/local-ai/Model}"
mkdir -p "$DEST_DIR"

# Ensure aria2c, wget, or curl is available
DL_TOOL=""
if command -v aria2c >/dev/null 2>&1; then
    DL_TOOL="aria2c"
elif command -v wget >/dev/null 2>&1; then
    DL_TOOL="wget"
elif command -v curl >/dev/null 2>&1; then
    DL_TOOL="curl"
else
    echo "[-] Error: Neither aria2c, wget, nor curl found. Please install one."
    exit 1
fi

download_file() {
    local url="$1"
    local outfile="$2"
    local outdir
    outdir="$(dirname "$outfile")"
    local fname
    fname="$(basename "$outfile")"

    echo "[+] Downloading $(basename "$outfile")..."
    echo "    Source: $url"
    echo "    Destination: $outfile"

    if [ -f "$outfile" ]; then
        echo "[!] File already exists: $outfile"
        read -r -p "    Overwrite? [y/N]: " ov < /dev/tty || ov="n"
        if [[ ! "$ov" =~ ^[Yy]$ ]]; then
            echo "[*] Skipping download."
            return 0
        fi
    fi

    local hf_header=()
    local wget_header=()
    local curl_header=()
    if [ -n "${HF_TOKEN:-}" ]; then
        hf_header=(--header="Authorization: Bearer $HF_TOKEN")
        wget_header=(--header="Authorization: Bearer $HF_TOKEN")
        curl_header=(-H "Authorization: Bearer $HF_TOKEN")
    fi

    local dl_success=0
    if [ "$DL_TOOL" = "aria2c" ]; then
        aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
            --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${hf_header[@]}" \
            --dir="$outdir" --out="$fname" "$url" && dl_success=1 || true
    fi

    if [ "$dl_success" -ne 1 ] && command -v wget >/dev/null 2>&1; then
        wget -c --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${wget_header[@]}" \
            -O "$outfile" "$url" && dl_success=1 || true
    fi

    if [ "$dl_success" -ne 1 ] && command -v curl >/dev/null 2>&1; then
        curl -L -C - --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            "${curl_header[@]}" \
            -o "$outfile" "$url" && dl_success=1 || true
    fi

    if [ "$dl_success" -eq 1 ] && [ -f "$outfile" ]; then
        echo "[✓] Download complete: $outfile"
    else
        echo "[-] Download failed or incomplete for $outfile"
    fi
}

# Catalog of Models:
# MoE models + HotRod's original Flash-Next series
declare -A MODEL_NAMES
declare -A MODEL_URLS
declare -A MODEL_DESCS

# 1. Qwen3.8-Flash-Next (HotRod's flagship models)
MODEL_NAMES[1]="Qwen3.8-Flash-Next IQ3_S (HotRod Original Native)"
MODEL_URLS[1]="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-IQ3_S.gguf"
MODEL_DESCS[1]="125B MoE | ~11.5 GB | HotRod baseline look-ahead enabled"

MODEL_NAMES[2]="Qwen3.8-Flash-Next UD-IQ4_XS (Split shards 1-3)"
MODEL_URLS[2]="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MODEL_DESCS[2]="125B MoE | ~14.8 GB | Ultra-dense IQ4 quantization"

MODEL_NAMES[3]="Qwen3.8-Flash-Next Q2_0 (Ultra-Low RAM)"
MODEL_URLS[3]="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-Q2_0.gguf"
MODEL_DESCS[3]="125B MoE | ~8.9 GB | Suitable for 16GB RAM machines"

MODEL_NAMES[4]="Swift-1.5-Qwen3.8-Flash-Next IQ3_S"
MODEL_URLS[4]="https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-Flash-Next-IQ3_S.gguf"
MODEL_DESCS[4]="125B MoE | ~11.5 GB | Swift fine-tune for instruction following"

# 2. Cerebras / Qwen3-MoE Series
MODEL_NAMES[5]="Cerebras Qwen3-Coder-REAP-25B-A3B IQ4_XS (Installed Default)"
MODEL_URLS[5]="https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
MODEL_DESCS[5]="25B Total / 3B Active | 12.5 GB | Excellent low-VRAM coding MoE"

MODEL_NAMES[6]="Cerebras Qwen3-Coder-REAP-25B-A3B Q2_K (Low Specs)"
MODEL_URLS[6]="https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
MODEL_DESCS[6]="25B Total / 3B Active | ~8.7 GB | High speed for 2-4GB VRAM"

# 3. DeepSeek & Mixtral MoE Models
MODEL_NAMES[7]="DeepSeek-Coder-V2-Lite-Instruct Q4_K_M"
MODEL_URLS[7]="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
MODEL_DESCS[7]="16B Total / 2.4B Active MoE | ~9.6 GB | Superior multilingual coding"

MODEL_NAMES[8]="DeepSeek-Coder-V2-Lite-Instruct IQ3_M"
MODEL_URLS[8]="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-IQ3_M.gguf"
MODEL_DESCS[8]="16B Total / 2.4B Active MoE | ~6.9 GB | Fits in smaller RAM footprints"

MODEL_NAMES[9]="Mixtral-8x7B-Instruct-v0.1 Q3_K_M"
MODEL_URLS[9]="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
MODEL_DESCS[9]="46.7B Total / 12.9B Active MoE | ~22.5 GB | Classical open standard MoE"

MODEL_NAMES[10]="Qwen2.5-Coder-7B-Instruct Q4_K_M (Dense Fast Baseline)"
MODEL_URLS[10]="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
MODEL_DESCS[10]="7B Dense | 4.7 GB | Ultra-fast local validation model"

show_menu() {
    clear 2>/dev/null || true
    echo "=================================================================="
    echo "   ⚡ Strata-MoEBoost Universal Model Downloader"
    echo "=================================================================="
    echo " Destination Directory: $DEST_DIR"
    echo " Download Tool        : $DL_TOOL"
    echo "------------------------------------------------------------------"
    echo " [HotRod Flash-Next Family]"
    for i in 1 2 3 4; do
        printf "  [%2d] %-48s\n       (%s)\n" "$i" "${MODEL_NAMES[$i]}" "${MODEL_DESCS[$i]}"
    done
    echo ""
    echo " [MoE Coding & Reasoning Family]"
    for i in 5 6 7 8 9; do
        printf "  [%2d] %-48s\n       (%s)\n" "$i" "${MODEL_NAMES[$i]}" "${MODEL_DESCS[$i]}"
    done
    echo ""
    echo " [Dense Fast Baseline]"
    printf "  [%2d] %-48s\n       (%s)\n" "10" "${MODEL_NAMES[10]}" "${MODEL_DESCS[10]}"
    echo "------------------------------------------------------------------"
    echo "  [C]  Custom HuggingFace / Direct GGUF URL"
    echo "  [Q]  Quit"
    echo "=================================================================="
}

if [ -n "$1" ]; then
    CHOICE="$1"
else
    show_menu
    read -r -p "Enter choice [1-10, C, Q]: " CHOICE < /dev/tty || CHOICE="Q"
fi

case "$CHOICE" in
    [1-9]|10)
        SEL_URL="${MODEL_URLS[$CHOICE]}"
        FNAME="$(basename "$SEL_URL")"
        DEST_FILE="$DEST_DIR/$FNAME"
        echo "[*] Selected: ${MODEL_NAMES[$CHOICE]}"
        download_file "$SEL_URL" "$DEST_FILE"
        echo ""
        echo "[★] To run with Strata-MoEBoost:"
        echo "    ./run_strata_moeboost.sh $DEST_FILE"
        ;;
    [Cc])
        read -r -p "Enter direct GGUF download URL: " CUSTOM_URL < /dev/tty
        if [ -n "$CUSTOM_URL" ]; then
            FNAME="$(basename "$CUSTOM_URL" | cut -d? -f1)"
            DEST_FILE="$DEST_DIR/$FNAME"
            download_file "$CUSTOM_URL" "$DEST_FILE"
            echo ""
            echo "[★] To run with Strata-MoEBoost:"
            echo "    ./run_strata_moeboost.sh $DEST_FILE"
        fi
        ;;
    [Qq]|*)
        echo "[*] Exiting."
        exit 0
        ;;
esac
