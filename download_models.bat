@echo off
rem ==============================================================================
rem download_models.bat - Windows Model Downloader for Strata-MoEBoost
rem ==============================================================================
setlocal enabledelayedexpansion
title Strata-MoEBoost Model Downloader

cd /d "%~dp0"
set "DEST_DIR=models"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

echo ==================================================================
echo    ⚡ Strata-MoEBoost Model Downloader (Windows)
echo ==================================================================
echo  Destination Directory: %DEST_DIR%
echo ------------------------------------------------------------------
echo  [MoE Coding & Reasoning Family]
echo   [1] Cerebras Qwen3-Coder-REAP-25B-A3B IQ4_XS (12.5 GB - Recommended)
echo   [2] Cerebras Qwen3-Coder-REAP-25B-A3B Q2_K (8.1 GB - Low VRAM)
echo   [3] Qwen3-Coder-Next IQ2_XXS (17.9 GB - 80B MoE Flagship)
echo   [4] Qwen1.5-MoE-A2.7B-Chat Q2_K (5.5 GB - Ultra Compact)
echo   [5] DeepSeek-Coder-V2-Lite-Instruct Q4_K_M (9.6 GB)
echo   [6] Mixtral-8x7B-Instruct-v0.1 Q3_K_M (22.5 GB)
echo ------------------------------------------------------------------
echo   [C] Custom HuggingFace / GGUF URL
echo   [Q] Quit
echo ==================================================================

set /p CHOICE="Enter choice [1-6, C, Q]: "

if /i "%CHOICE%"=="1" (
    set "URL=https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
    set "OUTFILE=%DEST_DIR%\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
) else if /i "%CHOICE%"=="2" (
    set "URL=https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
    set "OUTFILE=%DEST_DIR%\cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
) else if /i "%CHOICE%"=="3" (
    set "URL=https://huggingface.co/bartowski/Qwen_Qwen3-Coder-Next-GGUF/resolve/main/Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf"
    set "OUTFILE=%DEST_DIR%\Qwen_Qwen3-Coder-Next-IQ2_XXS.gguf"
) else if /i "%CHOICE%"=="4" (
    set "URL=https://huggingface.co/tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf"
    set "OUTFILE=%DEST_DIR%\Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf"
) else if /i "%CHOICE%"=="5" (
    set "URL=https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    set "OUTFILE=%DEST_DIR%\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
) else if /i "%CHOICE%"=="6" (
    set "URL=https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
    set "OUTFILE=%DEST_DIR%\mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
) else if /i "%CHOICE%"=="C" (
    set /p URL="Enter direct GGUF download URL: "
    set "OUTFILE=%DEST_DIR%\custom_model.gguf"
) else (
    echo Exiting.
    exit /b 0
)

echo.
echo [+] Downloading !URL! to !OUTFILE!...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -Uri '!URL!' -OutFile '!OUTFILE!'"
if errorlevel 1 (
    echo [!] Download failed via PowerShell. Trying curl...
    curl -L -C - -o "!OUTFILE!" "!URL!"
)

echo [✓] Done! To start Strata-MoEBoost with this model:
echo     run_strata_moeboost.bat "!OUTFILE!"
pause
