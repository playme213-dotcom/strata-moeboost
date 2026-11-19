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
echo  [HotRod Flash-Next Family]
echo   [1] Qwen3.8-Flash-Next IQ3_S (HotRod Original Native - 11.5 GB)
echo   [2] Swift-1.5-Qwen3.8-Flash-Next IQ3_S (11.5 GB)
echo.
echo  [MoE Coding ^& Reasoning Family]
echo   [3] Cerebras Qwen3-Coder-REAP-25B-A3B IQ4_XS (12.5 GB - Recommended)
echo   [4] Cerebras Qwen3-Coder-REAP-25B-A3B Q2_K (8.7 GB - Low VRAM)
echo   [5] DeepSeek-Coder-V2-Lite-Instruct Q4_K_M (9.6 GB)
echo   [6] Mixtral-8x7B-Instruct-v0.1 Q3_K_M (22.5 GB)
echo ------------------------------------------------------------------
echo   [C] Custom HuggingFace / GGUF URL
echo   [Q] Quit
echo ==================================================================

set /p CHOICE="Enter choice [1-6, C, Q]: "

if /i "%CHOICE%"=="1" (
    set "URL=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-IQ3_S.gguf"
    set "OUTFILE=%DEST_DIR%\Qwen3.8-Flash-Next-IQ3_S.gguf"
) else if /i "%CHOICE%"=="2" (
    set "URL=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-Flash-Next-IQ3_S.gguf"
    set "OUTFILE=%DEST_DIR%\Swift-1.5-Qwen3.8-Flash-Next-IQ3_S.gguf"
) else if /i "%CHOICE%"=="3" (
    set "URL=https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
    set "OUTFILE=%DEST_DIR%\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
) else if /i "%CHOICE%"=="4" (
    set "URL=https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B-GGUF/resolve/main/cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
    set "OUTFILE=%DEST_DIR%\cerebras_Qwen3-Coder-REAP-25B-A3B-Q2_K.gguf"
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
