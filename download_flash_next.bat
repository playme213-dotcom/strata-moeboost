@echo off
rem ==============================================================================
rem download_flash_next.bat - Windows Auto-Downloader for Flash-Next Coder (IQ1_M)
rem ==============================================================================
setlocal enabledelayedexpansion
title Download Flash-Next Coder 2-Shard Bundle

cd /d "%~dp0"
set "DEST_DIR=models"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

set "URL_BASE=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/IQ1_M"
set "SHARD1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
set "SHARD2=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"

echo ==========================================================
echo  ⚡ Downloading Qwen3.8 Flash-Next Coder (IQ1_M Bundle)
echo  Shard 1: %SHARD1% (~27.6 GB)
echo  Shard 2: %SHARD2% (~26.8 GB)
echo  Output : %DEST_DIR%
echo ==========================================================

where aria2c >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Using aria2c multi-connection auto-resume...
    aria2c -c -x 8 -s 8 -k 2M --file-allocation=none --dir="%DEST_DIR%" "%URL_BASE%/%SHARD1%" "%URL_BASE%/%SHARD2%"
) else (
    where curl >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [+] Downloading Shard 1 via curl...
        curl -L -C - -o "%DEST_DIR%\%SHARD1%" "%URL_BASE%/%SHARD1%"
        echo [+] Downloading Shard 2 via curl...
        curl -L -C - -o "%DEST_DIR%\%SHARD2%" "%URL_BASE%/%SHARD2%"
    ) else (
        echo [!] Neither aria2c nor curl found. Please install aria2c or curl.
        pause
        exit /b 1
    )
)

echo [✓] Flash-Next download finished!
pause
