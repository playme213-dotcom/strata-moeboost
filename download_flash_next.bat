@echo off
rem ==============================================================================
rem download_flash_next.bat - Windows Auto-Downloader for Flash-Next Gyro-S
rem ==============================================================================
setlocal enabledelayedexpansion
title Download Flash-Next Gyro-S

cd /d "%~dp0"
set "DEST_DIR=models"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

set "URL=https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/resolve/main/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf"
set "FILENAME=Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf"

echo ==========================================================
echo  ⚡ Downloading Official Qwen3.8 Flash-Next Gyro-S
echo  File   : %FILENAME% (~27.6 GB)
echo  Output : %DEST_DIR%
echo ==========================================================

where aria2c >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Using aria2c multi-connection auto-resume...
    aria2c -c -x 8 -s 8 -k 2M --file-allocation=none --dir="%DEST_DIR%" "%URL%"
) else (
    where curl >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [+] Downloading via curl...
        curl -L -C - -o "%DEST_DIR%\%FILENAME%" "%URL%"
    ) else (
        echo [!] Neither aria2c nor curl found. Please install aria2c or curl.
        pause
        exit /b 1
    )
)

echo [✓] Flash-Next Gyro download finished!
pause
