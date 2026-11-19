@echo off
rem ==============================================================================
rem run_strata_moeboost.bat - Windows Launcher for Strata-MoEBoost
rem ==============================================================================
setlocal enabledelayedexpansion
title Strata-MoEBoost Server

cd /d "%~dp0"

set "PYTHON_BIN="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul && set "PYTHON_BIN=python"
)

if not defined PYTHON_BIN (
    echo [!] Python not found. Run START-HERE.bat first to set up the environment.
    pause
    exit /b 1
)

set "MODEL=%~1"
if not defined MODEL (
    if exist "models\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf" (
        set "MODEL=models\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
    ) else if exist "C:\Model\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf" (
        set "MODEL=C:\Model\cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
    ) else (
        set "MODEL=cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
    )
)

set "PORT=%~2"
if not defined PORT set "PORT=4133"

echo ==========================================================
echo  ⚡ Starting Strata-MoEBoost Server
echo  Model: %MODEL%
echo  Port:  %PORT%
echo ==========================================================

"%PYTHON_BIN%" strata_moeboost_server.py --model "%MODEL%" --port "%PORT%" --mtp auto %*
if errorlevel 1 (
    echo [!] Server exited with error code %ERRORLEVEL%.
    pause
)
exit /b %ERRORLEVEL%
