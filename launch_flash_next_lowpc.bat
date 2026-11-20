@echo off
rem ==============================================================================
rem launch_flash_next_lowpc.bat - Windows Low-PC Launcher for Flash-Next
rem ==============================================================================
setlocal enabledelayedexpansion
title Strata-MoEBoost Flash-Next Low-PC Launcher

cd /d "%~dp0"

set "MODEL_SHARD1=%~1"
if not defined MODEL_SHARD1 (
    if exist "\media\local-ai\Model\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" (
        set "MODEL_SHARD1=\media\local-ai\Model\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    ) else if exist "models\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" (
        set "MODEL_SHARD1=models\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    ) else (
        set "MODEL_SHARD1=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    )
)

set "PORT=%~2"
if not defined PORT set "PORT=4133"

echo ==========================================================
echo  ⚡ Strata-MoEBoost Flash-Next Low-PC Launcher
echo  Model: %MODEL_SHARD1%
echo  Port:  %PORT%
echo  Mode:  Low-Spec Auto Tuned (mmap experts + 3-tier memory)
echo ==========================================================

call run_strata_moeboost.bat "%MODEL_SHARD1%" "%PORT%" 4 4 128 --low-spec
if errorlevel 1 (
    echo [!] Server exited with error code %ERRORLEVEL%.
    pause
)
exit /b %ERRORLEVEL%
