@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM   TRAIN_WSL2.bat — Launch Jamba training in WSL2 Ubuntu (Triton+torch.compile)
REM
REM   Usage:  TRAIN_WSL2.bat [--arch small] [--skip-download] [--epochs 15]
REM           All arguments are passed through to train_mamba.py
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║  NEXUS TRAINING — WSL2 Linux (torch.compile + Triton)       ║
echo ║  GPU: NVIDIA RTX 5080 via CUDA passthrough                  ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Default args if none provided
set ARGS=%*
if "%ARGS%"=="" set ARGS=--arch small --skip-download

echo [*] Activating venv and launching training...
echo [*] Args: %ARGS%
echo.

wsl -d Ubuntu-22.04 -- bash -c "source /opt/nexus-train/bin/activate && cd /mnt/f/Predictor/desktop/python_backend && python3 train_mamba.py %ARGS%"

echo.
echo [*] Training complete. Press any key to exit.
pause > nul
