@echo off
REM ═══════════════════════════════════════════════════════════════
REM  BaseJamba Training Launcher
REM  Target: SmallJamba (4.4M params) — 25 epochs with early stop
REM ═══════════════════════════════════════════════════════════════
REM
REM  Usage:
REM    1) Copy this entire training_kit\ folder to the training PC
REM    2) Install Python 3.12+ and CUDA-enabled PyTorch
REM    3) pip install -r requirements.txt
REM    4) Double-click this file or run: train.bat
REM
REM  Outputs saved to:
REM    models\nexus_small_jamba_v1.pth    (best model)
REM    models\mamba_scaler.pkl           (feature scaler)
REM    models\mamba_revin.pth            (RevIN state)
REM    models\checkpoints\               (per-epoch checkpoints)
REM

echo.
echo  ╔═══════════════════════════════════════════════╗
echo  ║   BaseJamba Training Kit v1.0                 ║
echo  ║   Model: SmallJamba (4.4M params)             ║
echo  ║   Data:  BTC 1m candles + ETH cross-asset     ║
echo  ╚═══════════════════════════════════════════════╝
echo.

REM Enable TF32 and expandable segments for less VRAM fragmentation
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Run training
python train_mamba.py ^
    --arch small ^
    --epochs 25 ^
    --lr 1e-4 ^
    --stride 5 ^
    --skip-download ^
    2>&1 | tee train_basejamba.log

echo.
echo ═══════════════════════════════════════════════
echo  Training complete! Check train_basejamba.log
echo  Best model saved to: models\nexus_small_jamba_v1.pth
echo ═══════════════════════════════════════════════
pause
