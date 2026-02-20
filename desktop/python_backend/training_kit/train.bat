@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Jamba Training Launcher (supports all architectures)
REM  Usage: train.bat [small|lite|medium|large]
REM ═══════════════════════════════════════════════════════════════
REM
REM  Examples:
REM    train.bat              → SmallJamba (4.4M params)
REM    train.bat lite         → LiteJamba  (9.7M params)
REM    train.bat medium       → MediumJamba (28M params)
REM
REM  Outputs saved to:
REM    models\nexus_<arch>_jamba_v1.pth  (best model)
REM    models\mamba_scaler.pkl           (feature scaler)
REM    models\mamba_revin.pth            (RevIN state)
REM    models\checkpoints\               (per-epoch checkpoints)
REM

REM Parse architecture from first argument (default: small)
set ARCH=%1
if "%ARCH%"=="" set ARCH=small

echo.
echo  ╔═══════════════════════════════════════════════╗
echo  ║   Jamba Training Kit v2.0                     ║
echo  ║   Architecture: %ARCH%                        ║
echo  ║   Data:  BTC 1m candles + ETH cross-asset     ║
echo  ╚═══════════════════════════════════════════════╝
echo.

REM Enable expandable segments for less VRAM fragmentation.
REM This is critical for preventing OOM on 16 GB GPUs.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Run training
python train_mamba.py ^
    --arch %ARCH% ^
    --epochs 25 ^
    --lr 1e-4 ^
    --stride 5 ^
    --skip-download ^
    2>&1 | tee train_%ARCH%_jamba.log

echo.
echo ═══════════════════════════════════════════════
echo  Training complete! Check train_%ARCH%_jamba.log
echo  Best model saved to: models\nexus_%ARCH%_jamba_v1.pth
echo ═══════════════════════════════════════════════
pause
