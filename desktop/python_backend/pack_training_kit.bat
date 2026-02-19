@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Pack Training Kit — copies all required files into training_kit\
REM  Run this ONCE from the python_backend directory to assemble the kit.
REM ═══════════════════════════════════════════════════════════════

echo.
echo  Packing BaseJamba Training Kit...
echo.

REM Copy training scripts
echo [1/4] Copying training scripts...
copy /Y "train_mamba.py" "training_kit\train_mamba.py"
copy /Y "mamba_model.py" "training_kit\mamba_model.py"

REM Copy data files
echo [2/4] Copying data files (this may take a minute for the 428MB dataset)...
if not exist "training_kit\data\pretrain" mkdir "training_kit\data\pretrain"
copy /Y "data\pretrain\familylinks_btc_1m.parquet" "training_kit\data\pretrain\"
copy /Y "data\eth_usdt.parquet" "training_kit\data\"

REM Create output directories
echo [3/4] Creating output directories...
if not exist "training_kit\models\checkpoints" mkdir "training_kit\models\checkpoints"

echo [4/4] Done!
echo.
echo ═══════════════════════════════════════════════════════════════
echo  Training kit ready at: training_kit\
echo.
echo  Contents:
echo    train_mamba.py         (training script with all fixes)
echo    mamba_model.py         (Jamba model architecture)
echo    train.bat              (one-click launcher)
echo    push_model.py          (HuggingFace upload)
echo    requirements.txt       (pip dependencies)
echo    TRAINING_README.md     (instructions)
echo    data\                  (dataset files ~436 MB)
echo.
echo  Transfer the entire training_kit\ folder to the training PC.
echo  Then follow TRAINING_README.md to start training.
echo ═══════════════════════════════════════════════════════════════
pause
