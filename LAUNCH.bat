@echo off
chcp 65001 >nul 2>&1
title Nexus Shadow-Quant — Booting...
mode con cols=72 lines=35
color 0B

:: ══════════════════════════════════════════════════════════════════
::   NEXUS SHADOW-QUANT — PREMIUM BOOT SEQUENCE
:: ══════════════════════════════════════════════════════════════════

cls
echo.
echo   [38;5;39m╔══════════════════════════════════════════════════════════════╗[0m
echo   [38;5;39m║[0m                                                              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;51m ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;51m ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;87m ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;123m ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;159m ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m  [38;5;195m ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝[0m              [38;5;39m║[0m
echo   [38;5;39m║[0m                                                              [38;5;39m║[0m
echo   [38;5;39m║[0m      [38;5;245m╔═══════════════════════════════════════════╗[0m       [38;5;39m║[0m
echo   [38;5;39m║[0m      [38;5;245m║[0m  [38;5;199mS H A D O W[0m [38;5;245m─[0m [38;5;51mQ U A N T[0m   [38;5;243mv7.0[0m       [38;5;245m║[0m       [38;5;39m║[0m
echo   [38;5;39m║[0m      [38;5;245m╚═══════════════════════════════════════════╝[0m       [38;5;39m║[0m
echo   [38;5;39m║[0m                                                              [38;5;39m║[0m
echo   [38;5;39m║[0m   [38;5;243m▸ Hybrid Jamba SSM Engine[0m                                [38;5;39m║[0m
echo   [38;5;39m║[0m   [38;5;243m▸ Institutional-Grade Analytics[0m                          [38;5;39m║[0m
echo   [38;5;39m║[0m   [38;5;243m▸ 6-Year BTC Forecasting Pipeline[0m                        [38;5;39m║[0m
echo   [38;5;39m║[0m                                                              [38;5;39m║[0m
echo   [38;5;39m╚══════════════════════════════════════════════════════════════╝[0m
echo.

:: ── System Checks ──
echo   [38;5;245m────────────────────── System Boot ──────────────────────[0m
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [38;5;196m  ✖ FATAL:[0m Python not found!
    echo.
    echo   [38;5;245m  Install Python 3.10+ from:[0m [38;5;51mhttps://python.org[0m
    echo   [38;5;245m  Make sure to check "Add Python to PATH" during install.[0m
    echo.
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo   [38;5;46m  ✔[0m [38;5;255m%PYVER%[0m [38;5;243mdetected[0m

:: Check CUDA
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" 2>nul && (
    for /f "tokens=*" %%g in ('python -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do (
        echo   [38;5;46m  ✔[0m [38;5;255m%%g[0m [38;5;243mCUDA ready[0m
    )
) || (
    echo   [38;5;214m  ⚠[0m [38;5;255mNo GPU detected[0m [38;5;243m— CPU mode[0m
)

:: Check Rich
python -c "import rich" 2>nul && (
    echo   [38;5;46m  ✔[0m [38;5;255mRich UI[0m [38;5;243mlibrary loaded[0m
) || (
    echo   [38;5;214m  ↻[0m [38;5;255mInstalling Rich UI...[0m
    python -m pip install rich -q 2>nul
    echo   [38;5;46m  ✔[0m [38;5;255mRich UI[0m [38;5;243minstalled[0m
)

echo.
echo   [38;5;245m─────────────────────────────────────────────────────────[0m
echo.
echo   [38;5;51m  ► Launching Nexus Engine...[0m
echo.

:: Small pause for dramatic effect
timeout /t 1 /nobreak >nul

:: Hand off to Python
title Nexus Shadow-Quant
cd /d "%~dp0"
python nexus_launcher.py
