@echo off
chcp 65001 >nul 2>&1
title Nexus Shadow-Quant
mode con cols=72 lines=40
color 0F

:: ══════════════════════════════════════════════════════════════════
::   NEXUS SHADOW-QUANT — Launcher
::   Validates Python, then hands off to the Rich-powered launcher.
:: ══════════════════════════════════════════════════════════════════

cls
echo.
echo   ============================================================
echo.
echo     N E X U S   S H A D O W - Q U A N T
echo.
echo     Hybrid Jamba SSM Engine  ^|  v7.0
echo.
echo   ============================================================
echo.

:: ── Verify Python is installed ──
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [ERROR] Python not found!
    echo.
    echo   Install Python 3.10+ from: https://python.org
    echo   Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo     %PYVER% detected

:: ── Ensure Rich is available (needed for the premium UI) ──
python -c "import rich" 2>nul || (
    echo     Installing Rich UI library...
    python -m pip install rich -q 2>nul
)

echo     Launching...
echo.

:: ── Hand off to the Python launcher (Rich handles all the UI) ──
cd /d "%~dp0"
python nexus_launcher.py
