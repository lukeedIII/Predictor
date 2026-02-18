@echo off
title Nexus Shadow-Quant — Launcher
color 0B

:: ============================================
::   NEXUS SHADOW-QUANT — SMART LAUNCHER
:: ============================================
::
::   Double-click this file to start.
::   It will check Python, install 'rich' if needed,
::   and launch the interactive menu.
::
:: ============================================

:: Check Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ERROR: Python not found!
    echo.
    echo   Install Python 3.10+ from: https://python.org
    echo   Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Launch the interactive Python menu
cd /d "%~dp0"
python nexus_launcher.py
