@echo off
title Nexus Shadow-Quant
color 0B
echo.
echo  ============================================
echo   NEXUS SHADOW-QUANT — STARTING...
echo  ============================================
echo.

:: Check if node_modules exist
if not exist "%~dp0desktop\node_modules" (
    echo  First run detected! Run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

:: Pre-Flight Cleanup: Annihilate Zombie Processes
echo  [0/2] Cleaning up any old ghost processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8420') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173') do taskkill /f /pid %%a >nul 2>&1
timeout /t 2 /nobreak >nul


:: Start Python backend
echo  [1/2] Starting Python backend...
start "Nexus Backend" /min cmd /c "cd /d "%~dp0desktop\python_backend" && python api_server.py"

:: Wait for backend to initialize
echo  Waiting for backend to boot (8 seconds)...
timeout /t 8 /nobreak >nul

:: Start frontend dev server
echo  [2/2] Starting frontend...
start "Nexus Frontend" /min cmd /c "cd /d "%~dp0desktop" && npx vite --port 5173 --open"

:: Wait for Vite to start
timeout /t 4 /nobreak >nul

echo.
echo  ============================================
echo   NEXUS SHADOW-QUANT IS RUNNING!
echo  ============================================
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8420
echo.
echo   Close this window to keep running.
echo   Press any key to STOP both servers.
echo.
pause

:: Kill both servers
echo.
echo  Shutting down...
taskkill /FI "WINDOWTITLE eq Nexus Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Nexus Frontend*" /F >nul 2>&1
echo  Done! Goodbye.
timeout /t 2 >nul
