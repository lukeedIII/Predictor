@echo off
echo ============================================
echo   Nexus Training Kit — Starting Server
echo ============================================
echo.
echo Open http://localhost:5555 in your browser
echo Press Ctrl+C to stop (checkpoint will be saved)
echo.

python train_server.py --port 5555

pause
