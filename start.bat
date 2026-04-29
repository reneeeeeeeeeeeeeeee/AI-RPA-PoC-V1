@echo off
echo ============================================
echo  ERP Assistant - Starting
echo ============================================
echo.

cd /d "%~dp0"

echo Starting backend server...
echo Please wait while the AI models are loading (may take 1-3 minutes)...
echo.
echo Then open: http://localhost:8000 in your browser
echo.

python backend\main.py

pause
