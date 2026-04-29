@echo off
echo ============================================
echo  ERP Assistant Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Creating folders...
mkdir uploads 2>nul
mkdir jobs 2>nul
mkdir frontend 2>nul
mkdir llm 2>nul
mkdir ui 2>nul

echo [2/3] Installing Python packages...
pip install -r requirements.txt

echo [3/3] Checking CUDA / GPU...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo.
echo ============================================
echo  Setup complete!
echo  Start with: start.bat
echo ============================================
pause
