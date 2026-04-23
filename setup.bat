@echo off
echo ============================================
echo  ERP-Assistent Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Erstelle Ordner...
mkdir uploads 2>nul
mkdir jobs 2>nul
mkdir frontend 2>nul
mkdir llm 2>nul
mkdir ui 2>nul

echo [2/3] Installiere Python-Pakete...
pip install -r requirements.txt

echo [3/3] Pruefe CUDA / GPU...
python -c "import torch; print('CUDA verfuegbar:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine')"

echo.
echo ============================================
echo  Setup abgeschlossen!
echo  Starte mit: start.bat
echo ============================================
pause
