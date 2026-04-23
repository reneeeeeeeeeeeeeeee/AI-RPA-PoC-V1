@echo off
echo ============================================
echo  ERP-Assistent starten
echo ============================================
echo.

cd /d "%~dp0"

echo Starte Backend-Server...
echo Bitte warte waehrend die KI-Modelle geladen werden (kann 1-3 Minuten dauern)...
echo.
echo Danach: http://localhost:8000 im Browser oeffnen
echo.

python backend\main.py

pause
