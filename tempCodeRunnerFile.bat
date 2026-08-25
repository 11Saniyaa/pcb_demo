@echo off
cd /d "%~dp0"
echo.
echo   Open the app:      http://127.0.0.1:5173/
echo   Backend API:       http://127.0.0.1:8000/
echo   API docs:          http://127.0.0.1:8000/docs
echo.
python start_demo.py
if errorlevel 1 pause
