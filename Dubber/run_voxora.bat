@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Voxora Python environment is missing.
    echo Run this file again after setup is complete.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0voxora_studio_rebuilt.py"
if errorlevel 1 (
    echo.
    echo Voxora stopped with an error. Copy the message above when asking for help.
    pause
)
