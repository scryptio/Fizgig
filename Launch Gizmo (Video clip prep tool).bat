@echo off
rem Gizmo — cut training clips Fizgig will accept. Tracked in the repo and arrives with an
rem ordinary update: it needs nothing the installer has not already put in the venv.
rem
rem This file must never be written by install_fizgig.py. Doing that to run_fizgig.bat once
rem left it modified in every clone, and the next git pull refused to run.
cd /d "%~dp0"

if not exist "venv\Scripts\pythonw.exe" (
    echo Fizgig's Python environment is missing.
    echo.
    echo Run install_fizgig.bat first, then try again.
    pause
    exit /b 1
)

start "" "venv\Scripts\pythonw.exe" "gizmo.pyw"
