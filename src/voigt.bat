@echo off
REM run_week5.bat
REM Runs a specific script inside week_5\ so that its "from src..." imports work.
REM
REM HOW TO USE:
REM   1. Copy this file anywhere you like (it does not need to live inside the repo).
REM   2. Edit the SCRIPT variable below to the script you want to run (no .py extension,
REM      using dots instead of slashes if it's nested).
REM   3. Edit PROJECT_ROOT below if your repo ever moves.
REM   4. Double-click the .bat, or run it from a terminal.

set PROJECT_ROOT=C:\Users\Ayank\OneDrive\Desktop\internship
set SCRIPT=RSQ_test_voigt

cd /d "%PROJECT_ROOT%"
uv run python -m week_5.%SCRIPT%

pause