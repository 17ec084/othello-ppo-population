@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0resume-winrate-experiment.ps1"
if errorlevel 1 (
    echo Resume failed. Please keep this window open and check the message above.
    pause
    exit /b 1
)
start "" "http://127.0.0.1:8766/"
