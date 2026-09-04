@echo off
REM Double-clickable wrapper around install.ps1 (bypasses the default
REM PowerShell script-execution restriction for this one run only).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
