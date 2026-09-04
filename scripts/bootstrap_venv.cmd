@echo off
rem CANN_OpHelper dependency bootstrap (CMD wrapper -> PowerShell script)
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%bootstrap_venv.ps1" %*
endlocal
