@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo [Bee Vision] Preparing the portable Python runtime...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_portable_runtime.ps1"
set "BEE_SETUP_EXIT=%ERRORLEVEL%"

if not "%BEE_SETUP_EXIT%"=="0" (
    echo.
    echo [Bee Vision] Runtime preparation failed.
    exit /b %BEE_SETUP_EXIT%
)

echo.
echo [Bee Vision] Runtime is ready. Run run_cli.bat --help to get started.
exit /b 0
