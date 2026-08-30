@echo off
setlocal
chcp 65001 >nul
set "BEE_PROJECT_ROOT=%~dp0"
set "BEE_PYTHON=%BEE_PROJECT_ROOT%.runtime\python313\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%BEE_PYTHON%" (
    echo [Bee Vision] Portable Python is not ready. Run setup_runtime.bat first. 1>&2
    exit /b 2
)

"%BEE_PYTHON%" "%BEE_PROJECT_ROOT%main.py" %*
exit /b %ERRORLEVEL%
