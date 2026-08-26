@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating Python environment...
  python -m venv .venv || goto :error
)

netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 goto :port_busy

echo [2/3] Installing or checking dependencies...
".venv\Scripts\python.exe" tools\bootstrap_runtime.py || goto :error

echo [3/3] Starting Bee Vision at http://127.0.0.1:8000
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto :error
echo Bee Vision has stopped.
pause
exit /b 0

:port_busy
echo.
echo Bee Vision is already running on port 8000.
echo Open http://127.0.0.1:8000 or close it from the web page before starting again.
echo.
pause
exit /b 1

:error
echo.
echo Startup failed. Please check the error above.
pause
exit /b 1
