@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating Python environment...
  python -m venv .venv || goto :error
)

echo [2/3] Installing or checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements-web.txt || goto :error

netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 goto :port_busy

echo [3/3] Starting Bee Vision at http://127.0.0.1:8000
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
".venv\Scripts\python.exe" -m uvicorn web_app:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto :error
echo Bee Vision has stopped.
pause
exit /b 0

:port_busy
echo.
echo Port 8000 is already in use, so Bee Vision cannot start.
echo Close the old Bee Vision command window or restart the computer, then try again.
echo.
pause
exit /b 1

:error
echo.
echo Startup failed. Please check the error above.
pause
exit /b 1
