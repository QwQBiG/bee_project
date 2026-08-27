@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Finding or preparing 64-bit Python 3.13...
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\prepare_windows_python.ps1" || goto :error
if not exist ".runtime\python-path.txt" goto :python_error
set /p BEE_PYTHON=<".runtime\python-path.txt"
if not exist "%BEE_PYTHON%" goto :python_error

netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 goto :port_busy

echo [2/4] Checking NVIDIA and installing dependencies...
"%BEE_PYTHON%" tools\bootstrap_runtime.py || goto :error

echo [3/4] Starting Bee Vision at http://127.0.0.1:8000
if not exist "logs" mkdir "logs"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$python = (Resolve-Path $env:BEE_PYTHON).Path; " ^
  "$server = Start-Process -FilePath $python -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'logs\server.log' -RedirectStandardError 'logs\server-error.log' -PassThru; " ^
  "for ($i = 0; $i -lt 60; $i++) { " ^
  "  if ($server.HasExited) { exit 2 }; " ^
  "  try { $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 1; if ($response.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8000'; exit 0 } } catch {}; " ^
  "  Start-Sleep -Milliseconds 500 " ^
  "}; " ^
  "Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue; exit 3"
if errorlevel 1 goto :server_error
echo [4/4] Bee Vision is ready.
exit /b 0

:port_busy
start "" "http://127.0.0.1:8000"
exit /b 0

:server_error
echo.
echo Bee Vision failed to start. Check logs\server-error.log for details.
goto :error

:python_error
echo.
echo Python 3.13 runtime preparation failed or returned an invalid path.
goto :error

:error
echo.
echo Startup failed. Please check the error above.
pause
exit /b 1
