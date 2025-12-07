@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo WikiInsight Engine - Production Mode
echo ========================================
echo.
echo Starting without auto-reload (better for Windows)
echo.

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Set ports (use env vars if set)
if "%API_PORT%"=="" set "API_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5173"

REM Check if port is already in use
netstat -an | findstr ":%API_PORT%" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port %API_PORT% is already in use!
    echo [INFO] Close the process using port %API_PORT% or set API_PORT to a different value.
    pause
    exit /b 1
)

REM Start backend without reload (avoids Windows socket issues)
echo [1/2] Starting FastAPI backend (production mode)...
start "WikiInsight API" cmd /k "cd /d %~dp0 && if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT%"

timeout /t 2 /nobreak >nul

REM Start frontend
echo [2/2] Starting React frontend...
start "WikiInsight Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --host --port %FRONTEND_PORT%"

echo.
echo Services started in production mode (no auto-reload).
echo Backend: http://127.0.0.1:%API_PORT%
echo Frontend: http://localhost:%FRONTEND_PORT%
echo.
pause
endlocal

