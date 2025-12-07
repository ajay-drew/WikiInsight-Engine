@echo off
cd /d "%~dp0"

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM Set ports (use env vars if set)
if "%API_PORT%"=="" set "API_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5173"

REM Start backend
start "API" cmd /k "cd /d %~dp0 && uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT% --reload"

REM Start frontend
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --host --port %FRONTEND_PORT%"

echo API: http://127.0.0.1:%API_PORT%
echo Frontend: http://localhost:%FRONTEND_PORT%
pause
