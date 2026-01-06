@echo off
cd /d "%~dp0"

if "%API_PORT%"=="" set "API_PORT=9000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5173"

echo Starting services...

REM Start backend
if exist "venv\Scripts\python.exe" (
    start "API" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT% --reload"
) else if exist ".venv\Scripts\python.exe" (
    start "API" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT% --reload"
) else (
    start "API" cmd /k "cd /d %~dp0 && python -m uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT% --reload"
)

timeout /t 2 /nobreak >nul

REM Start frontend
if not exist "frontend\node_modules" (cd frontend && npm install && cd ..)
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --host --port %FRONTEND_PORT%"

echo.
echo API: http://127.0.0.1:%API_PORT% | Frontend: http://localhost:%FRONTEND_PORT%
pause
