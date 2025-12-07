@echo off
setlocal

REM Run WikiInsight backend API (FastAPI) and React/Vite frontend.
REM Assumes this script is located in the project root.

REM Change to project root (directory of this script)
cd /d "%~dp0"

REM Activate virtual environment if it exists (support both venv and .venv)
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Default values if env vars are not set
if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8000
if not defined DASHBOARD_PORT set DASHBOARD_PORT=8501

echo Starting WikiInsight API on %API_HOST%:%API_PORT% ...
start "WikiInsight API" cmd /k "python -m uvicorn src.api.main:app --reload --host %API_HOST% --port %API_PORT%"

echo Starting WikiInsight Frontend (Vite dev server) ...
start "WikiInsight Frontend" cmd /k "cd frontend && npm run dev"

endlocal


