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
    echo [OK] Virtual environment activated (venv)
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated (.venv)
) else (
    echo [WARN] No virtual environment found. Using system Python.
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required data artifacts exist
if not exist "data\processed\cleaned_articles.parquet" (
    echo [WARN] Preprocessing artifacts not found. Search endpoint will be unavailable.
    echo [INFO] Run 'run_tools.cmd' or 'dvc repro' to generate required artifacts.
    echo.
)
if not exist "models\clustering\cluster_assignments.parquet" (
    echo [WARN] Clustering artifacts not found. Topic lookup endpoints will be unavailable.
    echo [INFO] Run 'run_tools.cmd' or 'dvc repro' to generate required artifacts.
    echo.
)

REM Set ports (use env vars if set)
if "%API_PORT%"=="" set "API_PORT=8000"
if "%FRONTEND_PORT%"=="" set "FRONTEND_PORT=5173"

echo.
echo Starting services...
echo.

REM Check if port is already in use
netstat -an | findstr ":%API_PORT%" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Port %API_PORT% is already in use!
    echo [INFO] Close the process using port %API_PORT% or set API_PORT to a different value.
    echo [INFO] Example: set API_PORT=8001 && run_app_production.cmd
    echo.
    pause
    exit /b 1
)

REM Start backend without reload (avoids Windows socket issues)
echo [1/2] Starting FastAPI backend (production mode)...
start "WikiInsight API" cmd /k "cd /d %~dp0 && if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && uvicorn src.api.main:app --host 127.0.0.1 --port %API_PORT%"

timeout /t 2 /nobreak >nul

REM Start frontend (React/Vite)
echo [2/2] Starting React frontend...
if not exist "frontend\node_modules" (
    echo [WARN] Frontend dependencies not installed. Installing...
    cd frontend
    call npm install
    cd ..
)

start "WikiInsight Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --host --port %FRONTEND_PORT%"

echo.
echo ========================================
echo Services Started Successfully!
echo ========================================
echo.
echo Backend API:    http://127.0.0.1:%API_PORT%
echo API Docs:       http://127.0.0.1:%API_PORT%/docs
echo Frontend:       http://localhost:%FRONTEND_PORT%
echo.
echo Features:
echo   - Hybrid Search (semantic + keyword with RRF)
echo   - Topic Clustering (AgglomerativeClustering)
echo   - Cluster Exploration & Lookup
echo   - Monitoring Dashboard
echo.
echo Press any key to close this window...
echo (Services will continue running in separate windows)
pause >nul
endlocal

