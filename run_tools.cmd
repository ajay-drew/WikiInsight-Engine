@echo off
setlocal

REM Run DVC pipeline and MLflow UI for WikiInsight Engine.
REM Redis (if used) can be added later as another start command.
REM Assumes this script is located in the project root.

REM Change to project root (directory of this script)
cd /d "%~dp0"

REM Activate virtual environment if it exists (support both venv and .venv)
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM ---------------------
REM 1) Run DVC pipeline
REM ---------------------
REM Check if DVC is installed
dvc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: DVC is not installed or not in PATH. Please install DVC: pip install dvc[s3]
    exit /b 1
)

echo Running DVC pipeline (dvc repro) ...
dvc repro
if errorlevel 1 (
    echo ERROR: DVC pipeline failed. Check the error messages above.
    echo Continuing to start MLflow UI anyway...
) else (
    echo DVC pipeline completed successfully.
)

REM ------------------------
REM 2) Start MLflow tracking
REM ------------------------
REM Check if MLflow is installed
mlflow --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: MLflow is not installed or not in PATH. Please install MLflow: pip install mlflow
    exit /b 1
)

REM Use env var if set, otherwise fall back to local SQLite file
if not defined MLFLOW_TRACKING_URI set MLFLOW_TRACKING_URI=sqlite:///mlflow.db

echo Starting MLflow UI on http://localhost:5000 using %MLFLOW_TRACKING_URI% ...
start "MLflow UI" cmd /k "mlflow ui --backend-store-uri %MLFLOW_TRACKING_URI% --port 5000"

echo.
echo MLflow UI is starting in a separate window.
echo MLflow UI: http://localhost:5000
echo.
echo Press any key to exit this window (MLflow UI will continue running)...
pause >nul

REM ------------------------
REM 3) Redis (optional)
REM ------------------------
REM Uncomment and adjust this block when Redis is installed and you want to run it:
REM echo Starting Redis server ...
REM start "Redis" cmd /k "redis-server"

endlocal


