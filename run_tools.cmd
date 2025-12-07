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

REM ---------------------
REM 1) Run DVC pipeline
REM ---------------------
echo Running DVC pipeline (dvc repro) ...
dvc repro

REM ------------------------
REM 2) Start MLflow tracking
REM ------------------------
REM Use env var if set, otherwise fall back to local SQLite file
if not defined MLFLOW_TRACKING_URI set MLFLOW_TRACKING_URI=sqlite:///mlflow.db

echo Starting MLflow UI on http://localhost:5000 using %MLFLOW_TRACKING_URI% ...
start "MLflow UI" cmd /k "mlflow ui --backend-store-uri %MLFLOW_TRACKING_URI% --port 5000"

REM ------------------------
REM 3) Redis (optional)
REM ------------------------
REM Uncomment and adjust this block when Redis is installed and you want to run it:
REM echo Starting Redis server ...
REM start "Redis" cmd /k "redis-server"

endlocal


