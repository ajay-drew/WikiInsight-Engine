@echo off
setlocal

echo ========================================
echo WikiInsight Engine - Data Pipeline Tools
echo ========================================
echo.

REM Change to project root (directory of this script)
cd /d "%~dp0"

REM Activate virtual environment if it exists (support both venv and .venv)
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

echo.

REM ---------------------
REM 1) Run DVC pipeline
REM ---------------------
echo [1/2] Running DVC Pipeline
echo ----------------------------------------

REM Check if DVC is installed
dvc --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] DVC is not installed or not in PATH.
    echo [INFO] Install with: pip install dvc[s3]
    pause
    exit /b 1
)

echo.
echo Running: dvc repro
echo This will execute:
echo   - fetch_data: Fetch Wikipedia articles
echo   - preprocess: Clean text and generate embeddings
echo   - cluster_topics: Cluster articles and build topic index
echo.
echo NOTE: This generates artifacts required for:
echo   - Topic clustering and lookup
echo   - Hybrid search engine (semantic + keyword search)
echo.

dvc repro
if errorlevel 1 (
    echo.
    echo [ERROR] DVC pipeline failed. Check the error messages above.
    echo [INFO] Continuing to start MLflow UI anyway...
) else (
    echo.
    echo [OK] DVC pipeline completed successfully!
    echo [INFO] Artifacts generated:
    echo   - data/processed/cleaned_articles.parquet
    echo   - data/features/embeddings.parquet
    echo   - models/clustering/*.pkl
    echo.
)

echo.

REM ------------------------
REM 2) Start MLflow tracking
REM ------------------------
echo [2/2] Starting MLflow UI
echo ----------------------------------------

REM Check if MLflow is installed
mlflow --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MLflow is not installed or not in PATH.
    echo [INFO] Install with: pip install mlflow
    pause
    exit /b 1
)

REM Use env var if set, otherwise fall back to local SQLite file
if not defined MLFLOW_TRACKING_URI set MLFLOW_TRACKING_URI=sqlite:///mlflow.db

echo.
echo Starting MLflow UI...
echo Tracking URI: %MLFLOW_TRACKING_URI%
echo.

start "MLflow UI" cmd /k "cd /d %~dp0 && if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && mlflow ui --backend-store-uri %MLFLOW_TRACKING_URI% --port 5000"

echo.
echo ========================================
echo MLflow UI Started Successfully!
echo ========================================
echo.
echo MLflow UI: http://localhost:5000
echo.
echo Tracked Metrics:
echo   - Clustering metrics (silhouette score, inertia, etc.)
echo   - Model parameters (n_clusters, method, etc.)
echo   - Experiment runs
echo.
echo Press any key to exit this window...
echo (MLflow UI will continue running in a separate window)
pause >nul

REM ------------------------
REM 3) Redis (optional)
REM ------------------------
REM Uncomment and adjust this block when Redis is installed and you want to run it:
REM echo.
REM echo [3/3] Starting Redis Server
REM echo ----------------------------------------
REM echo Starting Redis server ...
REM start "Redis" cmd /k "redis-server"

endlocal


