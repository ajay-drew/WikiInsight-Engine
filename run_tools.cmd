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
echo.
echo This will execute the following pipeline stages:
echo   1. fetch_data: Fetch Wikipedia articles via mwclient API
echo   2. preprocess: Clean text, normalize with NLTK, generate embeddings
echo   3. cluster_topics: Cluster articles, extract keywords, build k-NN index
echo.
echo Generated artifacts:
echo   - data/raw/articles.json (raw Wikipedia articles)
echo   - data/processed/cleaned_articles.parquet (cleaned text)
echo   - data/features/embeddings.parquet (sentence-transformer embeddings)
echo   - models/clustering/*.pkl (clustering model, k-NN index)
echo   - models/clustering/*.parquet (cluster assignments, summaries)
echo.
echo These artifacts are required for:
echo   - Topic clustering and lookup endpoints
echo   - Hybrid search engine (semantic + keyword search with RRF)
echo   - Cluster exploration and monitoring
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
echo   - Ingestion metrics (articles fetched, filtered, duration)
echo   - Preprocessing metrics (vocabulary size, embedding dimensions)
echo   - Clustering metrics (silhouette score, Davies-Bouldin index)
echo   - Drift detection scores
echo   - Cluster stability metrics (ARI, NMI)
echo   - Model parameters (n_clusters, method, etc.)
echo   - Experiment runs and artifacts
echo.
echo Press any key to exit this window...
echo (MLflow UI will continue running in a separate window)
pause >nul

REM Note: Redis and PostgreSQL services were removed as they are not used
REM in the current codebase. The system uses SQLite for MLflow and
REM in-memory data structures for the API.

endlocal


