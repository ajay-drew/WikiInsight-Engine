@echo off
cd /d "%~dp0\.."

echo Starting MLflow UI Server...
echo.

REM Check if mlflow.db exists
if not exist "mlflow.db" (
    echo Warning: mlflow.db not found. It will be created automatically.
    echo.
)

REM Check for virtual environment
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

REM Start MLflow UI server
echo MLflow UI will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

%PYTHON% -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

pause

