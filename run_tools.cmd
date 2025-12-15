@echo off
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (call "venv\Scripts\activate.bat") else if exist ".venv\Scripts\activate.bat" (call ".venv\Scripts\activate.bat")

echo Running DVC pipeline...
dvc repro

if not defined MLFLOW_TRACKING_URI set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
echo Starting MLflow UI...
start "MLflow" cmd /k "cd /d %~dp0 && if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && mlflow ui --backend-store-uri %MLFLOW_TRACKING_URI% --port 5000"

echo MLflow: http://localhost:5000
pause


