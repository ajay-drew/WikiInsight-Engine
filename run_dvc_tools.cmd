@echo off
cd /d "%~dp0"

echo ================================================
echo DVC Pipeline Tools
echo ================================================

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Warning: No virtual environment found. Using system Python.
)

REM Check if DVC is installed
python -c "import dvc" 2>nul
if errorlevel 1 (
    echo Error: DVC is not installed.
    echo Please install it with: pip install dvc
    pause
    exit /b 1
)

echo.
echo DVC Pipeline Status:
echo ================================================
dvc dag
echo.
echo ================================================
echo Pipeline Status:
echo ================================================
dvc status
echo.
echo ================================================
echo Metrics:
echo ================================================
dvc metrics show
echo.
echo ================================================
echo.
echo Available Commands:
echo   dvc repro          - Run/reproduce the pipeline
echo   dvc dag            - Show pipeline DAG
echo   dvc status         - Show pipeline status
echo   dvc metrics show   - Show metrics
echo   dvc metrics diff   - Compare metrics between runs
echo   dvc push           - Push data to remote storage
echo   dvc pull           - Pull data from remote storage
echo.
echo ================================================

pause

