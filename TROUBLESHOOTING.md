# Troubleshooting Guide

## Common Issues

### 503 Error: "Topic index is not available. Run the clustering pipeline first."

**Symptoms:**
- API returns 503 error when accessing `/clusters/overview` or `/topics/lookup`
- Error message: "Topic index is not available. Run the clustering pipeline first."

**Cause:**
The API server was started before the clustering pipeline completed, so it didn't load the topic index artifacts.

**Solution:**
1. **Verify artifacts exist:**
   ```powershell
   # Check if all required files exist
   Test-Path "models/clustering/cluster_assignments.parquet"
   Test-Path "models/clustering/clusters_summary.parquet"
   Test-Path "models/clustering/kmeans_model.pkl"
   Test-Path "models/clustering/nn_index.pkl"
   Test-Path "data/features/embeddings.parquet"
   ```

2. **If artifacts are missing, run the pipeline:**
   ```powershell
   # Option 1: Run full DVC pipeline
   dvc repro
   
   # Option 2: Run individual steps
   python -m src.ingestion.fetch_wikipedia_data
   python -m src.preprocessing.process_data
   python -m src.modeling.cluster_topics
   ```

3. **Restart the API server:**
   - Stop the current API server (close the window or press Ctrl+C)
   - Restart using `run_app.cmd` or `run_app_production.cmd`
   - The API will automatically load the topic index on startup

**Note:** The API loads artifacts only. startup. If you generate new artifacts while the API is running, you must restart it to load them.

---

### Search Endpoint Returns 503

**Symptoms:**
- `/api/search` endpoint returns 503
- Error: "Search engine is not available"

**Cause:**
Missing search engine artifacts (embeddings or cleaned articles).

**Solution:**
1. Run the preprocessing pipeline:
   ```powershell
   dvc repro
   # OR
   python -m src.preprocessing.process_data
   ```

2. Restart the API server

---

### Port Already in Use

**Symptoms:**
- Error when starting API: "Port 8000 is already in use"

**Solution:**
1. Find and close the process using the port:
   ```powershell
   netstat -ano | findstr :8000
   # Note the PID, then:
   taskkill /PID <PID> /F
   ```

2. Or use a different port:
   ```powershell
   set API_PORT=8001
   run_app.cmd
   ```

---

### Frontend Can't Connect to API

**Symptoms:**
- Frontend shows "Failed to fetch" errors
- Network errors in browser console

**Solution:**
1. Verify API is running: Open http://127.0.0.1:8000/docs
2. Check CORS settings in `src/api/main.py`
3. Verify frontend proxy configuration in `frontend/vite.config.ts`
4. Check that API_PORT and FRONTEND_PORT environment variables match

---

### DVC Pipeline Fails

**Symptoms:**
- `dvc repro` fails with errors
- Missing dependencies or parameters

**Solution:**
1. Check DVC status:
   ```powershell
   dvc status
   ```

2. Verify configuration files exist:
   - `config.yaml`
   - `params.yaml`
   - `dvc.yaml`

3. Check for missing dependencies:
   ```powershell
   pip install -e .
   ```

4. Review error messages and fix the specific issue

---

### MLflow UI Not Starting

**Symptoms:**
- `run_tools.cmd` fails to start MLflow UI
- Port 5000 already in use

**Solution:**
1. Check if MLflow is installed:
   ```powershell
   mlflow --version
   ```

2. Use a different port:
   ```powershell
   set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
   ```

3. Check if port 5000 is in use and close the process

