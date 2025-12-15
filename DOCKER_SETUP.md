# Docker Containerization Guide - Step by Step

This document explains how to containerize the WikiInsight Engine application using Docker.

## Overview

The WikiInsight Engine is containerized using a **multi-stage Docker build** that:
1. Builds the React frontend (Stage 1)
2. Sets up the Python backend with all dependencies (Stage 2)
3. Combines both into a single production-ready container

## Step-by-Step Process

### Step 1: Prepare the Project Structure

Ensure your project has:
- `requirements.txt` - Python dependencies
- `frontend/package.json` - Frontend dependencies
- `Dockerfile` - Docker build instructions
- `.dockerignore` - Files to exclude from build
- `docker-compose.yml` - Orchestration configuration

### Step 2: Understanding the Dockerfile

The Dockerfile uses **multi-stage builds**:

#### Stage 1: Frontend Builder
```dockerfile
FROM node:18-alpine AS frontend-builder
```
- Uses Node.js to build React application
- Installs npm dependencies
- Compiles TypeScript/React to static JavaScript
- Output: `frontend/dist/` directory with production-ready files

#### Stage 2: Python Backend
```dockerfile
FROM python:3.10-slim
```
- Installs system dependencies (gcc, g++, etc. for compiling Python packages)
- Installs Python packages from `requirements.txt`
- Downloads NLTK data (required for text processing)
- Downloads spaCy model (if used)
- Copies application code
- Copies built frontend from Stage 1
- Sets up directory structure
- Exposes port 8000
- Runs FastAPI server with uvicorn

### Step 3: Build the Docker Image

```bash
# Navigate to project root
cd /path/to/WikiInsight-Engine

# Build the image
docker build -t wikiinsight-engine:latest .
```

**What happens during build:**

1. **Frontend Build (2-3 minutes)**
   - Downloads Node.js base image
   - Installs npm packages
   - Compiles React app to static files
   - Creates optimized production build

2. **Backend Setup (5-10 minutes)**
   - Downloads Python base image
   - Installs system packages (gcc, g++, etc.)
   - Installs Python packages:
     - Core: pandas, numpy, scipy
     - ML: scikit-learn, sentence-transformers
     - API: fastapi, uvicorn
     - NLP: nltk, spacy
     - And 30+ other dependencies
   - Downloads NLTK data (~100MB)
   - Downloads spaCy model (~50MB)

3. **Final Assembly**
   - Copies all source code
   - Copies built frontend
   - Creates data directories
   - Sets permissions

**Total build time:** ~10-15 minutes (first time), ~5-8 minutes (cached)

### Step 4: Run the Container

#### Option A: Docker Run (Simple)

```bash
docker run -d \
  --name wikiinsight-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  wikiinsight-engine:latest
```

**Explanation:**
- `-d`: Run in detached mode (background)
- `--name`: Container name
- `-p 8000:8000`: Map host port 8000 to container port 8000
- `-v`: Volume mounts (persist data outside container)

#### Option B: Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Step 5: Verify the Container

```bash
# Check container status
docker ps

# Should show:
# CONTAINER ID   IMAGE                      STATUS          PORTS
# abc123def456   wikiinsight-engine:latest  Up 2 minutes    0.0.0.0:8000->8000/tcp

# Check logs
docker logs wikiinsight-api

# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"degraded"}  # (degraded because no data loaded yet)
```

### Step 6: Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend**: http://localhost:8000/ (served by FastAPI)
- **API Endpoints**: http://localhost:8000/api/*

## Understanding Volume Mounts

Volume mounts persist data outside the container:

```yaml
volumes:
  - ./data:/app/data          # Pipeline data (articles, embeddings)
  - ./models:/app/models      # Trained models
  - ./mlruns:/app/mlruns      # MLflow experiments
  - ./logs:/app/logs          # Application logs
```

**Why use volumes?**
- Data persists when container is removed
- Can backup data easily
- Can share data between containers
- Can access data from host machine

## Environment Variables

Create `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=wikiinsight

# Optional
DVC_REMOTE_URL=
DVC_REMOTE_TYPE=local
```

Load with: `docker-compose up` (automatically loads `.env`)

## Production Deployment Steps

### 1. Build Production Image

```bash
# Build with specific tag
docker build -t wikiinsight-engine:v1.0.0 .

# Tag for registry
docker tag wikiinsight-engine:v1.0.0 registry.example.com/wikiinsight-engine:v1.0.0

# Push to registry
docker push registry.example.com/wikiinsight-engine:v1.0.0
```

### 2. Deploy to Server

```bash
# Pull image
docker pull registry.example.com/wikiinsight-engine:v1.0.0

# Run container
docker run -d \
  --name wikiinsight-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /data/wikiinsight:/app/data \
  -v /data/wikiinsight/models:/app/models \
  -v /data/wikiinsight/logs:/app/logs \
  --env-file .env.production \
  registry.example.com/wikiinsight-engine:v1.0.0
```

### 3. Use Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name wikiinsight.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs wikiinsight-api

# Common issues:
# 1. Port already in use
#    Solution: Change port: -p 8001:8000

# 2. Permission denied
#    Solution: Fix volume permissions: chmod -R 755 ./data

# 3. Out of memory
#    Solution: Increase Docker memory limit
```

### Build fails

```bash
# Clear Docker cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t wikiinsight-engine:latest .
```

### Frontend not loading

```bash
# Check if frontend was built
docker exec wikiinsight-api ls -la /app/frontend/dist

# Rebuild frontend in container
docker exec -it wikiinsight-api sh
cd /app/frontend
npm install
npm run build
```

### Slow performance

- Increase container resources (CPU/Memory)
- Use production-optimized base images
- Enable Docker BuildKit: `DOCKER_BUILDKIT=1 docker build ...`

## Development Workflow

### Hot Reload Development

```bash
# Mount source code for live updates
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  wikiinsight-engine:latest \
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Debug Container

```bash
# Enter container shell
docker exec -it wikiinsight-api bash

# Check Python packages
pip list

# Check environment
env | grep -i python

# Test imports
python -c "import fastapi; print(fastapi.__version__)"
```

## Best Practices

1. **Use .dockerignore**: Reduces build context size
2. **Multi-stage builds**: Smaller final image
3. **Layer caching**: Order Dockerfile commands by change frequency
4. **Health checks**: Monitor container health
5. **Resource limits**: Set CPU/memory limits
6. **Security**: Run as non-root user
7. **Logging**: Use structured logging
8. **Backups**: Regularly backup volume data

## Summary

The containerization process:
1. ✅ Builds frontend React app
2. ✅ Sets up Python environment with all dependencies
3. ✅ Downloads NLP models (NLTK, spaCy)
4. ✅ Combines everything into single container
5. ✅ Exposes API on port 8000
6. ✅ Serves frontend through FastAPI
7. ✅ Persists data through volume mounts

**Result**: A single, portable container that runs the entire WikiInsight Engine application!

