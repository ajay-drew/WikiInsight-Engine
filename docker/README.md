# Docker Containerization Guide for WikiInsight Engine

This guide explains how to containerize the WikiInsight Engine application using Docker.

## Overview

The application consists of:
- **Backend**: FastAPI Python application (port 8000)
- **Frontend**: React + Vite application (built and served statically)

## Step-by-Step Containerization Process

### Step 1: Understanding the Multi-Stage Build

The Dockerfile uses a **multi-stage build** approach:

1. **Stage 1 (frontend-builder)**: Builds the React frontend
   - Uses Node.js to install dependencies
   - Compiles React app to static files
   - Output: `frontend/dist/` directory

2. **Stage 2 (main)**: Python backend with frontend assets
   - Installs Python dependencies
   - Downloads NLTK/spaCy models
   - Copies built frontend into container
   - Runs FastAPI server

### Step 2: Build the Docker Image

```bash
# Build the image
docker build -t wikiinsight-engine:latest .

# Or with a specific tag
docker build -t wikiinsight-engine:v1.0.0 .
```

**What happens during build:**
1. Frontend stage builds React app (~2-3 minutes)
2. Backend stage installs Python packages (~5-10 minutes)
3. Downloads NLTK data (~1-2 minutes)
4. Copies all application code
5. Sets up directory structure

### Step 3: Run the Container

#### Option A: Using Docker directly

```bash
# Run the container
docker run -d \
  --name wikiinsight-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  wikiinsight-engine:latest
```

#### Option B: Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Step 4: Verify the Container

```bash
# Check if container is running
docker ps

# Check container logs
docker logs wikiinsight-api

# Test health endpoint
curl http://localhost:8000/health

# Access API documentation
# Open browser: http://localhost:8000/docs
```

### Step 5: Access the Application

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend**: Served statically through FastAPI (if configured)

## Docker Compose Configuration

The `docker-compose.yml` file provides:

1. **Service Definition**: API service with proper configuration
2. **Volume Mounts**: Persist data, models, logs across restarts
3. **Environment Variables**: Load from `.env` file
4. **Health Checks**: Automatic container health monitoring
5. **Networking**: Isolated network for services

### Volume Mounts Explained

- `./data:/app/data` - Pipeline data (articles, embeddings, graphs)
- `./models:/app/models` - Trained models (clustering, etc.)
- `./mlruns:/app/mlruns` - MLflow experiment runs
- `./logs:/app/logs` - Application logs
- `./mlflow.db:/app/mlflow.db` - MLflow SQLite database

## Environment Variables

Create a `.env` file in the project root:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=wikiinsight

# Optional: DVC Remote
DVC_REMOTE_URL=
DVC_REMOTE_TYPE=local
```

## Production Considerations

### 1. Optimize Image Size

The current image is large (~2-3GB) due to ML dependencies. To optimize:

```dockerfile
# Use multi-stage build to reduce final image size
# Remove build dependencies after installation
# Use Alpine-based images (if compatible)
```

### 2. Security

- Run as non-root user (add to Dockerfile)
- Use secrets management for API keys
- Scan images for vulnerabilities: `docker scan wikiinsight-engine:latest`

### 3. Performance

- Adjust worker count in CMD: `--workers 4`
- Use reverse proxy (nginx) for production
- Enable gzip compression
- Use CDN for static frontend assets

### 4. Monitoring

- Add Prometheus metrics endpoint
- Configure log aggregation
- Set up health check alerts

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs wikiinsight-api

# Common issues:
# - Port 8000 already in use: Change port mapping
# - Missing data directories: Create them or use volumes
# - Permission errors: Check volume mount permissions
```

### Out of memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
# Or use: docker run --memory="4g" ...
```

### Slow startup

- First run downloads NLTK data (~100MB)
- Model downloads can take time
- Consider pre-building images with data

### Frontend not loading

- Check if frontend/dist exists in container
- Verify FastAPI static file serving configuration
- Check browser console for errors

## Development Workflow

### Build and test locally

```bash
# Build
docker build -t wikiinsight-engine:dev .

# Run with volume mounts for development
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  wikiinsight-engine:dev
```

### Update dependencies

```bash
# After updating requirements.txt
docker build --no-cache -t wikiinsight-engine:latest .
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build Docker image
  run: docker build -t wikiinsight-engine:${{ github.sha }} .

- name: Push to registry
  run: docker push wikiinsight-engine:${{ github.sha }}
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)

