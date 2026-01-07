# WikiInsight Engine - Docker Setup

This directory contains Docker configuration files for containerizing the WikiInsight Engine.

## Quick Start

1. **Build and start all services:**
   ```bash
   cd infrastructure
   docker-compose up -d
   ```

2. **Access the application:**
   - Frontend & API: http://localhost:9000
   - MLflow UI (if enabled): http://localhost:5000
   - PostgreSQL: localhost:5432

3. **View logs:**
   ```bash
   docker-compose logs -f api
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

## Services

### API Service (`api`)
- **Port**: 9000
- **Image**: Built from `infrastructure/Dockerfile`
- **Features**:
  - FastAPI backend with hybrid search
  - Serves built React frontend as static files
  - Connects to PostgreSQL database
  - Health check endpoint at `/health`

### PostgreSQL Service (`postgres`)
- **Port**: 5432
- **Image**: `pgvector/pgvector:pg16`
- **Features**:
  - PostgreSQL 16 with pgvector extension
  - Automatic extension initialization
  - Data persisted in Docker volume
  - Default credentials: `postgres/postgres`

### MLflow Service (`mlflow`) - Optional
- **Port**: 5000
- **Image**: `ghcr.io/mlflow/mlflow:latest`
- **Features**:
  - MLflow tracking server
  - Only starts with `--profile mlflow` flag

## Environment Variables

Create a `.env` file in the project root (copy from `env.example`):

```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/wikiinsight
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=9000
```

The docker-compose.yml automatically sets the database URL to use the `postgres` service name for internal networking.

## Volumes

The following directories are mounted as volumes for data persistence:

- `../data` → `/app/data` - Pipeline data and artifacts
- `../models` → `/app/models` - Trained models
- `../mlruns` → `/app/mlruns` - MLflow runs
- `../logs` → `/app/logs` - Application logs
- `../mlflow.db` → `/app/mlflow.db` - MLflow database
- `../pipeline_logs.db` → `/app/pipeline_logs.db` - Pipeline logs
- `../config.yaml` → `/app/config.yaml` - Configuration file

PostgreSQL data is stored in a Docker volume `postgres_data` for persistence.

## Building

To rebuild the Docker image:

```bash
docker-compose build api
```

Or rebuild all services:

```bash
docker-compose build
```

## Development

For development, you can mount the source code as a volume to enable hot-reloading:

```yaml
# Add to api service in docker-compose.yml
volumes:
  - ../src:/app/src
  - ../frontend:/app/frontend
```

Note: This requires rebuilding the frontend manually or using a separate dev server.

## Troubleshooting

### Database connection issues
- Ensure PostgreSQL service is healthy: `docker-compose ps`
- Check database logs: `docker-compose logs postgres`
- Verify pgvector extension: Connect to database and run `SELECT * FROM pg_extension WHERE extname = 'vector';`

### Port conflicts
- If port 9000 is in use, change it in `docker-compose.yml`:
  ```yaml
  ports:
    - "9001:9000"  # Host:Container
  ```

### Frontend not loading
- Ensure frontend is built: The Dockerfile builds it automatically
- Check API logs for frontend serving errors
- Verify `frontend/dist` exists in the container

### Out of memory
- Reduce workers in Dockerfile CMD: Change `--workers 4` to `--workers 2`
- Increase Docker memory limit in Docker Desktop settings

## Production Considerations

For production deployment:

1. **Use environment-specific configs**: Don't mount `config.yaml`, use environment variables
2. **Secure database**: Change default PostgreSQL credentials
3. **Use secrets management**: Don't commit `.env` files
4. **Enable HTTPS**: Use a reverse proxy (nginx/traefik) in front of the API
5. **Resource limits**: Add resource constraints to docker-compose.yml
6. **Backup strategy**: Regularly backup PostgreSQL volume and data directories
7. **Monitoring**: Add monitoring services (Prometheus, Grafana)

Example production docker-compose override:

```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: always
  postgres:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
    restart: always
```

Run with: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

