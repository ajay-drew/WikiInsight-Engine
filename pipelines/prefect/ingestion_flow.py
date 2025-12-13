"""
Prefect flow for running the Wikipedia ingestion step.

This wraps `src.ingestion.fetch_wikipedia_data.main` so that it can be scheduled
and monitored via Prefect. Includes retry logic, timeouts, error handling,
and MLflow logging.
"""

import logging
from time import perf_counter

from prefect import flow, task

from pipelines.prefect.config import get_retry_count, get_timeout_seconds
from pipelines.prefect.utils import (
    check_ingestion_artifacts,
    extract_ingestion_metrics,
)

from src.ingestion.fetch_wikipedia_data import main as fetch_main

logger = logging.getLogger(__name__)


@task(
    name="fetch_wikipedia_data",
    retries=get_retry_count("ingestion"),
    retry_delay_seconds=60,
    timeout_seconds=get_timeout_seconds("ingestion"),
    tags=["ingestion", "data-fetch"],
    log_prints=True,
)
def run_fetch_wikipedia_data() -> dict:
    """
    Fetch Wikipedia articles with retry logic and metrics extraction.
    
    Returns:
        Dict with metrics about the ingestion run
    """
    start_time = perf_counter()
    
    try:
        # Run the ingestion
        fetch_main()
        
        duration = perf_counter() - start_time
        
        # Extract metrics
        metrics = extract_ingestion_metrics()
        metrics["duration_seconds"] = duration
        metrics["success"] = True
        
        # Log to MLflow if available
        try:
            import mlflow
            import yaml
            
            # Load config for MLflow settings
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
            
            ml_cfg = config.get("mlops", {}).get("mlflow", {})
            tracking_uri = ml_cfg.get("tracking_uri")
            experiment_name = ml_cfg.get("experiment_name", "wikiinsight")
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name="ingestion_flow"):
                mlflow.log_param("stage", "ingestion")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"ingestion_{key}", value)
                mlflow.log_metric("ingestion_duration_seconds", duration)
        except Exception as exc:
            logger.warning("MLflow logging skipped or failed: %s", exc)
        
        logger.info("Ingestion completed successfully in %.2f seconds", duration)
        return metrics
        
    except Exception as exc:
        duration = perf_counter() - start_time
        logger.exception("Ingestion failed after %.2f seconds: %s", duration, exc)
        
        # Log failure to MLflow if available
        try:
            import mlflow
            import yaml
            
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
            
            ml_cfg = config.get("mlops", {}).get("mlflow", {})
            tracking_uri = ml_cfg.get("tracking_uri")
            experiment_name = ml_cfg.get("experiment_name", "wikiinsight")
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name="ingestion_flow"):
                mlflow.log_param("stage", "ingestion")
                mlflow.log_metric("ingestion_success", 0)
                mlflow.log_metric("ingestion_duration_seconds", duration)
                mlflow.log_param("ingestion_error", str(exc)[:200])
        except Exception:
            pass
        
        raise


@flow(name="ingestion_flow", log_prints=True)
def ingestion_flow() -> dict:
    """
    Prefect flow for Wikipedia article ingestion.
    
    Returns:
        Dict with metrics about the ingestion run
    """
    logger.info("Starting ingestion flow")
    
    # Check if artifacts already exist (optional: skip if exists)
    if check_ingestion_artifacts():
        logger.info("Ingestion artifacts already exist, skipping ingestion")
        metrics = extract_ingestion_metrics()
        metrics["skipped"] = True
        return metrics
    
    result = run_fetch_wikipedia_data()
    logger.info("Ingestion flow completed successfully")
    return result


if __name__ == "__main__":
    ingestion_flow()


