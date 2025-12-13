"""
Prefect flow for running the topic clustering step.

This wraps `src.modeling.cluster_topics.main` so that it can be scheduled
and monitored via Prefect. Includes retry logic, validation, timeouts,
and task dependencies.
"""

import logging
from time import perf_counter

from prefect import flow, task

from pipelines.prefect.config import get_retry_count, get_timeout_seconds
from pipelines.prefect.utils import (
    check_clustering_artifacts,
    extract_clustering_metrics,
    validate_embeddings,
)

from src.modeling.cluster_topics import main as cluster_main

logger = logging.getLogger(__name__)


@task(
    name="validate_embeddings",
    tags=["validation", "clustering"],
)
def validate_embeddings_task() -> dict:
    """
    Validate that embedding artifacts exist and are valid.
    
    Returns:
        Dict with validation results
    """
    logger.info("Validating embedding artifacts")
    validation = validate_embeddings()
    
    if not validation.get("valid", False):
        error_msg = validation.get("error", "Validation failed")
        logger.error("Embedding validation failed: %s", error_msg)
        raise ValueError(f"Embedding validation failed: {error_msg}")
    
    logger.info(
        "Embedding validation passed: %d articles, dim=%d",
        validation.get("article_count", 0),
        validation.get("embedding_dim", 0),
    )
    return validation


@task(
    name="detect_drift",
    tags=["monitoring", "drift"],
)
def run_drift_detection() -> dict:
    """
    Run drift detection after clustering.
    
    Returns:
        Dict with drift metrics
    """
    try:
        from src.monitoring.drift_metrics import run_drift_detection
        import yaml
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}
        
        monitoring_cfg = config.get("monitoring", {}).get("drift", {})
        enabled = monitoring_cfg.get("enabled", True)
        
        if not enabled:
            logger.info("Drift detection disabled in config")
            return {"skipped": True}
        
        threshold = monitoring_cfg.get("threshold", 0.3)
        drift_metrics = run_drift_detection(threshold=threshold, save_baseline=True, log_to_mlflow=True)
        
        logger.info("Drift detection completed")
        return drift_metrics
    except Exception as exc:
        logger.warning("Drift detection failed: %s", exc)
        return {"error": str(exc)}


@task(
    name="cluster_topics",
    retries=get_retry_count("clustering"),
    retry_delay_seconds=60,
    timeout_seconds=get_timeout_seconds("clustering"),
    tags=["clustering", "modeling"],
    log_prints=True,
)
def run_cluster_topics() -> dict:
    """
    Run topic clustering with metrics extraction.
    
    Returns:
        Dict with metrics about the clustering run
    """
    start_time = perf_counter()
    
    try:
        # Run clustering
        cluster_main()
        
        duration = perf_counter() - start_time
        
        # Extract metrics
        metrics = extract_clustering_metrics()
        metrics["duration_seconds"] = duration
        metrics["success"] = True
        
        logger.info("Clustering completed successfully in %.2f seconds", duration)
        return metrics
        
    except Exception as exc:
        duration = perf_counter() - start_time
        logger.exception("Clustering failed after %.2f seconds: %s", duration, exc)
        
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
            
            with mlflow.start_run(run_name="clustering_flow"):
                mlflow.log_param("stage", "clustering")
                mlflow.log_metric("clustering_success", 0)
                mlflow.log_metric("clustering_duration_seconds", duration)
                mlflow.log_param("clustering_error", str(exc)[:200])
        except Exception:
            pass
        
        raise


@flow(name="clustering_flow", log_prints=True)
def clustering_flow(skip_validation: bool = False) -> dict:
    """
    Prefect flow for topic clustering.
    
    Args:
        skip_validation: If True, skip validation step (for testing)
    
    Returns:
        Dict with metrics about the clustering run
    """
    logger.info("Starting clustering flow")
    
    # Check if artifacts already exist (optional: skip if exists)
    if check_clustering_artifacts():
        logger.info("Clustering artifacts already exist, skipping clustering")
        metrics = extract_clustering_metrics()
        metrics["skipped"] = True
        return metrics
    
    # Validate embeddings first
    if not skip_validation:
        validation_result = validate_embeddings_task()
        logger.info("Validation passed: %s", validation_result)
    
    # Run clustering
    result = run_cluster_topics()
    
    # Run drift detection if enabled
    try:
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}
        monitoring_cfg = config.get("monitoring", {}).get("drift", {})
        if monitoring_cfg.get("check_after_clustering", True):
            drift_result = run_drift_detection()
            result["drift_detection"] = drift_result
    except Exception as exc:
        logger.warning("Failed to run drift detection: %s", exc)
    
    logger.info("Clustering flow completed successfully")
    return result


if __name__ == "__main__":
    clustering_flow()


