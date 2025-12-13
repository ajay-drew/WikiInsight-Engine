"""
Prefect flow for running the preprocessing/embeddings step.

This wraps `src.preprocessing.process_data.main` so that it can be scheduled
and monitored via Prefect. Includes retry logic, validation, timeouts,
and task dependencies.
"""

import logging
from time import perf_counter

from prefect import flow, task

from pipelines.prefect.config import get_retry_count, get_timeout_seconds
from pipelines.prefect.utils import (
    check_preprocessing_artifacts,
    validate_raw_articles,
)

from src.preprocessing.process_data import main as preprocess_main

logger = logging.getLogger(__name__)


@task(
    name="validate_ingestion_output",
    tags=["validation", "embeddings"],
)
def validate_ingestion() -> dict:
    """
    Validate that ingestion artifacts exist and are valid.
    
    Returns:
        Dict with validation results
    """
    logger.info("Validating ingestion artifacts")
    validation = validate_raw_articles()
    
    if not validation.get("valid", False):
        error_msg = validation.get("error", "Validation failed")
        logger.error("Ingestion validation failed: %s", error_msg)
        raise ValueError(f"Ingestion validation failed: {error_msg}")
    
    logger.info(
        "Ingestion validation passed: %d articles",
        validation.get("article_count", 0),
    )
    return validation


@task(
    name="generate_embeddings",
    retries=get_retry_count("embeddings"),
    retry_delay_seconds=30,
    timeout_seconds=get_timeout_seconds("embeddings"),
    tags=["embeddings", "preprocessing"],
    log_prints=True,
)
def run_preprocess() -> dict:
    """
    Run preprocessing and embedding generation with metrics extraction.
    
    Returns:
        Dict with metrics about the preprocessing run
    """
    start_time = perf_counter()
    
    try:
        # Run preprocessing
        preprocess_main()
        
        duration = perf_counter() - start_time
        
        # Extract basic metrics
        metrics = {"duration_seconds": duration, "success": True}
        
        # Try to get more detailed metrics from MLflow or files
        if check_preprocessing_artifacts():
            try:
                import pandas as pd
                emb_df = pd.read_parquet("data/features/embeddings.parquet")
                cleaned_df = pd.read_parquet("data/processed/cleaned_articles.parquet")
                metrics["articles_processed"] = len(cleaned_df)
                metrics["embeddings_generated"] = len(emb_df)
                if "embedding" in emb_df.columns and len(emb_df) > 0:
                    sample_emb = emb_df["embedding"].iloc[0]
                    metrics["embedding_dimension"] = len(sample_emb) if isinstance(sample_emb, list) else 0
            except Exception as exc:
                logger.warning("Failed to extract detailed metrics: %s", exc)
        
        logger.info("Preprocessing completed successfully in %.2f seconds", duration)
        return metrics
        
    except Exception as exc:
        duration = perf_counter() - start_time
        logger.exception("Preprocessing failed after %.2f seconds: %s", duration, exc)
        
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
            
            with mlflow.start_run(run_name="embeddings_flow"):
                mlflow.log_param("stage", "embeddings")
                mlflow.log_metric("embeddings_success", 0)
                mlflow.log_metric("embeddings_duration_seconds", duration)
                mlflow.log_param("embeddings_error", str(exc)[:200])
        except Exception:
            pass
        
        raise


@flow(name="embeddings_flow", log_prints=True)
def embeddings_flow(skip_validation: bool = False) -> dict:
    """
    Prefect flow for preprocessing and embedding generation.
    
    Args:
        skip_validation: If True, skip validation step (for testing)
    
    Returns:
        Dict with metrics about the preprocessing run
    """
    logger.info("Starting embeddings flow")
    
    # Check if artifacts already exist (optional: skip if exists)
    if check_preprocessing_artifacts():
        logger.info("Preprocessing artifacts already exist, skipping preprocessing")
        return {"skipped": True, "success": True}
    
    # Validate ingestion output first
    if not skip_validation:
        validation_result = validate_ingestion()
        logger.info("Validation passed: %s", validation_result)
    
    # Run preprocessing
    result = run_preprocess()
    logger.info("Embeddings flow completed successfully")
    return result


if __name__ == "__main__":
    embeddings_flow()


