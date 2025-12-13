"""
Prefect flow that chains ingestion → embeddings → clustering.

This provides a single entrypoint to run the full Phase 1 pipeline:
    ingestion_flow → embeddings_flow → clustering_flow

Includes conditional execution, error handling, and flow result aggregation.
"""

import logging
from typing import Dict, Optional

from prefect import flow

from pipelines.prefect.config import get_notification_config
from pipelines.prefect.utils import (
    check_clustering_artifacts,
    check_ingestion_artifacts,
    check_preprocessing_artifacts,
)

from .clustering_flow import clustering_flow
from .embeddings_flow import embeddings_flow
from .ingestion_flow import ingestion_flow

logger = logging.getLogger(__name__)


def send_notification(message: str, is_error: bool = False) -> None:
    """Send notification via configured channel."""
    try:
        notif_cfg = get_notification_config()
        if not notif_cfg.get("enabled", False):
            return
        
        webhook_url = notif_cfg.get("webhook_url")
        if webhook_url:
            try:
                import requests
                requests.post(
                    webhook_url,
                    json={"message": message, "error": is_error},
                    timeout=5,
                )
            except Exception as exc:
                logger.warning("Failed to send webhook notification: %s", exc)
    except Exception as exc:
        logger.warning("Notification failed: %s", exc)


@flow(name="full_topic_pipeline", log_prints=True)
def full_topic_pipeline(
    skip_existing: bool = True,
    skip_ingestion: bool = False,
    skip_embeddings: bool = False,
    skip_clustering: bool = False,
) -> Dict:
    """
    Run the full topic clustering pipeline with conditional execution.
    
    Args:
        skip_existing: If True, skip stages where artifacts already exist
        skip_ingestion: If True, skip ingestion stage
        skip_embeddings: If True, skip embeddings stage
        skip_clustering: If True, skip clustering stage
    
    Returns:
        Dict with results from each stage and overall status
    """
    logger.info("Starting full topic pipeline")
    
    results = {
        "ingestion": None,
        "embeddings": None,
        "clustering": None,
        "success": False,
        "errors": [],
    }
    
    try:
        # Stage 1: Ingestion
        if not skip_ingestion:
            if skip_existing and check_ingestion_artifacts():
                logger.info("Skipping ingestion (artifacts exist)")
                results["ingestion"] = {"skipped": True}
            else:
                logger.info("Running ingestion flow")
                try:
                    results["ingestion"] = ingestion_flow()
                except Exception as exc:
                    error_msg = f"Ingestion failed: {str(exc)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    send_notification(f"Pipeline failed at ingestion: {error_msg}", is_error=True)
                    raise
        else:
            results["ingestion"] = {"skipped": True}
        
        # Stage 2: Embeddings (depends on ingestion)
        if not skip_embeddings:
            if skip_existing and check_preprocessing_artifacts():
                logger.info("Skipping embeddings (artifacts exist)")
                results["embeddings"] = {"skipped": True}
            else:
                logger.info("Running embeddings flow")
                try:
                    results["embeddings"] = embeddings_flow()
                except Exception as exc:
                    error_msg = f"Embeddings failed: {str(exc)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    send_notification(f"Pipeline failed at embeddings: {error_msg}", is_error=True)
                    raise
        else:
            results["embeddings"] = {"skipped": True}
        
        # Stage 3: Clustering (depends on embeddings)
        if not skip_clustering:
            if skip_existing and check_clustering_artifacts():
                logger.info("Skipping clustering (artifacts exist)")
                results["clustering"] = {"skipped": True}
            else:
                logger.info("Running clustering flow")
                try:
                    results["clustering"] = clustering_flow()
                except Exception as exc:
                    error_msg = f"Clustering failed: {str(exc)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    send_notification(f"Pipeline failed at clustering: {error_msg}", is_error=True)
                    raise
        else:
            results["clustering"] = {"skipped": True}
        
        # All stages completed successfully
        results["success"] = True
        logger.info("Full pipeline completed successfully")
        send_notification("Pipeline completed successfully", is_error=False)
        
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        results["success"] = False
        if not results["errors"]:
            results["errors"].append(str(exc))
        send_notification(f"Pipeline failed: {str(exc)}", is_error=True)
        raise
    
    return results


if __name__ == "__main__":
    full_topic_pipeline()


