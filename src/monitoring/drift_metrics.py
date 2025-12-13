"""
Drift metrics calculation and MLflow integration.

Calculates drift scores between baseline and current data,
stores metrics in MLflow, and generates drift reports.
"""

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd

from src.common.mlflow_utils import log_metrics_safely, start_mlflow_run
from src.monitoring.drift_detector import detect_drift_from_files

logger = logging.getLogger(__name__)

# Default artifact paths
EMBEDDINGS_PATH = "data/features/embeddings.parquet"
CLEANED_ARTICLES_PATH = "data/processed/cleaned_articles.parquet"
CLUSTER_ASSIGNMENTS_PATH = "models/clustering/cluster_assignments.parquet"

# Baseline artifact paths (previous run)
BASELINE_DIR = "data/baseline"
BASELINE_EMBEDDINGS_PATH = os.path.join(BASELINE_DIR, "embeddings.parquet")
BASELINE_ARTICLES_PATH = os.path.join(BASELINE_DIR, "cleaned_articles.parquet")
BASELINE_ASSIGNMENTS_PATH = os.path.join(BASELINE_DIR, "cluster_assignments.parquet")


def save_baseline_artifacts() -> None:
    """Save current artifacts as baseline for future drift detection."""
    os.makedirs(BASELINE_DIR, exist_ok=True)
    
    try:
        if os.path.exists(EMBEDDINGS_PATH):
            import shutil
            shutil.copy2(EMBEDDINGS_PATH, BASELINE_EMBEDDINGS_PATH)
            logger.info("Saved baseline embeddings")
        
        if os.path.exists(CLEANED_ARTICLES_PATH):
            import shutil
            shutil.copy2(CLEANED_ARTICLES_PATH, BASELINE_ARTICLES_PATH)
            logger.info("Saved baseline articles")
        
        if os.path.exists(CLUSTER_ASSIGNMENTS_PATH):
            import shutil
            shutil.copy2(CLUSTER_ASSIGNMENTS_PATH, BASELINE_ASSIGNMENTS_PATH)
            logger.info("Saved baseline cluster assignments")
    except Exception as exc:
        logger.warning("Failed to save baseline artifacts: %s", exc)


def calculate_drift_metrics(
    threshold: float = 0.3,
    config_path: str = "config.yaml",
) -> Dict[str, any]:
    """
    Calculate drift metrics between baseline and current data.
    
    Args:
        threshold: KS test p-value threshold for drift detection
        config_path: Path to config.yaml
    
    Returns:
        Dict with drift metrics and detection results
    """
    # Check if baseline exists
    if not os.path.exists(BASELINE_EMBEDDINGS_PATH):
        logger.info("No baseline artifacts found, skipping drift detection")
        return {"drift_detected": False, "reason": "no_baseline"}
    
    # Detect drift
    drift_results = detect_drift_from_files(
        baseline_embeddings_path=BASELINE_EMBEDDINGS_PATH,
        current_embeddings_path=EMBEDDINGS_PATH,
        baseline_articles_path=BASELINE_ARTICLES_PATH,
        current_articles_path=CLEANED_ARTICLES_PATH,
        baseline_assignments_path=BASELINE_ASSIGNMENTS_PATH,
        current_assignments_path=CLUSTER_ASSIGNMENTS_PATH,
    )
    
    # Determine if drift is detected based on threshold
    embedding_ks_pvalue = drift_results.get("embedding_ks_pvalue", 1.0)
    text_ks_pvalue = drift_results.get("text_length_ks_pvalue", 1.0)
    cluster_ks_pvalue = drift_results.get("cluster_cluster_size_ks_pvalue", 1.0)
    
    drift_detected = (
        embedding_ks_pvalue < threshold
        or text_ks_pvalue < threshold
        or cluster_ks_pvalue < threshold
    )
    
    drift_results["drift_detected"] = drift_detected
    drift_results["threshold"] = threshold
    
    return drift_results


def log_drift_metrics_to_mlflow(
    drift_metrics: Dict[str, any],
    config_path: str = "config.yaml",
) -> None:
    """Log drift metrics to MLflow."""
    try:
        with start_mlflow_run("drift_detection", config_path=config_path):
            # Filter numeric metrics
            numeric_metrics = {
                k: v for k, v in drift_metrics.items()
                if isinstance(v, (int, float))
            }
            log_metrics_safely(numeric_metrics, prefix="drift")
            
            # Log drift detection status
            log_metrics_safely({
                "drift_detected": 1.0 if drift_metrics.get("drift_detected") else 0.0,
            }, prefix="drift")
            
            logger.info("Logged drift metrics to MLflow")
    except Exception as exc:
        logger.warning("Failed to log drift metrics to MLflow: %s", exc)


def generate_drift_report(
    drift_metrics: Dict[str, any],
    output_path: str = "reports/drift_report.json",
) -> None:
    """
    Generate a drift report as JSON.
    
    Args:
        drift_metrics: Dict with drift metrics
        output_path: Path to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(drift_metrics, f, indent=2)
    
    logger.info("Saved drift report to %s", output_path)


def run_drift_detection(
    threshold: float = 0.3,
    save_baseline: bool = True,
    log_to_mlflow: bool = True,
    config_path: str = "config.yaml",
) -> Dict[str, any]:
    """
    Run complete drift detection workflow.
    
    Args:
        threshold: KS test p-value threshold
        save_baseline: If True, save current artifacts as new baseline
        log_to_mlflow: If True, log metrics to MLflow
        config_path: Path to config.yaml
    
    Returns:
        Dict with drift metrics
    """
    logger.info("Running drift detection (threshold=%.3f)", threshold)
    
    # Calculate drift metrics
    drift_metrics = calculate_drift_metrics(threshold=threshold, config_path=config_path)
    
    # Log to MLflow
    if log_to_mlflow:
        log_drift_metrics_to_mlflow(drift_metrics, config_path=config_path)
    
    # Generate report
    generate_drift_report(drift_metrics)
    
    # Save baseline for next run
    if save_baseline:
        save_baseline_artifacts()
    
    if drift_metrics.get("drift_detected"):
        logger.warning("Drift detected! Review drift metrics for details.")
    else:
        logger.info("No significant drift detected")
    
    return drift_metrics

