"""
Cluster stability tracking across runs.

Compares cluster assignments across runs using metrics like:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Jaccard similarity
- Cluster evolution tracking (birth/death, merge/split)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)

# Paths
CLUSTER_ASSIGNMENTS_PATH = "models/clustering/cluster_assignments.parquet"
CLUSTERS_SUMMARY_PATH = "models/clustering/clusters_summary.parquet"
BASELINE_DIR = "data/baseline"
BASELINE_ASSIGNMENTS_PATH = os.path.join(BASELINE_DIR, "cluster_assignments.parquet")
BASELINE_SUMMARY_PATH = os.path.join(BASELINE_DIR, "clusters_summary.parquet")


def calculate_stability_metrics(
    baseline_assignments: pd.DataFrame,
    current_assignments: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate cluster stability metrics between two runs.
    
    Args:
        baseline_assignments: Baseline assignments (title, cluster_id)
        current_assignments: Current assignments (title, cluster_id)
    
    Returns:
        Dict with stability metrics
    """
    # Merge on title to compare assignments
    merged = baseline_assignments.merge(
        current_assignments,
        on="title",
        suffixes=("_baseline", "_current"),
        how="inner",
    )
    
    if len(merged) == 0:
        return {
            "ari": 0.0,
            "nmi": 0.0,
            "jaccard": 0.0,
            "overlap_count": 0,
            "total_count": 0,
        }
    
    # Extract cluster labels
    baseline_labels = merged["cluster_id_baseline"].values
    current_labels = merged["cluster_id_current"].values
    
    # Adjusted Rand Index (ARI)
    ari = float(adjusted_rand_score(baseline_labels, current_labels))
    
    # Normalized Mutual Information (NMI)
    nmi = float(normalized_mutual_info_score(baseline_labels, current_labels))
    
    # Jaccard similarity (same cluster assignment)
    same_cluster = (baseline_labels == current_labels).sum()
    jaccard = same_cluster / len(merged)
    
    return {
        "ari": ari,
        "nmi": nmi,
        "jaccard": jaccard,
        "overlap_count": int(same_cluster),
        "total_count": int(len(merged)),
    }


def track_cluster_evolution(
    baseline_summary: pd.DataFrame,
    current_summary: pd.DataFrame,
) -> Dict[str, any]:
    """
    Track cluster evolution: birth, death, merge, split events.
    
    Args:
        baseline_summary: Baseline cluster summaries (cluster_id, size, keywords, etc.)
        current_summary: Current cluster summaries
    
    Returns:
        Dict with evolution metrics
    """
    baseline_clusters = set(baseline_summary["cluster_id"].values) if "cluster_id" in baseline_summary.columns else set()
    current_clusters = set(current_summary["cluster_id"].values) if "cluster_id" in current_summary.columns else set()
    
    # Cluster birth (new clusters)
    birth_clusters = current_clusters - baseline_clusters
    birth_count = len(birth_clusters)
    
    # Cluster death (disappeared clusters)
    death_clusters = baseline_clusters - current_clusters
    death_count = len(death_clusters)
    
    # Stable clusters (exist in both)
    stable_clusters = baseline_clusters & current_clusters
    stable_count = len(stable_clusters)
    
    # Cluster size changes for stable clusters
    size_changes = []
    if "size" in baseline_summary.columns and "size" in current_summary.columns:
        baseline_sizes = baseline_summary.set_index("cluster_id")["size"]
        current_sizes = current_summary.set_index("cluster_id")["size"]
        
        for cid in stable_clusters:
            baseline_size = baseline_sizes.get(cid, 0)
            current_size = current_sizes.get(cid, 0)
            if baseline_size > 0:
                size_change_pct = ((current_size - baseline_size) / baseline_size) * 100
                size_changes.append(size_change_pct)
    
    avg_size_change = float(np.mean(size_changes)) if size_changes else 0.0
    std_size_change = float(np.std(size_changes)) if size_changes else 0.0
    
    return {
        "birth_count": birth_count,
        "death_count": death_count,
        "stable_count": stable_count,
        "birth_clusters": list(birth_clusters),
        "death_clusters": list(death_clusters),
        "stable_clusters": list(stable_clusters),
        "avg_size_change_pct": avg_size_change,
        "std_size_change_pct": std_size_change,
    }


def calculate_cluster_stability(
    baseline_assignments_path: Optional[str] = None,
    current_assignments_path: Optional[str] = None,
    baseline_summary_path: Optional[str] = None,
    current_summary_path: Optional[str] = None,
) -> Dict[str, any]:
    """
    Calculate cluster stability metrics from files.
    
    Args:
        baseline_assignments_path: Path to baseline assignments
        current_assignments_path: Path to current assignments
        baseline_summary_path: Path to baseline summary
        current_summary_path: Path to current summary
    
    Returns:
        Dict with stability metrics
    """
    # Use default paths if not provided
    baseline_assignments_path = baseline_assignments_path or BASELINE_ASSIGNMENTS_PATH
    current_assignments_path = current_assignments_path or CLUSTER_ASSIGNMENTS_PATH
    baseline_summary_path = baseline_summary_path or BASELINE_SUMMARY_PATH
    current_summary_path = current_summary_path or CLUSTERS_SUMMARY_PATH
    
    results = {}
    
    # Check if baseline exists
    if not os.path.exists(baseline_assignments_path):
        logger.info("No baseline assignments found, skipping stability calculation")
        return {"stability_calculated": False, "reason": "no_baseline"}
    
    try:
        # Load assignments
        baseline_assignments = pd.read_parquet(baseline_assignments_path)
        current_assignments = pd.read_parquet(current_assignments_path)
        
        # Calculate stability metrics
        stability_metrics = calculate_stability_metrics(baseline_assignments, current_assignments)
        results.update(stability_metrics)
        
        # Track evolution if summaries exist
        if os.path.exists(baseline_summary_path) and os.path.exists(current_summary_path):
            baseline_summary = pd.read_parquet(baseline_summary_path)
            current_summary = pd.read_parquet(current_summary_path)
            
            evolution = track_cluster_evolution(baseline_summary, current_summary)
            results.update({f"evolution_{k}": v for k, v in evolution.items()})
        
        results["stability_calculated"] = True
        
    except Exception as exc:
        logger.warning("Failed to calculate cluster stability: %s", exc)
        results["stability_calculated"] = False
        results["error"] = str(exc)
    
    return results


def log_stability_to_mlflow(
    stability_metrics: Dict[str, any],
    config_path: str = "config.yaml",
) -> None:
    """Log stability metrics to MLflow."""
    from src.common.mlflow_utils import log_metrics_safely, start_mlflow_run
    
    try:
        with start_mlflow_run("cluster_stability", config_path=config_path):
            # Filter numeric metrics
            numeric_metrics = {
                k: v for k, v in stability_metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            log_metrics_safely(numeric_metrics, prefix="stability")
            
            logger.info("Logged stability metrics to MLflow")
    except Exception as exc:
        logger.warning("Failed to log stability metrics to MLflow: %s", exc)


def run_stability_check(
    log_to_mlflow: bool = True,
    config_path: str = "config.yaml",
) -> Dict[str, any]:
    """
    Run complete cluster stability check.
    
    Args:
        log_to_mlflow: If True, log metrics to MLflow
        config_path: Path to config.yaml
    
    Returns:
        Dict with stability metrics
    """
    logger.info("Running cluster stability check")
    
    stability_metrics = calculate_cluster_stability()
    
    if log_to_mlflow and stability_metrics.get("stability_calculated"):
        log_stability_to_mlflow(stability_metrics, config_path=config_path)
    
    if stability_metrics.get("stability_calculated"):
        ari = stability_metrics.get("ari", 0.0)
        logger.info("Cluster stability ARI: %.3f", ari)
    
    return stability_metrics

