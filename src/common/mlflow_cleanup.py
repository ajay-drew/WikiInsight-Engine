"""
MLflow cleanup and organization utilities.

Provides functions to:
- List experiments and runs
- Clean up old or corrupted runs
- Organize experiments with tags
- Export metrics summaries
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def list_experiments(
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
) -> List[Dict]:
    """
    List all MLflow experiments with metadata.
    
    Args:
        tracking_uri: MLflow tracking URI (overrides config)
        config_path: Path to config.yaml
        
    Returns:
        List of experiment dictionaries with metadata
    """
    try:
        import mlflow
        from src.common.mlflow_utils import load_mlflow_config
        
        # Load config
        ml_cfg = load_mlflow_config(config_path)
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
        
        experiments = mlflow.search_experiments()
        
        result = []
        for exp in experiments:
            # Get run count
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
            run_count = len(runs) if runs is not None else 0
            
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "run_count": run_count,
                "tags": exp.tags or {},
                "creation_time": exp.creation_time,
            })
        
        return result
        
    except ImportError:
        logger.warning("MLflow not installed")
        return []
    except Exception as exc:
        logger.warning("Failed to list experiments: %s", exc)
        return []


def cleanup_old_runs(
    experiment_name: str,
    days_old: Optional[int] = None,
    keep_latest: Optional[int] = None,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
    dry_run: bool = True,
) -> Dict[str, int]:
    """
    Clean up old runs from an experiment.
    
    Args:
        experiment_name: Name of experiment to clean
        days_old: Delete runs older than N days (if provided)
        keep_latest: Keep only the latest N runs (if provided)
        tracking_uri: MLflow tracking URI (overrides config)
        config_path: Path to config.yaml
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dictionary with counts of runs found, deleted, etc.
    """
    try:
        import mlflow
        from src.common.mlflow_utils import load_mlflow_config
        
        # Load config
        ml_cfg = load_mlflow_config(config_path)
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
        
        # Get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning("Experiment '%s' not found", experiment_name)
                return {"found": 0, "deleted": 0, "kept": 0}
        except Exception as exc:
            logger.warning("Failed to get experiment '%s': %s", experiment_name, exc)
            return {"found": 0, "deleted": 0, "kept": 0}
        
        # Get all runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        
        if runs is None or len(runs) == 0:
            return {"found": 0, "deleted": 0, "kept": 0}
        
        runs_to_delete = []
        
        # Filter by age
        if days_old is not None:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            for _, run in runs.iterrows():
                start_time = datetime.fromtimestamp(run["start_time"] / 1000)
                if start_time < cutoff_time:
                    runs_to_delete.append(run["run_id"])
        
        # Filter by keep_latest
        if keep_latest is not None:
            if len(runs) > keep_latest:
                runs_to_keep = runs.head(keep_latest)["run_id"].tolist()
                runs_to_delete = [
                    run_id for run_id in runs["run_id"].tolist()
                    if run_id not in runs_to_keep
                ]
        
        # Remove duplicates
        runs_to_delete = list(set(runs_to_delete))
        
        if dry_run:
            logger.info("DRY RUN: Would delete %d runs from experiment '%s'", len(runs_to_delete), experiment_name)
            return {
                "found": len(runs),
                "deleted": 0,
                "would_delete": len(runs_to_delete),
                "kept": len(runs) - len(runs_to_delete),
            }
        else:
            # Actually delete runs
            deleted_count = 0
            for run_id in runs_to_delete:
                try:
                    mlflow.delete_run(run_id)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning("Failed to delete run %s: %s", run_id, exc)
            
            logger.info("Deleted %d runs from experiment '%s'", deleted_count, experiment_name)
            return {
                "found": len(runs),
                "deleted": deleted_count,
                "kept": len(runs) - deleted_count,
            }
            
    except ImportError:
        logger.warning("MLflow not installed")
        return {"found": 0, "deleted": 0, "kept": 0}
    except Exception as exc:
        logger.warning("Failed to cleanup old runs: %s", exc)
        return {"found": 0, "deleted": 0, "kept": 0}


def organize_experiments(
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
) -> bool:
    """
    Add tags to an experiment for better organization.
    
    Args:
        experiment_name: Name of experiment to organize
        tags: Dictionary of tags to add
        tracking_uri: MLflow tracking URI (overrides config)
        config_path: Path to config.yaml
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import mlflow
        from src.common.mlflow_utils import load_mlflow_config
        
        # Load config
        ml_cfg = load_mlflow_config(config_path)
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
        
        # Get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning("Experiment '%s' not found", experiment_name)
                return False
        except Exception as exc:
            logger.warning("Failed to get experiment '%s': %s", experiment_name, exc)
            return False
        
        if tags:
            # Update experiment tags
            client = mlflow.tracking.MlflowClient()
            for key, value in tags.items():
                try:
                    client.set_experiment_tag(experiment.experiment_id, key, str(value))
                except Exception as exc:
                    logger.warning("Failed to set tag %s=%s: %s", key, value, exc)
            
            logger.info("Added tags to experiment '%s': %s", experiment_name, tags)
        
        return True
        
    except ImportError:
        logger.warning("MLflow not installed")
        return False
    except Exception as exc:
        logger.warning("Failed to organize experiments: %s", exc)
        return False


def export_metrics_summary(
    experiment_name: str,
    output_path: str = "mlflow_metrics_summary.csv",
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
) -> bool:
    """
    Export metrics summary from all runs in an experiment to CSV.
    
    Args:
        experiment_name: Name of experiment to export
        output_path: Path to output CSV file
        tracking_uri: MLflow tracking URI (overrides config)
        config_path: Path to config.yaml
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import mlflow
        from src.common.mlflow_utils import load_mlflow_config
        
        # Load config
        ml_cfg = load_mlflow_config(config_path)
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
        
        # Get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning("Experiment '%s' not found", experiment_name)
                return False
        except Exception as exc:
            logger.warning("Failed to get experiment '%s': %s", experiment_name, exc)
            return False
        
        # Get all runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )
        
        if runs is None or len(runs) == 0:
            logger.warning("No runs found in experiment '%s'", experiment_name)
            return False
        
        # Export to CSV
        runs.to_csv(output_path, index=False)
        logger.info("Exported %d runs to %s", len(runs), output_path)
        
        return True
        
    except ImportError:
        logger.warning("MLflow not installed")
        return False
    except Exception as exc:
        logger.warning("Failed to export metrics summary: %s", exc)
        return False


def clean_corrupted_runs(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
    dry_run: bool = True,
) -> Dict[str, int]:
    """
    Clean up corrupted runs (runs with migration errors or invalid data).
    
    Args:
        experiment_name: Name of experiment to clean
        tracking_uri: MLflow tracking URI (overrides config)
        config_path: Path to config.yaml
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dictionary with counts of corrupted runs found/deleted
    """
    try:
        import mlflow
        from src.common.mlflow_utils import load_mlflow_config
        
        # Load config
        ml_cfg = load_mlflow_config(config_path)
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
        
        # Get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning("Experiment '%s' not found", experiment_name)
                return {"found": 0, "deleted": 0}
        except Exception as exc:
            logger.warning("Failed to get experiment '%s': %s", experiment_name, exc)
            return {"found": 0, "deleted": 0}
        
        # Get all runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
        )
        
        if runs is None or len(runs) == 0:
            return {"found": 0, "deleted": 0}
        
        corrupted_runs = []
        
        # Check each run for corruption indicators
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            try:
                # Try to get run details - if this fails, run might be corrupted
                run_info = mlflow.get_run(run_id)
                # Check for common corruption indicators
                if run_info.status == "FAILED" and (
                    "migration" in str(run_info).lower() or
                    "alembic" in str(run_info).lower() or
                    "revision" in str(run_info).lower()
                ):
                    corrupted_runs.append(run_id)
            except Exception:
                # If we can't even get run info, it's likely corrupted
                corrupted_runs.append(run_id)
        
        if dry_run:
            logger.info("DRY RUN: Found %d potentially corrupted runs", len(corrupted_runs))
            return {
                "found": len(corrupted_runs),
                "deleted": 0,
                "would_delete": len(corrupted_runs),
            }
        else:
            deleted_count = 0
            for run_id in corrupted_runs:
                try:
                    mlflow.delete_run(run_id)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning("Failed to delete corrupted run %s: %s", run_id, exc)
            
            logger.info("Deleted %d corrupted runs from experiment '%s'", deleted_count, experiment_name)
            return {
                "found": len(corrupted_runs),
                "deleted": deleted_count,
            }
            
    except ImportError:
        logger.warning("MLflow not installed")
        return {"found": 0, "deleted": 0}
    except Exception as exc:
        logger.warning("Failed to clean corrupted runs: %s", exc)
        return {"found": 0, "deleted": 0}
