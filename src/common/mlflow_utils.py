"""
Centralized MLflow helper functions.

Provides consistent MLflow setup, metric naming conventions,
and artifact logging helpers across all pipeline stages.
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_mlflow_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load MLflow configuration from config.yaml."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    return config.get("mlops", {}).get("mlflow", {})


def setup_mlflow_experiment(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
) -> None:
    """
    Set up MLflow experiment and tracking URI.
    
    Args:
        experiment_name: Experiment name (overrides config)
        tracking_uri: Tracking URI (overrides config)
        config_path: Path to config.yaml
    """
    try:
        import mlflow
        
        ml_cfg = load_mlflow_config(config_path)
        
        # Use provided values or fall back to config
        final_tracking_uri = tracking_uri or ml_cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        final_experiment_name = experiment_name or ml_cfg.get("experiment_name", "wikiinsight")
        
        if final_tracking_uri:
            mlflow.set_tracking_uri(final_tracking_uri)
            logger.debug("MLflow tracking URI set to: %s", final_tracking_uri)
        
        if final_experiment_name:
            try:
                mlflow.set_experiment(final_experiment_name)
                logger.debug("MLflow experiment set to: %s", final_experiment_name)
            except Exception as exp_exc:
                # If experiment setup fails due to DB issues, try to create it
                logger.warning("Failed to set experiment, trying to create: %s", exp_exc)
                try:
                    mlflow.create_experiment(final_experiment_name)
                    mlflow.set_experiment(final_experiment_name)
                    logger.info("Created new MLflow experiment: %s", final_experiment_name)
                except Exception as create_exc:
                    logger.warning("Could not create experiment either: %s", create_exc)
                    # Continue anyway - might still work
            
    except ImportError:
        logger.warning("MLflow not installed, skipping setup")
    except Exception as exc:
        logger.warning("Failed to setup MLflow: %s", exc)


def log_params_safely(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters to MLflow with error handling.
    
    Args:
        params: Dictionary of parameters to log
        prefix: Optional prefix for parameter names
    """
    try:
        import mlflow
        
        # Check if we're in an active run
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                logger.debug("No active MLflow run - skipping parameter logging")
                return
        except Exception:
            logger.debug("Could not check MLflow run status - attempting to log anyway")
        
        for key, value in params.items():
            param_name = f"{prefix}_{key}" if prefix else key
            # MLflow only accepts strings, numbers, or booleans
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(param_name, value)
            else:
                mlflow.log_param(param_name, str(value))
                
    except Exception as exc:
        error_msg = str(exc)
        if "Can't locate revision" in error_msg:
            logger.debug("MLflow DB migration issue - skipping parameter logging (non-critical)")
        else:
            logger.warning("Failed to log parameters to MLflow: %s", exc)


def log_metrics_safely(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Log metrics to MLflow with error handling.
    
    Args:
        metrics: Dictionary of metrics to log (values must be numeric)
        prefix: Optional prefix for metric names
    """
    try:
        import mlflow
        
        # Check if we're in an active run
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                logger.debug("No active MLflow run - skipping metric logging")
                return
        except Exception:
            logger.debug("Could not check MLflow run status - attempting to log anyway")
        
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue  # Skip non-numeric values
            
            metric_name = f"{prefix}_{key}" if prefix else key
            mlflow.log_metric(metric_name, float(value))
            
    except Exception as exc:
        error_msg = str(exc)
        if "Can't locate revision" in error_msg:
            logger.debug("MLflow DB migration issue - skipping metric logging (non-critical)")
        else:
            logger.warning("Failed to log metrics to MLflow: %s", exc)


def start_mlflow_run(
    run_name: str,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
):
    """
    Start an MLflow run with proper setup.
    
    Args:
        run_name: Name for this run
        experiment_name: Experiment name (overrides config)
        tracking_uri: Tracking URI (overrides config)
        config_path: Path to config.yaml
    
    Returns:
        MLflow run context manager
    """
    try:
        import mlflow
        
        setup_mlflow_experiment(experiment_name, tracking_uri, config_path)
        
            # Try to start run, handle DB migration errors gracefully
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as run_exc:
            error_msg = str(run_exc)
            # Check if it's a migration error
            if "Can't locate revision" in error_msg or "alembic" in error_msg.lower():
                logger.warning("MLflow database migration issue detected.")
                logger.warning("  - Error: %s", error_msg)
                logger.warning("  - Solution: Run 'python fix_mlflow_db.py' to fix the database")
                logger.warning("  - MLflow logging will be disabled for this run")
                logger.warning("  - This is non-critical - pipeline will continue normally")
                from contextlib import nullcontext
                return nullcontext()
            else:
                # Log other errors but don't fail
                logger.warning("MLflow run start failed: %s", error_msg)
                logger.warning("MLflow logging will be disabled for this run")
                from contextlib import nullcontext
                return nullcontext()
        
    except ImportError:
        logger.warning("MLflow not installed, returning dummy context manager")
        from contextlib import nullcontext
        return nullcontext()
    except Exception as exc:
        logger.warning("Failed to start MLflow run: %s", exc)
        from contextlib import nullcontext
        return nullcontext()


def log_artifact_safely(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log an artifact to MLflow with error handling.
    
    Args:
        local_path: Path to local file/directory to log
        artifact_path: Optional path within artifact store
    """
    try:
        import mlflow
        
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            logger.warning("Artifact path does not exist: %s", local_path)
            
    except Exception as exc:
        logger.warning("Failed to log artifact to MLflow: %s", exc)

