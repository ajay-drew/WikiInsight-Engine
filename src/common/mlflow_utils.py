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
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and logged as warnings.
    
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
            try:
                mlflow.set_tracking_uri(final_tracking_uri)
                logger.debug("MLflow tracking URI set to: %s", final_tracking_uri)
            except Exception as uri_exc:
                logger.debug("Failed to set MLflow tracking URI (non-critical): %s", uri_exc)
                return  # Exit early if URI setup fails
        
        if final_experiment_name:
            try:
                mlflow.set_experiment(final_experiment_name)
                logger.debug("MLflow experiment set to: %s", final_experiment_name)
            except Exception as exp_exc:
                # If experiment setup fails due to DB issues, try to create it
                logger.debug("Failed to set experiment, trying to create: %s", exp_exc)
                try:
                    mlflow.create_experiment(final_experiment_name)
                    mlflow.set_experiment(final_experiment_name)
                    logger.debug("Created new MLflow experiment: %s", final_experiment_name)
                except Exception as create_exc:
                    logger.debug("Could not create experiment (non-critical): %s", create_exc)
                    # Continue anyway - might still work
            
    except ImportError:
        logger.debug("MLflow not installed, skipping setup (non-critical)")
    except Exception as exc:
        logger.debug("Failed to setup MLflow (non-critical): %s", exc)


def log_params_safely(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters to MLflow with error handling.
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and logged as debug messages.
    
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
                logger.debug("No active MLflow run - skipping parameter logging (non-critical)")
                return
        except Exception:
            logger.debug("Could not check MLflow run status (non-critical)")
            return
        
        for key, value in params.items():
            try:
                param_name = f"{prefix}_{key}" if prefix else key
                # MLflow only accepts strings, numbers, or booleans
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(param_name, value)
                else:
                    mlflow.log_param(param_name, str(value))
            except Exception as param_exc:
                logger.debug("Failed to log parameter %s (non-critical): %s", key, param_exc)
                # Continue with other parameters
                
    except ImportError:
        logger.debug("MLflow not installed (non-critical)")
    except Exception as exc:
        error_msg = str(exc)
        if "Can't locate revision" in error_msg:
            logger.debug("MLflow DB migration issue (non-critical)")
        else:
            logger.debug("Failed to log parameters to MLflow (non-critical): %s", exc)


def log_metrics_safely(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Log metrics to MLflow with error handling.
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and logged as debug messages.
    
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
                logger.debug("No active MLflow run - skipping metric logging (non-critical)")
                return
        except Exception:
            logger.debug("Could not check MLflow run status (non-critical)")
            return
        
        for key, value in metrics.items():
            try:
                if not isinstance(value, (int, float)):
                    continue  # Skip non-numeric values
                
                metric_name = f"{prefix}_{key}" if prefix else key
                mlflow.log_metric(metric_name, float(value))
            except Exception as metric_exc:
                logger.debug("Failed to log metric %s (non-critical): %s", key, metric_exc)
                # Continue with other metrics
            
    except ImportError:
        logger.debug("MLflow not installed (non-critical)")
    except Exception as exc:
        error_msg = str(exc)
        if "Can't locate revision" in error_msg:
            logger.debug("MLflow DB migration issue (non-critical)")
        else:
            logger.debug("Failed to log metrics to MLflow (non-critical): %s", exc)


def start_mlflow_run(
    run_name: str,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    config_path: str = "config.yaml",
):
    """
    Start an MLflow run with proper setup.
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and a null context manager is returned.
    
    Args:
        run_name: Name for this run
        experiment_name: Experiment name (overrides config)
        tracking_uri: Tracking URI (overrides config)
        config_path: Path to config.yaml
    
    Returns:
        MLflow run context manager (or nullcontext if MLflow fails)
    """
    from contextlib import nullcontext
    
    try:
        import mlflow
        
        # Setup is non-blocking - errors are logged but don't propagate
        setup_mlflow_experiment(experiment_name, tracking_uri, config_path)
        
        # Try to start run, handle all errors gracefully
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as run_exc:
            error_msg = str(run_exc)
            # Check if it's a migration error
            if "Can't locate revision" in error_msg or "alembic" in error_msg.lower():
                logger.debug("MLflow database migration issue (non-critical): %s", error_msg)
            else:
                logger.debug("MLflow run start failed (non-critical): %s", error_msg)
            # Always return nullcontext on error - never raise
            return nullcontext()
        
    except ImportError:
        logger.debug("MLflow not installed (non-critical)")
        return nullcontext()
    except Exception as exc:
        logger.debug("Failed to start MLflow run (non-critical): %s", exc)
        return nullcontext()


def log_artifact_safely(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log an artifact to MLflow with error handling.
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and logged as debug messages.
    
    Args:
        local_path: Path to local file/directory to log
        artifact_path: Optional path within artifact store
    """
    try:
        import mlflow
        
        if os.path.exists(local_path):
            try:
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as artifact_exc:
                logger.debug("Failed to log artifact (non-critical): %s", artifact_exc)
        else:
            logger.debug("Artifact path does not exist (non-critical): %s", local_path)
            
    except ImportError:
        logger.debug("MLflow not installed (non-critical)")
    except Exception as exc:
        logger.debug("Failed to log artifact to MLflow (non-critical): %s", exc)

