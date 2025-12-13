"""
Alerting system for pipeline failures, drift, and instability.

Sends alerts via webhook or log file when:
- Pipeline failures occur
- High drift scores detected
- Cluster instability detected
- API error rate spikes
"""

import logging
import os
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


def load_alert_config(config_path: str = "config.yaml") -> Dict:
    """Load alerting configuration from config.yaml."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    return config.get("monitoring", {}).get("alerts", {})


def send_webhook_alert(
    message: str,
    webhook_url: str,
    is_error: bool = False,
) -> bool:
    """
    Send alert via webhook.
    
    Args:
        message: Alert message
        webhook_url: Webhook URL
        is_error: Whether this is an error alert
    
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        import requests
        
        payload = {
            "message": message,
            "error": is_error,
            "timestamp": str(logging.Formatter().formatTime(logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            ))),
        }
        
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Failed to send webhook alert: %s", exc)
        return False


def send_alert(
    message: str,
    is_error: bool = False,
    config_path: str = "config.yaml",
) -> None:
    """
    Send alert via configured channels.
    
    Args:
        message: Alert message
        is_error: Whether this is an error alert
        config_path: Path to config.yaml
    """
    alert_cfg = load_alert_config(config_path)
    
    if not alert_cfg.get("enabled", False):
        return
    
    # Log to file
    if is_error:
        logger.error("ALERT: %s", message)
    else:
        logger.warning("ALERT: %s", message)
    
    # Webhook
    webhook_url = alert_cfg.get("webhook_url")
    if webhook_url:
        send_webhook_alert(message, webhook_url, is_error)


def check_drift_alert(
    drift_metrics: Dict[str, any],
    threshold: float = 0.3,
    config_path: str = "config.yaml",
) -> None:
    """
    Check drift metrics and send alert if threshold exceeded.
    
    Args:
        drift_metrics: Dict with drift metrics
        threshold: Drift threshold
        config_path: Path to config.yaml
    """
    if not drift_metrics.get("drift_detected"):
        return
    
    # Load threshold from config
    monitoring_cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            monitoring_cfg = config.get("monitoring", {}).get("drift", {})
    
    threshold = monitoring_cfg.get("threshold", threshold)
    
    embedding_pvalue = drift_metrics.get("embedding_ks_pvalue", 1.0)
    text_pvalue = drift_metrics.get("text_length_ks_pvalue", 1.0)
    cluster_pvalue = drift_metrics.get("cluster_cluster_size_ks_pvalue", 1.0)
    
    if embedding_pvalue < threshold or text_pvalue < threshold or cluster_pvalue < threshold:
        message = (
            f"Data drift detected! "
            f"Embedding p-value: {embedding_pvalue:.4f}, "
            f"Text p-value: {text_pvalue:.4f}, "
            f"Cluster p-value: {cluster_pvalue:.4f}"
        )
        send_alert(message, is_error=True, config_path=config_path)


def check_stability_alert(
    stability_metrics: Dict[str, any],
    ari_threshold: float = 0.7,
    config_path: str = "config.yaml",
) -> None:
    """
    Check cluster stability and send alert if below threshold.
    
    Args:
        stability_metrics: Dict with stability metrics
        ari_threshold: Minimum ARI for stability
        config_path: Path to config.yaml
    """
    if not stability_metrics.get("stability_calculated"):
        return
    
    # Load threshold from config
    monitoring_cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            monitoring_cfg = config.get("monitoring", {}).get("stability", {})
    
    ari_threshold = monitoring_cfg.get("ari_threshold", ari_threshold)
    
    ari = stability_metrics.get("ari", 1.0)
    if ari < ari_threshold:
        message = (
            f"Cluster instability detected! "
            f"ARI: {ari:.3f} (threshold: {ari_threshold:.3f})"
        )
        send_alert(message, is_error=True, config_path=config_path)


def check_api_error_alert(
    error_rate: float,
    threshold: float = 0.1,  # 10% error rate
    config_path: str = "config.yaml",
) -> None:
    """
    Check API error rate and send alert if threshold exceeded.
    
    Args:
        error_rate: Current error rate (0.0 to 1.0)
        threshold: Error rate threshold
        config_path: Path to config.yaml
    """
    if error_rate > threshold:
        message = f"High API error rate detected: {error_rate:.1%} (threshold: {threshold:.1%})"
        send_alert(message, is_error=True, config_path=config_path)

