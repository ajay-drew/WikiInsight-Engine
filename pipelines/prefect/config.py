"""
Prefect configuration management.

Loads Prefect settings from config.yaml and provides helper functions
for retry/timeout settings and Prefect API client connection.
"""

import os
from typing import Dict, Optional

import yaml


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_prefect_config(config_path: str = "config.yaml") -> Dict:
    """Get Prefect-specific configuration section."""
    config = load_config(config_path)
    return config.get("prefect", {})


def get_retry_count(stage: str, config_path: str = "config.yaml") -> int:
    """Get retry count for a specific pipeline stage."""
    prefect_cfg = get_prefect_config(config_path)
    retries = prefect_cfg.get("retries", {})
    return int(retries.get(stage, 2))  # Default to 2 retries


def get_timeout_seconds(stage: str, config_path: str = "config.yaml") -> Optional[int]:
    """Get timeout in seconds for a specific pipeline stage."""
    prefect_cfg = get_prefect_config(config_path)
    timeouts = prefect_cfg.get("timeouts", {})
    timeout = timeouts.get(stage)
    return int(timeout) if timeout else None


def get_server_url(config_path: str = "config.yaml") -> Optional[str]:
    """Get Prefect server URL if configured."""
    prefect_cfg = get_prefect_config(config_path)
    return prefect_cfg.get("server_url") or os.getenv("PREFECT_API_URL")


def get_notification_config(config_path: str = "config.yaml") -> Dict:
    """Get notification configuration."""
    prefect_cfg = get_prefect_config(config_path)
    return prefect_cfg.get("notifications", {})

