"""
Shared logging configuration helpers.

Uses `config.yaml` logging section and optional environment overrides
to configure the root logger with both console and file handlers.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(config_path: str = "config.yaml") -> None:
    """
    Initialize application-wide logging configuration.

    - Reads `logging.level`, `logging.format`, and `logging.file` from config.yaml.
    - Allows overriding the log level via LOG_LEVEL environment variable.
    - Configures both console and file handlers (if a file path is provided).
    """
    config = _load_yaml(config_path)
    logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}

    # Determine log level (env var takes precedence)
    env_level = os.getenv("LOG_LEVEL")
    level_name = (env_level or logging_cfg.get("level") or "INFO").upper()

    log_format = logging_cfg.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log_file = logging_cfg.get("file")

    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": level_name,
        },
    }
    root_handlers = ["console"]

    if log_file:
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "level": level_name,
            "filename": log_file,
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    dict_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
            },
        },
        "handlers": handlers,
        "root": {
            "level": level_name,
            "handlers": root_handlers,
        },
    }

    logging.config.dictConfig(dict_config)


