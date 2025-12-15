"""
Pipeline progress tracking for real-time updates via SSE.

This module provides centralized progress tracking that writes to a JSON file
which can be read by the FastAPI SSE endpoint for real-time progress streaming.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROGRESS_FILE = os.path.join("data", "pipeline_progress.json")


def _ensure_data_dir():
    """Ensure data directory exists."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)


def reset_progress() -> None:
    """Reset progress file to initial state."""
    _ensure_data_dir()
    initial_state = {
        "current_stage": None,
        "stages": {
            "ingestion": {"status": "pending", "progress": 0.0, "message": "", "eta": None},
            "preprocessing": {"status": "pending", "progress": 0.0, "message": "", "eta": None},
            "clustering": {"status": "pending", "progress": 0.0, "message": "", "eta": None},
            "build_graph": {"status": "pending", "progress": 0.0, "message": "", "eta": None},
        },
        "started_at": None,
        "updated_at": None,
        "overall_progress": 0.0,
    }
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(initial_state, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to reset progress file: %s", exc)


def update_progress(
    stage: str,
    status: str,
    progress: float,
    message: str = "",
    eta: Optional[float] = None,
) -> None:
    """
    Update progress for a specific stage.

    Args:
        stage: Stage name ("ingestion", "preprocessing", "clustering", "build_graph")
        status: Status ("pending", "running", "completed", "error")
        progress: Progress percentage (0.0 to 100.0)
        message: Human-readable message about current task
        eta: Estimated time remaining in seconds (optional)
    """
    _ensure_data_dir()

    # Load existing progress or initialize
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
        else:
            reset_progress()
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load progress file, resetting: %s", exc)
        reset_progress()
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)

    # Update stage
    if stage not in data["stages"]:
        data["stages"][stage] = {"status": "pending", "progress": 0.0, "message": "", "eta": None}

    data["stages"][stage]["status"] = status
    data["stages"][stage]["progress"] = max(0.0, min(100.0, progress))
    data["stages"][stage]["message"] = message
    data["stages"][stage]["eta"] = eta

    # Update current stage if this stage is running
    if status == "running":
        data["current_stage"] = stage

    # Calculate overall progress
    # Stage weights: ingestion 25%, preprocessing 25%, clustering 35%, build_graph 15%
    stage_weights = {
        "ingestion": 0.25,
        "preprocessing": 0.25,
        "clustering": 0.35,
        "build_graph": 0.15,
    }

    overall = 0.0
    for stage_name, weight in stage_weights.items():
        stage_progress = data["stages"][stage_name].get("progress", 0.0)
        if data["stages"][stage_name].get("status") == "completed":
            overall += weight * 100.0
        else:
            overall += weight * stage_progress

    data["overall_progress"] = overall

    # Update timestamps
    if data["started_at"] is None:
        data["started_at"] = datetime.now().isoformat()
    data["updated_at"] = datetime.now().isoformat()

    # Write back
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write progress file: %s", exc)


def get_progress() -> Dict:
    """
    Get current progress state.

    Returns:
        Dictionary with current progress information
    """
    if not os.path.exists(PROGRESS_FILE):
        reset_progress()

    try:
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read progress file: %s", exc)
        reset_progress()
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)


def mark_stage_completed(stage: str, message: str = "") -> None:
    """Mark a stage as completed."""
    update_progress(stage, "completed", 100.0, message or f"{stage.capitalize()} completed")


def mark_stage_error(stage: str, message: str = "") -> None:
    """Mark a stage as error."""
    update_progress(stage, "error", 0.0, message or f"{stage.capitalize()} failed")

