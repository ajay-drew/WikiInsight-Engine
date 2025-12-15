"""
Tests for pipeline progress tracking.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.common.pipeline_progress import (
    get_progress,
    mark_stage_completed,
    mark_stage_error,
    reset_progress,
    update_progress,
)


@pytest.fixture
def temp_progress_file(tmp_path):
    """Create a temporary progress file for testing."""
    with patch("src.common.pipeline_progress.PROGRESS_FILE", str(tmp_path / "progress.json")):
        yield str(tmp_path / "progress.json")


def test_reset_progress(temp_progress_file):
    """Test that reset_progress creates initial state."""
    reset_progress()
    
    progress = get_progress()
    assert progress["current_stage"] is None
    assert "ingestion" in progress["stages"]
    assert "preprocessing" in progress["stages"]
    assert "clustering" in progress["stages"]
    assert "build_graph" in progress["stages"]
    assert progress["overall_progress"] == 0.0


def test_update_progress(temp_progress_file):
    """Test updating progress for a stage."""
    reset_progress()
    
    update_progress("ingestion", "running", 50.0, "Fetching articles...", eta=120.0)
    
    progress = get_progress()
    assert progress["current_stage"] == "ingestion"
    assert progress["stages"]["ingestion"]["status"] == "running"
    assert progress["stages"]["ingestion"]["progress"] == 50.0
    assert progress["stages"]["ingestion"]["message"] == "Fetching articles..."
    assert progress["stages"]["ingestion"]["eta"] == 120.0


def test_mark_stage_completed(temp_progress_file):
    """Test marking a stage as completed."""
    reset_progress()
    
    mark_stage_completed("ingestion", "Ingestion complete")
    
    progress = get_progress()
    assert progress["stages"]["ingestion"]["status"] == "completed"
    assert progress["stages"]["ingestion"]["progress"] == 100.0
    assert progress["stages"]["ingestion"]["message"] == "Ingestion complete"


def test_mark_stage_error(temp_progress_file):
    """Test marking a stage as error."""
    reset_progress()
    
    mark_stage_error("clustering", "Clustering failed")
    
    progress = get_progress()
    assert progress["stages"]["clustering"]["status"] == "error"
    assert progress["stages"]["clustering"]["message"] == "Clustering failed"


def test_overall_progress_calculation(temp_progress_file):
    """Test that overall progress is calculated correctly."""
    reset_progress()
    
    # Update ingestion to 50%
    update_progress("ingestion", "running", 50.0, "Halfway...")
    progress = get_progress()
    # Ingestion weight is 0.25, so 50% of 0.25 = 12.5%
    assert 10.0 < progress["overall_progress"] < 15.0
    
    # Mark ingestion as complete (25% of total)
    mark_stage_completed("ingestion")
    progress = get_progress()
    assert 24.0 < progress["overall_progress"] < 26.0
    
    # Update preprocessing to 50% (25% weight, 50% = 12.5%)
    update_progress("preprocessing", "running", 50.0, "Processing...")
    progress = get_progress()
    # 25% (ingestion) + 12.5% (preprocessing) = 37.5%
    assert 35.0 < progress["overall_progress"] < 40.0


def test_progress_file_persistence(temp_progress_file):
    """Test that progress persists to file."""
    reset_progress()
    update_progress("ingestion", "running", 75.0, "Almost done...")
    
    # Read file directly
    with open(temp_progress_file, "r") as f:
        data = json.load(f)
    
    assert data["stages"]["ingestion"]["progress"] == 75.0
    assert data["stages"]["ingestion"]["status"] == "running"

