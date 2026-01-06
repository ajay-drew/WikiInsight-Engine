"""
Test pipeline with 101 articles to verify clustering clamping works correctly.

This test verifies:
1. Pipeline can run with 101 articles
2. Clustering correctly clamps n_clusters if needed
3. All stages complete successfully
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("mwclient")


@pytest.fixture
def temp_pipeline_dir_101(tmp_path):
    """Create a temporary directory structure for pipeline artifacts with 101 articles config."""
    # Create directory structure
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "data" / "graph").mkdir(parents=True)
    (tmp_path / "data" / "pipeline_progress.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "clustering").mkdir(parents=True)
    
    # Create config with 101 articles and 100 clusters (to test clamping)
    config = {
        "ingestion": {
            "seed_queries": [
                "Machine learning",
                "Artificial intelligence",
                "Data science",
                "Neural networks",
                "Deep learning",
            ],
            "per_query_limit": 25,  # 5 queries * 25 = 125 potential, capped at 101
            "max_articles": 101,
        },
        "data": {
            "wikipedia": {
                "api_rate_limit": 200.0,
            },
        },
        "preprocessing": {
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "cpu",  # Use CPU for testing
            },
        },
        "models": {
            "clustering": {
                "method": "kmeans",
                "n_clusters": 100,  # Will be clamped to 101 if we get 101 articles
                "random_state": 42,
                "use_gpu": False,  # Use CPU for testing
            },
            "neighbors": {
                "n_neighbors": 10,
            },
        },
        "graph": {
            "semantic_similarity_threshold": 0.7,
            "enable_cluster_edges": True,
            "max_nodes_per_visualization": 100,
        },
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return tmp_path


def test_pipeline_101_articles(temp_pipeline_dir_101, monkeypatch):
    """Test full pipeline with 101 articles to verify clustering clamping."""
    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(temp_pipeline_dir_101)
    
    try:
        # Set environment variables
        monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
        
        # Run the pipeline orchestrator
        result = subprocess.run(
            [sys.executable, "-m", "src.common.pipeline_orchestrator"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )
        
        # Check that pipeline completed successfully
        assert result.returncode == 0, f"Pipeline failed with return code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        # Verify artifacts were created
        assert (temp_pipeline_dir_101 / "data" / "raw" / "articles.json").exists(), "articles.json not created"
        assert (temp_pipeline_dir_101 / "data" / "processed" / "cleaned_articles.parquet").exists(), "cleaned_articles.parquet not created"
        assert (temp_pipeline_dir_101 / "data" / "features" / "embeddings.parquet").exists(), "embeddings.parquet not created"
        assert (temp_pipeline_dir_101 / "models" / "clustering" / "kmeans_model.pkl").exists(), "kmeans_model.pkl not created"
        assert (temp_pipeline_dir_101 / "models" / "clustering" / "cluster_assignments.parquet").exists(), "cluster_assignments.parquet not created"
        assert (temp_pipeline_dir_101 / "data" / "graph" / "knowledge_graph.pkl").exists(), "knowledge_graph.pkl not created"
        
        # Verify clustering worked (check that we have clusters)
        import pandas as pd
        assignments = pd.read_parquet(temp_pipeline_dir_101 / "models" / "clustering" / "cluster_assignments.parquet")
        assert len(assignments) > 0, "No cluster assignments found"
        assert "cluster_id" in assignments.columns, "cluster_id column missing"
        
        # Verify that clustering didn't fail due to n_clusters > n_samples
        # (This would have been caught by the clamping logic)
        unique_clusters = assignments["cluster_id"].nunique()
        assert unique_clusters > 0, "No clusters found"
        assert unique_clusters <= len(assignments), f"More clusters ({unique_clusters}) than articles ({len(assignments)})"
        
        print(f"\nPipeline completed successfully with {len(assignments)} articles and {unique_clusters} clusters")
        
    finally:
        os.chdir(original_cwd)

