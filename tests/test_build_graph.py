"""
Tests for graph construction pipeline (src/graph/build_graph.py).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.build_graph import (
    CLEANED_ARTICLES_PATH,
    CLUSTER_ASSIGNMENTS_PATH,
    EMBEDDINGS_PATH,
    load_articles,
    load_cluster_assignments,
    load_embeddings,
    main,
)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        os.makedirs(os.path.join(tmpdir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "features"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "graph"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "clustering"), exist_ok=True)
        
        # Create sample data files
        articles_df = pd.DataFrame({
            "title": ["Article A", "Article B"],
            "cleaned_text": ["text a", "text b"],
            "links": [["Article B"], []],
            "categories": [["Cat1"], ["Cat2"]],
        })
        articles_df.to_parquet(
            os.path.join(tmpdir, "processed", "cleaned_articles.parquet"),
            index=False,
        )
        
        cluster_assignments = pd.DataFrame({
            "title": ["Article A", "Article B"],
            "cluster_id": [0, 1],
        })
        cluster_assignments.to_parquet(
            os.path.join(tmpdir, "clustering", "cluster_assignments.parquet"),
            index=False,
        )
        
        embeddings_df = pd.DataFrame({
            "title": ["Article A", "Article B"],
            "embedding": [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            ],
        })
        embeddings_df.to_parquet(
            os.path.join(tmpdir, "features", "embeddings.parquet"),
            index=False,
        )
        
        yield tmpdir


def test_load_articles(temp_data_dir):
    """Test loading articles from parquet file."""
    test_path = os.path.join(temp_data_dir, "processed", "cleaned_articles.parquet")
    with patch("src.graph.build_graph.CLEANED_ARTICLES_PATH", test_path):
        df = load_articles()
        assert len(df) == 2
        assert "title" in df.columns
        assert "links" in df.columns


def test_load_articles_missing_file():
    """Test loading articles when file is missing."""
    import src.graph.build_graph as bg_module
    original = bg_module.CLEANED_ARTICLES_PATH
    try:
        bg_module.CLEANED_ARTICLES_PATH = "nonexistent.parquet"
        with pytest.raises(FileNotFoundError):
            load_articles()
    finally:
        bg_module.CLEANED_ARTICLES_PATH = original


def test_load_cluster_assignments(temp_data_dir):
    """Test loading cluster assignments from parquet file."""
    test_path = os.path.join(temp_data_dir, "clustering", "cluster_assignments.parquet")
    import src.graph.build_graph as bg_module
    original = bg_module.CLUSTER_ASSIGNMENTS_PATH
    try:
        bg_module.CLUSTER_ASSIGNMENTS_PATH = test_path
        df = load_cluster_assignments()
        assert len(df) == 2
        assert "title" in df.columns
        assert "cluster_id" in df.columns
    finally:
        bg_module.CLUSTER_ASSIGNMENTS_PATH = original


def test_load_cluster_assignments_missing_file():
    """Test loading cluster assignments when file is missing."""
    import src.graph.build_graph as bg_module
    original = bg_module.CLUSTER_ASSIGNMENTS_PATH
    try:
        bg_module.CLUSTER_ASSIGNMENTS_PATH = "nonexistent.parquet"
        with pytest.raises(FileNotFoundError):
            load_cluster_assignments()
    finally:
        bg_module.CLUSTER_ASSIGNMENTS_PATH = original


def test_load_embeddings(temp_data_dir):
    """Test loading embeddings from parquet file."""
    test_path = os.path.join(temp_data_dir, "features", "embeddings.parquet")
    import src.graph.build_graph as bg_module
    original = bg_module.EMBEDDINGS_PATH
    try:
        bg_module.EMBEDDINGS_PATH = test_path
        embeddings = load_embeddings()
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
    finally:
        bg_module.EMBEDDINGS_PATH = original


def test_load_embeddings_missing_file():
    """Test loading embeddings when file is missing."""
    import src.graph.build_graph as bg_module
    original = bg_module.EMBEDDINGS_PATH
    try:
        bg_module.EMBEDDINGS_PATH = "nonexistent.parquet"
        with pytest.raises(FileNotFoundError):
            load_embeddings()
    finally:
        bg_module.EMBEDDINGS_PATH = original


def test_load_embeddings_missing_column(temp_data_dir):
    """Test loading embeddings when embedding column is missing."""
    # Create invalid embeddings file
    invalid_df = pd.DataFrame({"title": ["A"], "other": [1]})
    invalid_path = os.path.join(temp_data_dir, "features", "invalid_embeddings.parquet")
    invalid_df.to_parquet(invalid_path, index=False)
    
    import src.graph.build_graph as bg_module
    original = bg_module.EMBEDDINGS_PATH
    try:
        bg_module.EMBEDDINGS_PATH = invalid_path
        with pytest.raises(ValueError, match="embedding"):
            load_embeddings()
    finally:
        bg_module.EMBEDDINGS_PATH = original


@patch("src.graph.build_graph.load_config")
@patch("src.graph.build_graph.load_articles")
@patch("src.graph.build_graph.load_cluster_assignments")
@patch("src.graph.build_graph.load_embeddings")
@patch("src.graph.build_graph.KnowledgeGraphBuilder")
@patch("src.graph.build_graph.GRAPH_PATH")
def test_main_success(
    mock_graph_path,
    mock_builder_class,
    mock_load_embeddings,
    mock_load_clusters,
    mock_load_articles,
    mock_load_config,
    temp_data_dir,
):
    """Test successful graph construction pipeline."""
    # Setup mocks
    mock_config = {"graph": {"semantic_similarity_threshold": 0.7, "enable_cluster_edges": True}}
    mock_load_config.return_value = mock_config
    
    mock_articles = pd.DataFrame({
        "title": ["A", "B"],
        "links": [["B"], []],
        "categories": [["Cat1"], ["Cat2"]],
    })
    mock_load_articles.return_value = mock_articles
    
    mock_clusters = pd.DataFrame({
        "title": ["A", "B"],
        "cluster_id": [0, 1],
    })
    mock_load_clusters.return_value = mock_clusters
    
    mock_embeddings = np.random.normal(size=(2, 10))
    mock_load_embeddings.return_value = mock_embeddings
    
    mock_builder = MagicMock()
    mock_builder.build_graph.return_value = MagicMock()
    mock_builder_class.return_value = mock_builder
    
    graph_file = os.path.join(temp_data_dir, "graph", "knowledge_graph.pkl")
    mock_graph_path.__str__ = lambda x: graph_file
    
    # Run main
    main()
    
    # Verify builder was called
    mock_builder.build_graph.assert_called_once()
    mock_builder.save_graph.assert_called_once()


@patch("src.graph.build_graph.load_config")
@patch("src.graph.build_graph.load_articles")
def test_main_missing_articles(mock_load_articles, mock_load_config):
    """Test main function when articles are missing."""
    mock_load_config.return_value = {}
    mock_load_articles.side_effect = FileNotFoundError("Articles not found")
    
    with pytest.raises(FileNotFoundError):
        main()

