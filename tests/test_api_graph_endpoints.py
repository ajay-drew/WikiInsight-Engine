"""
Tests for graph API endpoints in src/api/main.py.
"""

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.graph.graph_service import GraphService
from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.modeling.topic_index import TopicIndex


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    articles_df = pd.DataFrame({
        "title": ["Article A", "Article B", "Article C"],
        "links": [["Article B"], ["Article C"], []],
        "categories": [["Cat1"], ["Cat1"], ["Cat2"]],
    })
    
    cluster_assignments = pd.DataFrame({
        "title": ["Article A", "Article B", "Article C"],
        "cluster_id": [0, 0, 1],
    })
    
    embeddings = np.random.normal(size=(3, 10))
    
    builder = KnowledgeGraphBuilder(
        semantic_threshold=1.0,
        enable_cluster_edges=True,
    )
    return builder.build_graph(articles_df, cluster_assignments, embeddings)


@pytest.fixture
def mock_graph_service(sample_graph):
    """Create a mock GraphService."""
    return GraphService(graph=sample_graph)


@pytest.fixture
def mock_topic_index():
    """Create a mock TopicIndex."""
    assignments_df = pd.DataFrame({
        "title": ["Article A", "Article B", "Article C"],
        "cluster_id": [0, 0, 1],
    })
    
    mock_index = MagicMock(spec=TopicIndex)
    mock_index.assignments_df = assignments_df
    return mock_index


@pytest.fixture
def client_with_graph(mock_graph_service, mock_topic_index):
    """Create a test client with graph service injected."""
    with patch("src.api.main._graph_service", mock_graph_service):
        with patch("src.api.main._topic_index", mock_topic_index):
            yield TestClient(app)


def test_graph_neighbors_endpoint(client_with_graph):
    """Test /api/graph/neighbors/{article_title} endpoint."""
    resp = client_with_graph.get("/api/graph/neighbors/Article%20A")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "article_title" in data
    assert "neighbors" in data
    assert isinstance(data["neighbors"], list)


def test_graph_neighbors_endpoint_nonexistent(client_with_graph):
    """Test graph neighbors endpoint with nonexistent article."""
    resp = client_with_graph.get("/api/graph/neighbors/NonExistent")
    assert resp.status_code == 200  # Should return empty neighbors list
    data = resp.json()
    assert data["neighbors"] == []


def test_graph_neighbors_endpoint_no_service():
    """Test graph neighbors endpoint when graph service is unavailable."""
    with patch("src.api.main._graph_service", None):
        client = TestClient(app)
        resp = client.get("/api/graph/neighbors/Article%20A")
        assert resp.status_code == 503


def test_graph_path_endpoint(client_with_graph):
    """Test /api/graph/path/{from_title}/{to_title} endpoint."""
    resp = client_with_graph.get("/api/graph/path/Article%20A/Article%20C")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "from_title" in data
    assert "to_title" in data
    assert "path" in data
    assert "found" in data
    assert isinstance(data["found"], bool)


def test_graph_path_endpoint_no_service():
    """Test graph path endpoint when graph service is unavailable."""
    with patch("src.api.main._graph_service", None):
        client = TestClient(app)
        resp = client.get("/api/graph/path/Article%20A/Article%20C")
        assert resp.status_code == 503


def test_graph_visualization_endpoint(client_with_graph):
    """Test /api/graph/visualization/{cluster_id} endpoint."""
    resp = client_with_graph.get("/api/graph/visualization/0")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)


def test_graph_visualization_endpoint_no_service():
    """Test graph visualization endpoint when graph service is unavailable."""
    with patch("src.api.main._graph_service", None):
        client = TestClient(app)
        resp = client.get("/api/graph/visualization/0")
        assert resp.status_code == 503


def test_graph_visualization_endpoint_no_index(client_with_graph):
    """Test graph visualization endpoint when topic index is unavailable."""
    with patch("src.api.main._topic_index", None):
        client = TestClient(app)
        resp = client.get("/api/graph/visualization/0")
        assert resp.status_code == 503


def test_graph_article_endpoint(client_with_graph):
    """Test /api/graph/article/{article_title} endpoint."""
    resp = client_with_graph.get("/api/graph/article/Article%20A")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    
    # Article A should be in nodes
    node_ids = [node["id"] for node in data["nodes"]]
    assert "Article A" in node_ids


def test_graph_article_endpoint_no_service():
    """Test graph article endpoint when graph service is unavailable."""
    with patch("src.api.main._graph_service", None):
        client = TestClient(app)
        resp = client.get("/api/graph/article/Article%20A")
        assert resp.status_code == 503

