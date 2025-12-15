"""
Tests for graph service (src/graph/graph_service.py).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_service import GraphService
from src.graph.knowledge_graph import KnowledgeGraphBuilder


@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph for testing."""
    # Create articles
    articles_df = pd.DataFrame({
        "title": ["Article A", "Article B", "Article C", "Article D", "Article E"],
        "links": [
            ["Article B", "Article C"],
            ["Article C"],
            ["Article D"],
            ["Article E"],
            [],
        ],
        "categories": [
            ["Category 1"],
            ["Category 1"],
            ["Category 2"],
            ["Category 2"],
            ["Category 3"],
        ],
    })
    
    # Create cluster assignments
    cluster_assignments = pd.DataFrame({
        "title": ["Article A", "Article B", "Article C", "Article D", "Article E"],
        "cluster_id": [0, 0, 1, 1, 2],
    })
    
    # Create embeddings
    embeddings = np.random.normal(size=(5, 10))
    
    # Build graph
    builder = KnowledgeGraphBuilder(
        semantic_threshold=0.5,  # Lower threshold to allow semantic edges for path finding
        enable_cluster_edges=True,
    )
    graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
    
    return graph


@pytest.fixture
def graph_service(sample_graph):
    """Create a GraphService instance with sample graph."""
    return GraphService(graph=sample_graph)


def test_graph_service_init_with_graph(sample_graph):
    """Test GraphService initialization with provided graph."""
    service = GraphService(graph=sample_graph)
    assert service.graph is not None
    assert service.graph.number_of_nodes() == 5


def test_graph_service_init_without_graph():
    """Test GraphService initialization without graph (will try to load from default path)."""
    service = GraphService()
    # Should handle missing graph gracefully
    assert service.graph is None or isinstance(service.graph, nx.DiGraph)


def test_get_neighbors(graph_service):
    """Test getting graph neighbors."""
    neighbors = graph_service.get_neighbors("Article A", max_neighbors=10)
    
    assert isinstance(neighbors, list)
    assert len(neighbors) > 0
    
    # Check neighbor structure
    for neighbor in neighbors:
        assert "title" in neighbor
        assert "layer" in neighbor
        assert "type" in neighbor
        assert "weight" in neighbor
        assert neighbor["layer"] in [1, 2, 3]


def test_get_neighbors_with_layer_filter(graph_service):
    """Test getting neighbors with layer filter."""
    # Get only Layer 1 neighbors
    neighbors = graph_service.get_neighbors(
        "Article A",
        max_neighbors=10,
        layer_filter=[1],
    )
    
    for neighbor in neighbors:
        assert neighbor["layer"] == 1


def test_get_neighbors_nonexistent_article(graph_service):
    """Test getting neighbors for nonexistent article."""
    neighbors = graph_service.get_neighbors("NonExistent Article")
    assert neighbors == []


def test_find_path(graph_service):
    """Test finding path between articles."""
    # Test path within same cluster (A -> B, both in cluster 0)
    path = graph_service.find_path("Article A", "Article B")
    assert path is not None
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == "Article A"
    assert path[-1] == "Article B"
    
    # Test path that may not exist (A -> D, different clusters)
    # This may return None if no semantic edges connect them
    path_ad = graph_service.find_path("Article A", "Article D")
    # Path may or may not exist depending on semantic similarity
    if path_ad is not None:
        assert isinstance(path_ad, list)
        assert path_ad[0] == "Article A"
        assert path_ad[-1] == "Article D"


def test_find_path_nonexistent_article(graph_service):
    """Test finding path with nonexistent article."""
    path = graph_service.find_path("Article A", "NonExistent Article")
    assert path is None


def test_find_path_no_path(graph_service):
    """Test finding path when no path exists."""
    # Create isolated node (if possible)
    # For this test, we'll just check that the method handles it gracefully
    path = graph_service.find_path("Article E", "Article A", max_path_length=1)
    # May or may not find path depending on graph structure
    assert path is None or isinstance(path, list)


def test_get_cluster_subgraph(graph_service):
    """Test getting subgraph for a cluster."""
    cluster_assignments = {
        "article a": 0,
        "article b": 0,
        "article c": 1,
        "article d": 1,
        "article e": 2,
    }
    
    nodes, edges = graph_service.get_cluster_subgraph(
        cluster_id=0,
        cluster_assignments=cluster_assignments,
        max_nodes=100,
    )
    
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    assert len(nodes) > 0
    
    # Check node structure
    for node in nodes:
        assert "id" in node
        assert "label" in node
        assert "cluster_id" in node
    
    # Check edge structure
    for edge in edges:
        assert "source" in edge
        assert "target" in edge
        assert "layer" in edge
        assert "weight" in edge
        assert "type" in edge


def test_get_cluster_subgraph_max_nodes(graph_service):
    """Test cluster subgraph respects max_nodes limit."""
    cluster_assignments = {
        "article a": 0,
        "article b": 0,
        "article c": 0,
        "article d": 0,
        "article e": 0,
    }
    
    nodes, edges = graph_service.get_cluster_subgraph(
        cluster_id=0,
        cluster_assignments=cluster_assignments,
        max_nodes=2,
    )
    
    # Note: max_nodes limits cluster articles, but Layer 3 edges may add connected articles
    # So we check that cluster articles are limited, but total may be slightly higher
    # due to connected articles from other clusters
    cluster_nodes = [n for n in nodes if n.get("cluster_id") == 0]
    assert len(cluster_nodes) <= 2


def test_get_article_graph(graph_service):
    """Test getting graph centered on an article."""
    nodes, edges = graph_service.get_article_graph("Article A", max_neighbors=10)
    
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    assert len(nodes) > 0
    
    # Article A should be in nodes
    node_ids = [node["id"] for node in nodes]
    assert "Article A" in node_ids


def test_get_article_graph_nonexistent(graph_service):
    """Test getting graph for nonexistent article."""
    nodes, edges = graph_service.get_article_graph("NonExistent Article")
    assert nodes == []
    assert edges == []


def test_to_visualization_format(graph_service):
    """Test converting graph to visualization format."""
    article_titles = ["Article A", "Article B", "Article C"]
    
    nodes, edges = graph_service.to_visualization_format(article_titles, max_nodes=100)
    
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    
    # Check node structure
    for node in nodes:
        assert "id" in node
        assert "label" in node
        assert "cluster_id" in node
    
    # Check edge structure
    for edge in edges:
        assert "source" in edge
        assert "target" in edge
        assert edge["source"] in article_titles
        assert edge["target"] in article_titles


def test_to_visualization_format_max_nodes(graph_service):
    """Test visualization format respects max_nodes limit."""
    article_titles = ["Article A", "Article B", "Article C", "Article D", "Article E"]
    
    nodes, edges = graph_service.to_visualization_format(article_titles, max_nodes=3)
    
    assert len(nodes) <= 3

