"""
Tests for knowledge graph construction (src/graph/knowledge_graph.py).
"""

import os
import tempfile
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.graph.knowledge_graph import KnowledgeGraphBuilder


@pytest.fixture
def sample_articles_df():
    """Create sample articles DataFrame with links and categories."""
    return pd.DataFrame({
        "title": ["Article A", "Article B", "Article C", "Article D"],
        "links": [
            ["Article B", "Article C"],  # A links to B and C
            ["Article C"],  # B links to C
            ["Article D"],  # C links to D
            [],  # D has no links
        ],
        "categories": [
            ["Category 1"],
            ["Category 1", "Category 2"],
            ["Category 2"],
            ["Category 3"],
        ],
    })


@pytest.fixture
def sample_cluster_assignments():
    """Create sample cluster assignments."""
    return pd.DataFrame({
        "title": ["Article A", "Article B", "Article C", "Article D"],
        "cluster_id": [0, 0, 1, 1],  # A and B in cluster 0, C and D in cluster 1
    })


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with some similarity structure."""
    # Create embeddings where A and B are similar, C and D are similar
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.9, 0.1, 0.0])  # Similar to A
    emb_c = np.array([0.0, 0.0, 1.0])
    emb_d = np.array([0.0, 0.1, 0.9])  # Similar to C
    
    return np.array([emb_a, emb_b, emb_c, emb_d])


def test_knowledge_graph_builder_init():
    """Test KnowledgeGraphBuilder initialization."""
    builder = KnowledgeGraphBuilder(
        semantic_threshold=0.7,
        enable_cluster_edges=True,
    )
    assert builder.semantic_threshold == 0.7
    assert builder.enable_cluster_edges is True
    assert builder.graph is None


def test_build_graph_basic(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test basic graph construction."""
    builder = KnowledgeGraphBuilder(
        semantic_threshold=0.8,  # High threshold to avoid semantic edges in this test
        enable_cluster_edges=True,
    )
    
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() > 0
    
    # Check that all articles are nodes
    for title in sample_articles_df["title"]:
        assert title in graph


def test_build_graph_layer2_cluster_edges(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test Layer 2 (cluster relationships) edge construction."""
    builder = KnowledgeGraphBuilder(
        semantic_threshold=1.0,  # Disable semantic edges
        enable_cluster_edges=True,
    )
    
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # Check Layer 2 edges
    layer2_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("layer") == 2
    ]
    
    # Articles in same cluster should be connected
    # Cluster 0: A and B
    assert ("Article A", "Article B") in layer2_edges or ("Article B", "Article A") in layer2_edges
    # Cluster 1: C and D
    assert ("Article C", "Article D") in layer2_edges or ("Article D", "Article C") in layer2_edges


def test_build_graph_layer3_semantic_edges(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test Layer 3 (semantic similarity) edge construction."""
    builder = KnowledgeGraphBuilder(
        semantic_threshold=0.85,  # Low threshold to capture similarity
        enable_cluster_edges=False,  # Disable cluster edges
    )
    
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # Check Layer 3 edges
    layer3_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("layer") == 3
    ]
    
    # A and B should be semantically similar (cosine similarity ~0.9)
    # C and D should be semantically similar (cosine similarity ~0.9)
    assert len(layer3_edges) > 0


def test_build_graph_node_attributes(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test that nodes have correct attributes."""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # Check node attributes
    for title in sample_articles_df["title"]:
        assert "index" in graph.nodes[title]
        assert "cluster_id" in graph.nodes[title]
        
        # Check cluster_id matches assignments
        row = sample_cluster_assignments[sample_cluster_assignments["title"] == title]
        if not row.empty:
            expected_cluster = int(row.iloc[0]["cluster_id"])
            assert graph.nodes[title]["cluster_id"] == expected_cluster


def test_build_graph_edge_attributes(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test that edges have correct attributes."""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # Check edge attributes
    for u, v, data in graph.edges(data=True):
        assert "layer" in data
        assert data["layer"] in [1, 2, 3]
        assert "weight" in data
        assert "type" in data
        assert data["type"] in ["direct", "cluster", "semantic"]


def test_build_graph_filters_missing_links(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test that links to articles not in corpus are filtered out."""
    # Add a link to a non-existent article
    sample_articles_df.loc[0, "links"] = ["Article B", "Article C", "NonExistent Article"]
    
    builder = KnowledgeGraphBuilder(
        semantic_threshold=1.0,
        enable_cluster_edges=False,
    )
    
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # NonExistent Article should not be in graph
    assert "NonExistent Article" not in graph
    
    # Article A should not have edge to NonExistent Article
    assert ("Article A", "NonExistent Article") not in graph.edges()


def test_save_and_load_graph(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test saving and loading graph."""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "test_graph.pkl")
        builder.save_graph(graph_path)
        
        assert os.path.exists(graph_path)
        
        # Load graph
        loaded_graph = KnowledgeGraphBuilder.load_graph(graph_path)
        
        assert isinstance(loaded_graph, nx.DiGraph)
        assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
        assert loaded_graph.number_of_edges() == graph.number_of_edges()
        
        # Check node attributes are preserved
        for node in graph.nodes():
            assert node in loaded_graph
            assert loaded_graph.nodes[node]["cluster_id"] == graph.nodes[node]["cluster_id"]


def test_build_graph_with_cluster_edges_disabled(
    sample_articles_df,
    sample_cluster_assignments,
    sample_embeddings,
):
    """Test graph construction with cluster edges disabled."""
    builder = KnowledgeGraphBuilder(
        semantic_threshold=1.0,
        enable_cluster_edges=False,
    )
    
    graph = builder.build_graph(
        sample_articles_df,
        sample_cluster_assignments,
        sample_embeddings,
    )
    
    # Should have no Layer 2 edges
    layer2_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("layer") == 2
    ]
    assert len(layer2_edges) == 0


def test_build_graph_empty_corpus():
    """Test graph construction with empty corpus."""
    empty_df = pd.DataFrame({"title": [], "links": [], "categories": []})
    empty_clusters = pd.DataFrame({"title": [], "cluster_id": []})
    empty_embeddings = np.array([]).reshape(0, 3)
    
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(empty_df, empty_clusters, empty_embeddings)
    
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0

