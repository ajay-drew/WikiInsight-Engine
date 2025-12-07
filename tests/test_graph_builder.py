"""
Unit tests for GraphBuilder (citation and similarity graphs).
"""

from src.preprocessing.graph_builder import GraphBuilder


def test_build_citation_graph_basic():
    """GraphBuilder.build_citation_graph should create nodes and edges from links."""
    articles = [
        {"title": "A", "links": ["B", "C"]},
        {"title": "B", "links": ["C"]},
        {"title": "C", "links": []},
    ]

    builder = GraphBuilder()
    graph = builder.build_citation_graph(articles)

    # Nodes
    assert set(graph.nodes()) == {"A", "B", "C"}
    # Directed edges from links
    assert ("A", "B") in graph.edges()
    assert ("A", "C") in graph.edges()
    assert ("B", "C") in graph.edges()
    assert ("C", "A") not in graph.edges()


def test_build_similarity_graph_threshold():
    """GraphBuilder.build_similarity_graph should respect the similarity threshold."""
    embeddings = {
        "A": [1.0, 0.0],
        "B": [0.9, 0.1],  # similar to A
        "C": [0.0, 1.0],  # orthogonal
    }

    builder = GraphBuilder()
    graph = builder.build_similarity_graph(embeddings, threshold=0.8)

    # Nodes should be present even without edges
    assert set(graph.nodes()) == {"A", "B", "C"}

    # A and B are similar enough to connect; C should be isolated at this threshold
    assert ("A", "B") in graph.edges() or ("B", "A") in graph.edges()
    assert "C" in graph.nodes()
    # C should not be connected to A or B with a high threshold
    assert ("A", "C") not in graph.edges()
    assert ("B", "C") not in graph.edges()


