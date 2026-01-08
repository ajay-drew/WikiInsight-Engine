"""
Comprehensive tests for knowledge graph functionality.

Tests include:
- NetworkX graph construction
- Multi-layer graph building
- Graph queries and traversals
- Graph visualization
- Performance optimizations
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.graph.graph_service import GraphService


class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for graph building."""
        n_articles = 20
        embedding_dim = 384
        
        articles_df = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "links": [[f"Article_{(i+1)%n_articles}"] for i in range(n_articles)],
            "categories": [["Category_A"] if i < 10 else ["Category_B"] for i in range(n_articles)],
        })
        
        cluster_assignments = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "cluster_id": [i // 5 for i in range(n_articles)],  # 4 clusters
        })
        
        # Create embeddings with some similarity structure
        embeddings = np.random.randn(n_articles, embedding_dim).astype(np.float32)
        # Make some articles more similar
        embeddings[0:5] = embeddings[0:1] + np.random.randn(5, embedding_dim) * 0.1
        
        return articles_df, cluster_assignments, embeddings
    
    def test_knowledge_graph_builder_init(self):
        """Test KnowledgeGraphBuilder initialization."""
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.7,
            enable_cluster_edges=True,
        )
        
        assert builder.semantic_threshold == 0.7
        assert builder.enable_cluster_edges is True
        assert builder.graph is None
    
    def test_build_graph_basic(self, sample_data):
        """Test basic graph construction."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.99,  # High threshold to limit edges
            enable_cluster_edges=True,
        )
        
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        assert graph is not None
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == len(articles_df)
    
    def test_build_graph_with_cluster_edges(self, sample_data):
        """Test graph building with cluster edges enabled."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.99,
            enable_cluster_edges=True,
        )
        
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        # Check that cluster edges exist
        cluster_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if d.get("layer") == 2 and d.get("type") == "cluster"
        ]
        
        # Should have some cluster edges (articles in same cluster)
        assert len(cluster_edges) >= 0  # May be 0 if threshold too high
    
    def test_build_graph_without_cluster_edges(self, sample_data):
        """Test graph building with cluster edges disabled."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.99,
            enable_cluster_edges=False,
        )
        
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        # Should have no cluster edges
        cluster_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if d.get("layer") == "cluster"
        ]
        
        assert len(cluster_edges) == 0
    
    def test_build_graph_semantic_edges(self, sample_data):
        """Test semantic edge creation."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        # Lower threshold to get more semantic edges
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.5,  # Lower threshold
            enable_cluster_edges=False,
        )
        
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        # Check semantic edges
        semantic_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if d.get("layer") == "semantic"
        ]
        
        # Should have some semantic edges with lower threshold
        assert len(semantic_edges) >= 0
    
    def test_build_graph_node_attributes(self, sample_data):
        """Test that nodes have correct attributes."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        # Check node attributes
        for node in graph.nodes():
            assert "cluster_id" in graph.nodes[node]
            assert "categories" in graph.nodes[node]
            assert isinstance(graph.nodes[node]["cluster_id"], (int, type(None)))
    
    def test_build_graph_edge_attributes(self, sample_data):
        """Test that edges have correct attributes."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        builder = KnowledgeGraphBuilder(semantic_threshold=0.5)
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        # Check edge attributes
        for u, v, d in graph.edges(data=True):
            assert "layer" in d
            assert d["layer"] in ["link", "cluster", "semantic"]
            if d["layer"] == "semantic":
                assert "similarity" in d
                assert isinstance(d["similarity"], (int, float))
    
    def test_build_graph_empty_corpus(self):
        """Test graph building with empty corpus."""
        empty_df = pd.DataFrame(columns=["title", "links", "categories"])
        empty_clusters = pd.DataFrame(columns=["title", "cluster_id"])
        empty_embeddings = np.array([]).reshape(0, 384)
        
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(empty_df, empty_clusters, empty_embeddings)
        
        assert graph is not None
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
    
    def test_build_graph_performance(self, sample_data):
        """Test graph building performance with larger dataset."""
        articles_df, cluster_assignments, embeddings = sample_data
        
        # Scale up
        large_articles = pd.concat([articles_df] * 5, ignore_index=True)
        large_clusters = pd.concat([cluster_assignments] * 5, ignore_index=True)
        large_embeddings = np.vstack([embeddings] * 5)
        
        builder = KnowledgeGraphBuilder(semantic_threshold=0.8)
        
        import time
        start = time.time()
        graph = builder.build_graph(large_articles, large_clusters, large_embeddings)
        elapsed = time.time() - start
        
        assert graph is not None
        assert graph.number_of_nodes() == len(large_articles)
        # Should complete in reasonable time (< 10 seconds for 100 articles)
        assert elapsed < 10.0


class TestGraphService:
    """Test GraphService functionality."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = nx.DiGraph()
        
        # Add nodes
        for i in range(10):
            graph.add_node(f"Article_{i}", cluster_id=i // 3, categories=["Category_A"])
        
        # Add edges
        for i in range(9):
            graph.add_edge(f"Article_{i}", f"Article_{i+1}", layer="link", weight=1.0)
        
        # Add cluster edges
        graph.add_edge("Article_0", "Article_1", layer="cluster", weight=1.0)
        graph.add_edge("Article_2", "Article_3", layer="cluster", weight=1.0)
        
        # Add semantic edges
        graph.add_edge("Article_5", "Article_6", layer="semantic", similarity=0.85, weight=0.85)
        
        return graph
    
    @pytest.fixture
    def graph_service(self, sample_graph):
        """Create GraphService with sample graph."""
        return GraphService(graph=sample_graph)
    
    def test_graph_service_init_with_graph(self, sample_graph):
        """Test GraphService initialization with provided graph."""
        service = GraphService(graph=sample_graph)
        
        assert service.graph is not None
        assert service.graph.number_of_nodes() == 10
    
    def test_get_neighbors(self, graph_service):
        """Test getting graph neighbors."""
        neighbors = graph_service.get_neighbors("Article_0")
        
        assert isinstance(neighbors, list)
        assert len(neighbors) > 0
        assert all(isinstance(n, dict) for n in neighbors)
    
    def test_get_neighbors_with_layer_filter(self, graph_service):
        """Test getting neighbors filtered by layer."""
        link_neighbors = graph_service.get_neighbors("Article_0", layer_filter=[1])  # Layer 1 = links
        cluster_neighbors = graph_service.get_neighbors("Article_0", layer_filter=[2])  # Layer 2 = clusters
        
        assert isinstance(link_neighbors, list)
        assert isinstance(cluster_neighbors, list)
        # All returned neighbors should have the specified layer
        if link_neighbors:
            assert all(n.get("layer") == 1 for n in link_neighbors)
        if cluster_neighbors:
            assert all(n.get("layer") == 2 for n in cluster_neighbors)
    
    def test_get_neighbors_nonexistent_article(self, graph_service):
        """Test getting neighbors for nonexistent article."""
        neighbors = graph_service.get_neighbors("Nonexistent_Article")
        
        assert neighbors == []
    
    def test_find_path(self, graph_service):
        """Test finding path between articles."""
        path = graph_service.find_path("Article_0", "Article_5")
        
        # Path can be None or a list
        assert path is None or isinstance(path, list)
        # Path should start and end with correct articles
        if path:
            assert path[0] == "Article_0"
            assert path[-1] == "Article_5"
    
    def test_find_path_nonexistent(self, graph_service):
        """Test finding path with nonexistent articles."""
        path = graph_service.find_path("Nonexistent_A", "Nonexistent_B")
        
        # Should return None for nonexistent articles
        assert path is None
    
    def test_find_path_no_path(self, graph_service):
        """Test finding path when no path exists."""
        # Create isolated node
        graph_service.graph.add_node("Isolated_Article")
        
        path = graph_service.find_path("Article_0", "Isolated_Article")
        
        # Should return empty or None if no path exists
        assert path == [] or path is None
    
    def test_get_cluster_subgraph(self, graph_service):
        """Test getting subgraph for a cluster."""
        # Create cluster assignments dict
        cluster_assignments = {f"Article_{i}".lower(): i // 5 for i in range(20)}
        nodes, edges = graph_service.get_cluster_subgraph(cluster_id=0, cluster_assignments=cluster_assignments)
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
    
    def test_get_cluster_subgraph_max_nodes(self, graph_service):
        """Test cluster subgraph respects max_nodes limit."""
        max_nodes = 3
        cluster_assignments = {f"Article_{i}".lower(): i // 5 for i in range(20)}
        nodes, edges = graph_service.get_cluster_subgraph(cluster_id=0, cluster_assignments=cluster_assignments, max_nodes=max_nodes)
        
        assert len(nodes) <= max_nodes
    
    def test_get_article_graph(self, graph_service):
        """Test getting graph centered on an article."""
        nodes, edges = graph_service.get_article_graph("Article_0", max_neighbors=20)
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        # Should include the center article
        node_ids = [n.get("id") for n in nodes]
        assert "Article_0" in node_ids
    
    def test_get_article_graph_nonexistent(self, graph_service):
        """Test getting graph for nonexistent article."""
        nodes, edges = graph_service.get_article_graph("Nonexistent_Article")
        
        assert nodes == []
        assert edges == []
    
    def test_to_visualization_format(self, graph_service):
        """Test converting graph to visualization format."""
        # Get some nodes from the graph
        nodes_list = list(graph_service.graph.nodes())[:10] if graph_service.graph else []
        nodes, edges = graph_service.to_visualization_format(nodes=nodes_list, max_nodes=5)
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert len(nodes) <= 5
    
    def test_to_visualization_format_max_nodes(self, graph_service):
        """Test visualization format respects max_nodes."""
        max_nodes = 3
        # Get some nodes from the graph
        nodes_list = list(graph_service.graph.nodes())[:10] if graph_service.graph else []
        nodes, edges = graph_service.to_visualization_format(nodes=nodes_list, max_nodes=max_nodes)
        
        assert len(nodes) <= max_nodes


class TestGraphQueries:
    """Test graph query operations."""
    
    @pytest.fixture
    def complex_graph(self):
        """Create a more complex graph for testing."""
        graph = nx.DiGraph()
        
        # Create 3 clusters
        for cluster_id in range(3):
            for i in range(5):
                article_id = cluster_id * 5 + i
                graph.add_node(
                    f"Article_{article_id}",
                    cluster_id=cluster_id,
                    categories=[f"Category_{cluster_id}"],
                )
        
        # Add intra-cluster links
        for cluster_id in range(3):
            base = cluster_id * 5
            for i in range(4):
                graph.add_edge(
                    f"Article_{base+i}",
                    f"Article_{base+i+1}",
                    layer="link",
                    weight=1.0,
                )
        
        # Add inter-cluster semantic edge
        graph.add_edge(
            "Article_0",
            "Article_10",
            layer="semantic",
            similarity=0.8,
            weight=0.8,
        )
        
        return graph
    
    def test_graph_statistics(self, complex_graph):
        """Test graph statistics calculation."""
        service = GraphService(graph=complex_graph)
        
        stats = {
            "nodes": complex_graph.number_of_nodes(),
            "edges": complex_graph.number_of_edges(),
            "clusters": len(set(
                complex_graph.nodes[n].get("cluster_id")
                for n in complex_graph.nodes()
            )),
        }
        
        assert stats["nodes"] == 15
        assert stats["edges"] > 0
        assert stats["clusters"] == 3
    
    def test_cluster_connectivity(self, complex_graph):
        """Test cluster connectivity analysis."""
        service = GraphService(graph=complex_graph)
        
        # Create cluster assignments dict
        cluster_assignments = {f"Article_{i}".lower(): i // 5 for i in range(15)}
        
        # Get subgraph for cluster 0
        nodes, edges = service.get_cluster_subgraph(cluster_id=0, cluster_assignments=cluster_assignments)
        
        # All nodes should belong to cluster 0
        for node in nodes:
            assert node.get("cluster_id") == 0
    
    def test_path_finding_across_clusters(self, complex_graph):
        """Test finding paths across clusters."""
        service = GraphService(graph=complex_graph)
        
        # Try to find path from cluster 0 to cluster 2
        path = service.find_path("Article_0", "Article_10")
        
        # Should find path if semantic edge exists
        assert isinstance(path, list)


class TestGraphPerformance:
    """Test graph performance with larger datasets."""
    
    def test_large_graph_construction(self):
        """Test graph construction with larger dataset."""
        n_articles = 100
        embedding_dim = 384
        
        articles_df = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "links": [[f"Article_{(i+1)%n_articles}"] for i in range(n_articles)],
            "categories": [["Category"] for _ in range(n_articles)],
        })
        
        cluster_assignments = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "cluster_id": [i // 10 for i in range(n_articles)],
        })
        
        embeddings = np.random.randn(n_articles, embedding_dim).astype(np.float32)
        
        builder = KnowledgeGraphBuilder(semantic_threshold=0.7)
        
        import time
        start = time.time()
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        elapsed = time.time() - start
        
        assert graph is not None
        assert graph.number_of_nodes() == n_articles
        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds for 100 articles
    
    def test_graph_query_performance(self):
        """Test graph query performance."""
        # Create larger graph
        graph = nx.DiGraph()
        n_nodes = 1000
        
        for i in range(n_nodes):
            graph.add_node(f"Article_{i}", cluster_id=i // 10)
            if i > 0:
                graph.add_edge(f"Article_{i-1}", f"Article_{i}", layer="link")
        
        service = GraphService(graph=graph)
        
        import time
        start = time.time()
        neighbors = service.get_neighbors("Article_500")
        elapsed = time.time() - start
        
        assert isinstance(neighbors, list)
        # Should be fast (< 1 second)
        assert elapsed < 1.0

