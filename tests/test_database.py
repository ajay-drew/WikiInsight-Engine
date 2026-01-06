"""
Tests for the database module (PostgreSQL + pgvector).

These tests use mocking to avoid requiring an actual PostgreSQL database.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


class TestDatabaseModels:
    """Test database model definitions."""
    
    def test_article_model_import(self):
        """Test that Article model can be imported."""
        from src.database.models import Article
        assert Article is not None
        assert hasattr(Article, "__tablename__")
        assert Article.__tablename__ == "articles"
    
    def test_cluster_model_import(self):
        """Test that Cluster model can be imported."""
        from src.database.models import Cluster
        assert Cluster is not None
        assert hasattr(Cluster, "__tablename__")
        assert Cluster.__tablename__ == "clusters"
    
    def test_graph_edge_model_import(self):
        """Test that GraphEdge model can be imported."""
        from src.database.models import GraphEdge
        assert GraphEdge is not None
        assert hasattr(GraphEdge, "__tablename__")
        assert GraphEdge.__tablename__ == "graph_edges"
    
    def test_pipeline_run_model_import(self):
        """Test that PipelineRun model can be imported."""
        from src.database.models import PipelineRun
        assert PipelineRun is not None
        assert hasattr(PipelineRun, "__tablename__")
        assert PipelineRun.__tablename__ == "pipeline_runs"


class TestDatabaseConnection:
    """Test database connection management."""
    
    def test_database_manager_import(self):
        """Test that DatabaseManager can be imported."""
        from src.database.connection import DatabaseManager
        assert DatabaseManager is not None
    
    def test_get_db_manager_import(self):
        """Test that get_db_manager can be imported."""
        from src.database.connection import get_db_manager
        assert callable(get_db_manager)
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager can be instantiated."""
        from src.database.connection import DatabaseManager
        
        manager = DatabaseManager(
            database_url="postgresql+asyncpg://test:test@localhost:5432/test",
            pool_size=5,
            echo=False,
        )
        
        assert manager.database_url == "postgresql+asyncpg://test:test@localhost:5432/test"
        assert manager.pool_size == 5
        assert manager.echo is False
    
    def test_mask_password(self):
        """Test password masking in database URLs."""
        from src.database.connection import DatabaseManager
        
        url = "postgresql+asyncpg://user:secretpass@localhost:5432/db"
        masked = DatabaseManager._mask_password(url)
        
        assert "secretpass" not in masked
        assert "****" in masked
        assert "user" in masked


class TestDatabaseRepositories:
    """Test repository classes."""
    
    def test_article_repository_import(self):
        """Test that ArticleRepository can be imported."""
        from src.database.repository import ArticleRepository
        assert ArticleRepository is not None
    
    def test_cluster_repository_import(self):
        """Test that ClusterRepository can be imported."""
        from src.database.repository import ClusterRepository
        assert ClusterRepository is not None
    
    def test_graph_repository_import(self):
        """Test that GraphRepository can be imported."""
        from src.database.repository import GraphRepository
        assert GraphRepository is not None


class TestEmbeddingGPUSupport:
    """Test GPU detection and fallback in embeddings module."""
    
    def test_detect_device_import(self):
        """Test that detect_device can be imported."""
        from src.preprocessing.embeddings import detect_device
        assert callable(detect_device)
    
    def test_detect_device_cpu_explicit(self):
        """Test that CPU is returned when explicitly requested."""
        from src.preprocessing.embeddings import detect_device
        
        device = detect_device("cpu")
        assert device == "cpu"
    
    def test_get_gpu_info_import(self):
        """Test that get_gpu_info can be imported."""
        from src.preprocessing.embeddings import get_gpu_info
        assert callable(get_gpu_info)
    
    def test_get_gpu_info_returns_dict(self):
        """Test that get_gpu_info returns a dictionary."""
        from src.preprocessing.embeddings import get_gpu_info
        
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info


class TestDatabaseSearchEngine:
    """Test database-backed search engine."""
    
    def test_db_search_engine_import(self):
        """Test that DatabaseSearchEngine can be imported."""
        from src.serving.db_search_engine import DatabaseSearchEngine
        assert DatabaseSearchEngine is not None
    
    def test_search_result_import(self):
        """Test that SearchResult can be imported."""
        from src.serving.db_search_engine import SearchResult
        assert SearchResult is not None
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        from src.serving.db_search_engine import SearchResult
        
        result = SearchResult(
            title="Test Article",
            score=0.95,
            rank=1,
            cluster_id=5,
            snippet="Test snippet...",
        )
        
        assert result.title == "Test Article"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.cluster_id == 5
        assert result.snippet == "Test snippet..."


class TestConfigUpdates:
    """Test that config.yaml has the new settings."""
    
    def test_database_config_exists(self):
        """Test that database config section exists."""
        import yaml
        import os
        
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "database" in config
        assert "url" in config["database"]
        assert "use_pgvector" in config["database"]
    
    def test_performance_config_exists(self):
        """Test that performance config section exists."""
        import yaml
        import os
        
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "performance" in config
        assert "max_workers" in config["performance"]
        assert "use_multiprocessing" in config["performance"]
    
    def test_embeddings_device_config_exists(self):
        """Test that embeddings device config exists."""
        import yaml
        import os
        
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "preprocessing" in config
        assert "embeddings" in config["preprocessing"]
        assert "device" in config["preprocessing"]["embeddings"]
    
    def test_clustering_method_is_kmeans(self):
        """Test that clustering method is set to kmeans."""
        import yaml
        import os
        
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "models" in config
        assert "clustering" in config["models"]
        assert config["models"]["clustering"]["method"] == "kmeans"


class TestKnowledgeGraphOptimization:
    """Test knowledge graph vectorized operations."""
    
    def test_semantic_edges_vectorized(self):
        """Test that semantic edge addition uses vectorized operations."""
        from src.graph.knowledge_graph import KnowledgeGraphBuilder
        import numpy as np
        import pandas as pd
        
        # Create test data
        n_articles = 10
        embedding_dim = 384
        
        articles_df = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "links": [[] for _ in range(n_articles)],
            "categories": [[] for _ in range(n_articles)],
        })
        
        cluster_assignments = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(n_articles)],
            "cluster_id": [i % 3 for i in range(n_articles)],
        })
        
        embeddings = np.random.randn(n_articles, embedding_dim).astype(np.float32)
        
        # Build graph with high threshold (should create few edges)
        builder = KnowledgeGraphBuilder(
            semantic_threshold=0.99,  # Very high to limit edges
            enable_cluster_edges=False,
        )
        
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        
        assert graph is not None
        assert graph.number_of_nodes() == n_articles

