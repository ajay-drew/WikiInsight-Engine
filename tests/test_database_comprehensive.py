"""
Comprehensive tests for PostgreSQL database functionality.

Tests include:
- Database connection and initialization
- CRUD operations for all models
- pgvector similarity search
- Async operations
- Transaction handling
- Error handling
"""

import os
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.models import Article, Cluster, GraphEdge, PipelineRun
from src.database.connection import DatabaseManager
from src.database.repository import (
    ArticleRepository,
    ClusterRepository,
    GraphRepository,
)


class TestDatabaseConnection:
    """Test database connection management."""
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager can be initialized."""
        manager = DatabaseManager(
            database_url="postgresql+asyncpg://test:test@localhost:5432/test",
            pool_size=5,
        )
        
        assert manager.database_url == "postgresql+asyncpg://test:test@localhost:5432/test"
        assert manager.pool_size == 5
        assert manager._initialized is False
    
    def test_database_manager_get_url_from_env(self):
        """Test database URL retrieval from environment."""
        test_url = "postgresql+asyncpg://env:pass@localhost:5432/envdb"
        
        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            manager = DatabaseManager()
            assert manager.database_url == test_url
    
    def test_database_manager_default_url(self):
        """Test default database URL."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                manager = DatabaseManager()
                assert "postgresql+asyncpg://" in manager.database_url
                assert "wikiinsight" in manager.database_url
    
    def test_mask_password(self):
        """Test password masking in URLs."""
        manager = DatabaseManager()
        
        url = "postgresql+asyncpg://user:secret@localhost:5432/db"
        masked = manager._mask_password(url)
        
        assert "secret" not in masked
        assert "****" in masked
        assert "user" in masked
    
    @pytest.mark.asyncio
    async def test_database_health_check_mocked(self):
        """Test database health check."""
        manager = DatabaseManager()
        
        # Mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        manager.session = MagicMock(return_value=mock_session)
        
        health = await manager.health_check()
        assert health is True
    
    @pytest.mark.asyncio
    async def test_database_get_stats_mocked(self):
        """Test getting database statistics."""
        manager = DatabaseManager()
        
        # Mock session and results
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.side_effect = [100, 10, 500]  # articles, clusters, edges
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        manager.session = MagicMock(return_value=mock_session)
        
        stats = await manager.get_stats()
        
        assert stats["articles"] == 100
        assert stats["clusters"] == 10
        assert stats["edges"] == 500
        assert stats["connected"] is True


class TestArticleRepository:
    """Test ArticleRepository CRUD operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = MagicMock()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()
        return session
    
    @pytest.fixture
    def article_repo(self, mock_session):
        """Create ArticleRepository with mocked session."""
        return ArticleRepository(mock_session)
    
    @pytest.fixture
    def sample_article_data(self):
        """Create sample article data."""
        return {
            "title": "Test Article",
            "raw_text": "Raw article text",
            "cleaned_text": "Cleaned article text",
            "word_count": 100,
            "categories": ["Science", "Technology"],
            "links": ["Article1", "Article2"],
            "embedding": np.random.randn(384).tolist(),
        }
    
    @pytest.mark.asyncio
    async def test_create_article(self, article_repo, mock_session, sample_article_data):
        """Test creating a single article."""
        article = await article_repo.create(sample_article_data)
        
        assert article is not None
        assert article.title == "Test Article"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_batch_articles(self, article_repo, mock_session, sample_article_data):
        """Test creating multiple articles."""
        articles_data = [sample_article_data.copy() for _ in range(5)]
        articles_data[0]["title"] = "Article 1"
        articles_data[1]["title"] = "Article 2"
        
        articles = await article_repo.create_batch(articles_data)
        
        assert len(articles) == 5
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, article_repo, mock_session):
        """Test getting article by ID."""
        mock_article = Article(id=1, title="Test")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_article
        
        mock_session.execute.return_value = mock_result
        
        article = await article_repo.get_by_id(1)
        
        assert article is not None
        assert article.id == 1
        assert article.title == "Test"
    
    @pytest.mark.asyncio
    async def test_get_by_title(self, article_repo, mock_session):
        """Test getting article by title."""
        mock_article = Article(id=1, title="Test Article")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_article
        
        mock_session.execute.return_value = mock_result
        
        article = await article_repo.get_by_title("Test Article")
        
        assert article is not None
        assert article.title == "Test Article"
    
    @pytest.mark.asyncio
    async def test_search_similar_articles(self, article_repo, mock_session):
        """Test pgvector similarity search."""
        query_embedding = np.random.randn(384).astype(np.float32)
        
        # Mock search results
        mock_article1 = Article(id=1, title="Article 1")
        mock_article2 = Article(id=2, title="Article 2")
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_article1, mock_article2]
        
        mock_session.execute.return_value = mock_result
        
        results = await article_repo.search_similar(query_embedding, top_k=5)
        
        assert len(results) == 2
        assert results[0].title == "Article 1"
        assert results[1].title == "Article 2"
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count(self, article_repo, mock_session):
        """Test counting articles."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        
        mock_session.execute.return_value = mock_result
        
        count = await article_repo.count()
        
        assert count == 100


class TestClusterRepository:
    """Test ClusterRepository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session
    
    @pytest.fixture
    def cluster_repo(self, mock_session):
        """Create ClusterRepository with mocked session."""
        return ClusterRepository(mock_session)
    
    @pytest.mark.asyncio
    async def test_create_cluster(self, cluster_repo, mock_session):
        """Test creating a cluster."""
        cluster_data = {
            "size": 50,
            "keywords": ["AI", "ML", "Deep Learning"],
            "top_articles": ["Article1", "Article2"],
            "centroid": np.random.randn(384).tolist(),
        }
        
        cluster = await cluster_repo.create(cluster_data)
        
        assert cluster is not None
        assert cluster.size == 50
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, cluster_repo, mock_session):
        """Test getting cluster by ID."""
        mock_cluster = Cluster(id=1, size=50)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cluster
        
        mock_session.execute.return_value = mock_result
        
        cluster = await cluster_repo.get_by_id(1)
        
        assert cluster is not None
        assert cluster.id == 1
        assert cluster.size == 50


class TestGraphRepository:
    """Test GraphRepository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.add = MagicMock()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session
    
    @pytest.fixture
    def graph_repo(self, mock_session):
        """Create GraphRepository with mocked session."""
        return GraphRepository(mock_session)
    
    @pytest.mark.asyncio
    async def test_create_edge(self, graph_repo, mock_session):
        """Test creating a graph edge."""
        edge_data = {
            "source_title": "Article A",
            "target_title": "Article B",
            "layer": "semantic",
            "weight": 0.85,
        }
        
        edge = await graph_repo.create(edge_data)
        
        assert edge is not None
        assert edge.source_title == "Article A"
        assert edge.target_title == "Article B"
        assert edge.layer == "semantic"
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_batch_edges(self, graph_repo, mock_session):
        """Test creating multiple edges."""
        edges_data = [
            {
                "source_title": f"Article {i}",
                "target_title": f"Article {i+1}",
                "layer": "semantic",
                "weight": 0.8,
            }
            for i in range(5)
        ]
        
        edges = await graph_repo.create_batch(edges_data)
        
        assert len(edges) == 5
        mock_session.add_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_neighbors(self, graph_repo, mock_session):
        """Test getting graph neighbors."""
        mock_edge1 = GraphEdge(source_title="Article A", target_title="Article B", layer="semantic")
        mock_edge2 = GraphEdge(source_title="Article A", target_title="Article C", layer="link")
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_edge1, mock_edge2]
        
        mock_session.execute.return_value = mock_result
        
        neighbors = await graph_repo.get_neighbors("Article A")
        
        assert len(neighbors) == 2
        assert all(edge.source_title == "Article A" for edge in neighbors)


class TestDatabaseSearchEngine:
    """Test DatabaseSearchEngine functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        manager = AsyncMock()
        manager.initialize = AsyncMock()
        manager.session = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = MagicMock()
        model.encode = MagicMock(return_value=np.random.randn(384))
        return model
    
    @pytest.mark.asyncio
    async def test_database_search_engine_initialization(self, mock_db_manager, mock_embedding_model):
        """Test DatabaseSearchEngine initialization."""
        from src.serving.db_search_engine import DatabaseSearchEngine
        
        engine = DatabaseSearchEngine(
            db_manager=mock_db_manager,
            embedding_model=mock_embedding_model,
        )
        
        await engine.initialize()
        
        assert engine._initialized is True
        mock_db_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_search_semantic_only(self, mock_db_manager, mock_embedding_model):
        """Test semantic-only search."""
        from src.serving.db_search_engine import DatabaseSearchEngine
        
        # Mock article repository
        mock_article = Article(id=1, title="Test Article", cleaned_text="Test content")
        mock_repo = AsyncMock()
        mock_repo.search_similar = AsyncMock(return_value=[mock_article])
        
        engine = DatabaseSearchEngine(
            db_manager=mock_db_manager,
            embedding_model=mock_embedding_model,
        )
        engine._initialized = True
        
        with patch("src.serving.db_search_engine.ArticleRepository", return_value=mock_repo):
            with patch.object(mock_db_manager, "session", return_value=AsyncMock()):
                results = await engine.search(
                    query="test query",
                    top_k=5,
                    semantic_weight=1.0,
                    keyword_weight=0.0,
                )
                
                assert len(results) > 0
                assert results[0].title == "Test Article"
    
    @pytest.mark.asyncio
    async def test_database_search_hybrid(self, mock_db_manager, mock_embedding_model):
        """Test hybrid search (semantic + keyword)."""
        from src.serving.db_search_engine import DatabaseSearchEngine
        
        # Mock repositories
        mock_article = Article(id=1, title="Test Article", cleaned_text="Test content")
        mock_repo = AsyncMock()
        mock_repo.search_similar = AsyncMock(return_value=[mock_article])
        mock_repo.fulltext_search = AsyncMock(return_value=[mock_article])
        
        engine = DatabaseSearchEngine(
            db_manager=mock_db_manager,
            embedding_model=mock_embedding_model,
        )
        engine._initialized = True
        
        with patch("src.serving.db_search_engine.ArticleRepository", return_value=mock_repo):
            with patch.object(mock_db_manager, "session", return_value=AsyncMock()):
                results = await engine.search(
                    query="test query",
                    top_k=5,
                    semantic_weight=0.5,
                    keyword_weight=0.5,
                )
                
                assert len(results) > 0


class TestDatabaseModels:
    """Test database model definitions."""
    
    def test_article_model_fields(self):
        """Test Article model has all required fields."""
        assert hasattr(Article, "id")
        assert hasattr(Article, "title")
        assert hasattr(Article, "raw_text")
        assert hasattr(Article, "cleaned_text")
        assert hasattr(Article, "embedding")
        assert hasattr(Article, "cluster_id")
    
    def test_cluster_model_fields(self):
        """Test Cluster model has all required fields."""
        assert hasattr(Cluster, "id")
        assert hasattr(Cluster, "size")
        assert hasattr(Cluster, "keywords")
        assert hasattr(Cluster, "centroid")
    
    def test_graph_edge_model_fields(self):
        """Test GraphEdge model has all required fields."""
        assert hasattr(GraphEdge, "id")
        assert hasattr(GraphEdge, "source_title")
        assert hasattr(GraphEdge, "target_title")
        assert hasattr(GraphEdge, "layer")
        assert hasattr(GraphEdge, "weight")
    
    def test_pipeline_run_model_fields(self):
        """Test PipelineRun model has all required fields."""
        assert hasattr(PipelineRun, "id")
        assert hasattr(PipelineRun, "run_id")
        assert hasattr(PipelineRun, "status")
        assert hasattr(PipelineRun, "config_snapshot")


class TestDatabaseTransactions:
    """Test database transaction handling."""
    
    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        """Test session context manager commits on success."""
        manager = DatabaseManager()
        
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        
        manager._session_factory = MagicMock()
        manager._session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        manager._session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        manager._initialized = True
        
        async with manager.session() as session:
            assert session == mock_session
        
        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self):
        """Test session rolls back on error."""
        manager = DatabaseManager()
        
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        
        manager._session_factory = MagicMock()
        manager._session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        manager._session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        manager._initialized = True
        
        try:
            async with manager.session() as session:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called_once()

