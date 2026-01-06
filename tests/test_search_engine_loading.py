"""
Tests for HybridSearchEngine loading behavior.

Ensures search engine:
- Does NOT load on API startup
- Is cleared when pipeline starts
- Loads only after pipeline completes
"""

import os
import pytest
import shutil
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd
import numpy as np


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings_file(temp_data_dir):
    """Create mock embeddings.parquet file."""
    os.makedirs(os.path.join(temp_data_dir, "data", "features"), exist_ok=True)
    embeddings_path = os.path.join(temp_data_dir, "data", "features", "embeddings.parquet")
    
    # Create mock embeddings DataFrame
    embeddings = np.random.rand(10, 384).astype(np.float32)
    emb_df = pd.DataFrame({
        "embedding": [emb for emb in embeddings]
    })
    emb_df.to_parquet(embeddings_path)
    return embeddings_path


@pytest.fixture
def mock_articles_file(temp_data_dir):
    """Create mock cleaned_articles.parquet file."""
    os.makedirs(os.path.join(temp_data_dir, "data", "processed"), exist_ok=True)
    articles_path = os.path.join(temp_data_dir, "data", "processed", "cleaned_articles.parquet")
    
    # Create mock articles DataFrame
    articles_df = pd.DataFrame({
        "title": [f"Article {i}" for i in range(10)],
        "cleaned_text": [f"Text content for article {i}" for i in range(10)],
    })
    articles_df.to_parquet(articles_path)
    return articles_path


@pytest.fixture
def mock_config_file(temp_data_dir):
    """Create minimal config.yaml file."""
    import yaml
    config_path = os.path.join(temp_data_dir, "config.yaml")
    config = {
        "preprocessing": {
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "batch_size": 64,
                "device": "cpu"
            }
        },
        "search": {
            "bm25": {
                "title_weight": 2.0,
                "body_weight": 1.0,
                "use_nltk_normalization": True
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestSearchEngineLoadingBehavior:
    """Test search engine loading behavior at different stages."""
    
    def test_search_engine_not_loaded_on_startup(self):
        """Verify search engine is NOT loaded when API starts."""
        # Import after setting up environment
        import sys
        from unittest.mock import patch
        
        # Mock the _load_all_data to verify it's NOT called
        with patch('src.api.main._load_all_data') as mock_load:
            # Simulate API startup by calling lifespan startup
            from src.api.main import lifespan, app
            
            # Create async context manager
            async def test_startup():
                async with lifespan(app):
                    # Verify _load_all_data was NOT called during startup
                    mock_load.assert_not_called()
            
            # Run the async function
            import asyncio
            asyncio.run(test_startup())
    
    def test_search_engine_cleared_on_pipeline_start(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify search engine is cleared when pipeline starts."""
        # Set up environment to point to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            # Import API module
            from src.api.main import _search_engine, _clear_all_data, _load_all_data
            
            # First, manually load data (simulating previous pipeline completion)
            # This should be done via reload endpoint, but for test we'll call directly
            with patch('src.preprocessing.embeddings.EmbeddingGenerator') as mock_emb_gen:
                mock_model = MagicMock()
                mock_emb_gen.return_value.model = mock_model
                mock_emb_gen.return_value.encode.return_value = np.random.rand(384)
                
                # Load data
                import asyncio
                asyncio.run(_load_all_data())
                
                # Verify search engine was loaded
                from src.api.main import _search_engine
                assert _search_engine is not None, "Search engine should be loaded after _load_all_data"
            
            # Now simulate pipeline start - clear data
            _clear_all_data()
            
            # Verify search engine is cleared
            from src.api.main import _search_engine
            assert _search_engine is None, "Search engine should be cleared when pipeline starts"
            
        finally:
            os.chdir(original_cwd)
            # Reset module state
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_search_engine_loads_after_pipeline_completes(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify search engine loads after pipeline completes."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from src.api.main import _search_engine, _load_all_data
            
            # Initially, search engine should be None
            assert _search_engine is None, "Search engine should be None before pipeline"
            
            # Simulate pipeline completion by calling reload
            with patch('src.preprocessing.embeddings.EmbeddingGenerator') as mock_emb_gen:
                mock_model = MagicMock()
                mock_emb_gen.return_value.model = mock_model
                mock_emb_gen.return_value.encode.return_value = np.random.rand(384)
                
                # Load data (simulating pipeline completion)
                import asyncio
                asyncio.run(_load_all_data())
                
                # Verify search engine is now loaded
                from src.api.main import _search_engine
                assert _search_engine is not None, "Search engine should be loaded after pipeline completes"
                
        finally:
            os.chdir(original_cwd)
            # Reset module state
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_search_endpoint_503_before_pipeline(self):
        """Verify search endpoint returns 503 before pipeline runs."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        # Ensure search engine is None
        import src.api.main
        src.api.main._search_engine = None
        src.api.main._db_search_engine = None
        
        client = TestClient(app)
        response = client.post("/api/search", json={"query": "test", "top_k": 3})
        
        assert response.status_code == 503, f"Expected 503, got {response.status_code}"
        assert "not available" in response.json()["detail"].lower()
    
    def test_search_endpoint_503_during_pipeline(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify search endpoint returns 503 during pipeline (after clear, before reload)."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from fastapi.testclient import TestClient
            from src.api.main import app, _clear_all_data
            
            # Simulate: pipeline started, data cleared
            _clear_all_data()
            
            # Verify search engine is None
            import src.api.main
            assert src.api.main._search_engine is None
            
            client = TestClient(app)
            response = client.post("/api/search", json={"query": "test", "top_k": 3})
            
            assert response.status_code == 503, f"Expected 503 during pipeline, got {response.status_code}"
            assert "not available" in response.json()["detail"].lower()
            
        finally:
            os.chdir(original_cwd)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_search_engine_stays_none_during_pipeline_execution(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify search engine stays None even if files exist during pipeline execution."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from src.api.main import _search_engine, _clear_all_data
            
            # Simulate: pipeline started, data cleared
            _clear_all_data()
            
            # Verify search engine is None
            assert _search_engine is None, "Search engine should be None after pipeline start"
            
            # Even though files exist, search engine should NOT auto-load during pipeline
            # This simulates the scenario where pipeline is running and files are being created
            import src.api.main
            assert src.api.main._search_engine is None, "Search engine should stay None during pipeline execution"
            assert src.api.main._db_search_engine is None, "Database search engine should stay None during pipeline execution"
            
        finally:
            os.chdir(original_cwd)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_pipeline_start_clears_search_engine(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify pipeline start endpoint clears search engine."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from fastapi.testclient import TestClient
            from src.api.main import app, _load_all_data
            
            # First, load search engine (simulating previous pipeline completion)
            with patch('src.preprocessing.embeddings.EmbeddingGenerator') as mock_emb_gen:
                mock_model = MagicMock()
                mock_emb_gen.return_value.model = mock_model
                mock_emb_gen.return_value.encode.return_value = np.random.rand(384)
                
                import asyncio
                asyncio.run(_load_all_data())
                
                # Verify search engine is loaded
                import src.api.main
                assert src.api.main._search_engine is not None, "Search engine should be loaded"
            
            # Now simulate pipeline start via API endpoint
            client = TestClient(app)
            response = client.post(
                "/api/pipeline/start",
                json={
                    "seed_queries": ["test query 1", "test query 2", "test query 3"],
                    "per_query_limit": 5,
                    "max_articles": 50
                }
            )
            
            # Pipeline start should succeed
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Verify search engine is now cleared
            import src.api.main
            assert src.api.main._search_engine is None, "Search engine should be cleared when pipeline starts"
            assert src.api.main._db_search_engine is None, "Database search engine should be cleared when pipeline starts"
            
        finally:
            os.chdir(original_cwd)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_reload_endpoint_loads_search_engine(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify reload endpoint loads search engine after pipeline completes."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from fastapi.testclient import TestClient
            from src.api.main import app
            
            # Ensure search engine is None (simulating before pipeline completion)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            
            # Call reload endpoint (simulating pipeline completion)
            client = TestClient(app)
            with patch('src.preprocessing.embeddings.EmbeddingGenerator') as mock_emb_gen:
                mock_model = MagicMock()
                mock_emb_gen.return_value.model = mock_model
                mock_emb_gen.return_value.encode.return_value = np.random.rand(384)
                
                response = client.post("/api/pipeline/reload")
            
            # Reload should succeed
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Verify search engine is now loaded
            import src.api.main
            assert src.api.main._search_engine is not None, "Search engine should be loaded after reload endpoint"
            
        finally:
            os.chdir(original_cwd)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_search_endpoint_works_after_reload(self, temp_data_dir, mock_embeddings_file, mock_articles_file, mock_config_file):
        """Verify search endpoint works after reload (pipeline completion)."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            
            from fastapi.testclient import TestClient
            from src.api.main import app
            
            # Ensure search engine is None
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            
            # Load search engine via reload endpoint
            client = TestClient(app)
            with patch('src.preprocessing.embeddings.EmbeddingGenerator') as mock_emb_gen:
                # Create a proper mock model that returns embeddings
                mock_model = MagicMock()
                # Mock encode to return proper shape (batch_size, embedding_dim)
                mock_model.encode = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
                mock_emb_gen.return_value.model = mock_model
                mock_emb_gen.return_value.encode = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
                
                # Reload
                reload_response = client.post("/api/pipeline/reload")
                assert reload_response.status_code == 200
                
                # Verify search engine is loaded
                import src.api.main
                assert src.api.main._search_engine is not None, "Search engine should be loaded after reload"
                
                # Mock the search method to avoid actual embedding computation issues
                # Create a proper SearchResult object
                from src.serving.search_engine import SearchResult
                mock_result = SearchResult(
                    title="Test Article",
                    score=0.9,
                    rank=1
                )
                
                with patch.object(src.api.main._search_engine, 'search') as mock_search:
                    mock_search.return_value = [mock_result]
                    
                    # Now search should work
                    search_response = client.post("/api/search", json={"query": "test", "top_k": 3})
                    assert search_response.status_code == 200, f"Expected 200, got {search_response.status_code}: {search_response.text}"
                    assert "results" in search_response.json()
            
        finally:
            os.chdir(original_cwd)
            import src.api.main
            src.api.main._search_engine = None
            src.api.main._db_search_engine = None
            src.api.main._cleaned_articles_df = None
    
    def test_lifespan_does_not_load_search_engine(self):
        """Verify API lifespan does NOT load search engine on startup."""
        from unittest.mock import patch, MagicMock
        from src.api.main import lifespan, app
        import asyncio
        
        # Mock _load_all_data to ensure it's NOT called
        with patch('src.api.main._load_all_data') as mock_load:
            async def test_lifespan():
                async with lifespan(app):
                    # Verify _load_all_data was NOT called during startup
                    mock_load.assert_not_called()
                    
                    # Verify search engine is None
                    import src.api.main
                    assert src.api.main._search_engine is None, "Search engine should be None on startup"
                    assert src.api.main._db_search_engine is None, "Database search engine should be None on startup"
            
            asyncio.run(test_lifespan())

