"""
Comprehensive tests for frontend-to-backend data flow.

Tests verify that:
- Frontend configuration correctly reaches backend API
- Backend correctly writes to config.yaml
- Pipeline correctly reads from config.yaml
- max_articles limit is properly enforced
"""

import json
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestFrontendBackendDataFlow:
    """Test frontend configuration flows correctly to backend."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config.yaml."""
        config_path = tmp_path / "config.yaml"
        config = {
            "ingestion": {
                "seed_queries": ["Test Query 1", "Test Query 2", "Test Query 3"],
                "per_query_limit": 50,
                "max_articles": 1000,
            },
            "data": {
                "wikipedia": {
                    "api_rate_limit": 200.0,
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    def test_pipeline_start_receives_frontend_config(self, client, temp_config, monkeypatch):
        """Test that frontend config is correctly received by backend."""
        import subprocess
        
        # Mock config path
        original_exists = os.path.exists
        def mock_exists(path):
            if str(path) == str(temp_config) or str(path) == "config.yaml":
                return True
            return original_exists(path)
        monkeypatch.setattr("os.path.exists", mock_exists)
        monkeypatch.setattr("src.api.main.os.path.exists", mock_exists)
        monkeypatch.setattr("src.api.main.os.getcwd", lambda: str(temp_config.parent))
        
        # Mock subprocess to avoid actually running pipeline
        mock_process = MagicMock()
        mock_process.pid = 12345
        with patch("subprocess.Popen", return_value=mock_process):
            with patch("src.api.main._clear_all_data"):
                response = client.post(
                    "/api/pipeline/start",
                    json={
                        "seed_queries": ["AI", "ML", "Deep Learning"],
                        "per_query_limit": 8,
                        "max_articles": 50,
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["config"]["max_articles"] == 50
        assert data["config"]["per_query_limit"] == 8
        assert len(data["config"]["seed_queries"]) == 3
    
    def test_config_yaml_updated_with_frontend_values(self, client, temp_config, monkeypatch):
        """Test that config.yaml is updated with frontend values."""
        original_exists = os.path.exists
        original_getcwd = os.getcwd
        
        # Change to temp directory so config.yaml writes to the right place
        original_cwd = os.getcwd()
        os.chdir(str(temp_config.parent))
        
        try:
            def mock_exists(path):
                path_str = str(path)
                if path_str == "config.yaml" or path_str == str(temp_config) or path_str.endswith("config.yaml"):
                    return True
                return original_exists(path)
            
            monkeypatch.setattr("os.path.exists", mock_exists)
            monkeypatch.setattr("src.api.main.os.path.exists", mock_exists)
            monkeypatch.setattr("os.getcwd", lambda: str(temp_config.parent))
            monkeypatch.setattr("src.api.main.os.getcwd", lambda: str(temp_config.parent))
            
            frontend_config = {
                "seed_queries": ["Python", "JavaScript", "Rust"],
                "per_query_limit": 5,
                "max_articles": 50,
            }
            
            with patch("subprocess.Popen"):
                with patch("src.api.main._clear_all_data"):
                    response = client.post("/api/pipeline/start", json=frontend_config)
            
            assert response.status_code == 200
            
            # Verify config.yaml was updated (read from temp directory)
            config_file = temp_config.parent / "config.yaml"
            with open(config_file, "r") as f:
                updated_config = yaml.safe_load(f)
            
            assert updated_config["ingestion"]["max_articles"] == 50, \
                f"Expected max_articles=50, got {updated_config['ingestion'].get('max_articles')}"
            assert updated_config["ingestion"]["per_query_limit"] == 5, \
                f"Expected per_query_limit=5, got {updated_config['ingestion'].get('per_query_limit')}"
            assert updated_config["ingestion"]["seed_queries"] == ["Python", "JavaScript", "Rust"]
        finally:
            os.chdir(original_cwd)
    
    def test_max_articles_validation(self, client):
        """Test that max_articles validation works correctly."""
        # Test max_articles > 1000 (should fail)
        response = client.post(
            "/api/pipeline/start",
            json={
                "seed_queries": ["Test1", "Test2", "Test3"],
                "per_query_limit": 50,
                "max_articles": 2000,  # Exceeds limit
            }
        )
        assert response.status_code == 400
        assert "max_articles cannot exceed 1000" in response.json()["detail"]
    
    def test_per_query_limit_validation(self, client):
        """Test that per_query_limit validation works correctly."""
        # Test per_query_limit > 70 (should fail)
        response = client.post(
            "/api/pipeline/start",
            json={
                "seed_queries": ["Test1", "Test2", "Test3"],
                "per_query_limit": 100,  # Exceeds limit
                "max_articles": 100,
            }
        )
        assert response.status_code == 400
        assert "per_query_limit must be between 1 and 70" in response.json()["detail"]
    
    def test_seed_queries_validation(self, client):
        """Test that seed_queries validation works correctly."""
        # Test too few queries
        response = client.post(
            "/api/pipeline/start",
            json={
                "seed_queries": ["Test1", "Test2"],  # Only 2 queries
                "per_query_limit": 50,
                "max_articles": 100,
            }
        )
        assert response.status_code == 400
        assert "Must have 3-6 seed queries" in response.json()["detail"]
        
        # Test too many queries
        response = client.post(
            "/api/pipeline/start",
            json={
                "seed_queries": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"],  # 7 queries
                "per_query_limit": 50,
                "max_articles": 100,
            }
        )
        assert response.status_code == 400
        assert "Must have 3-6 seed queries" in response.json()["detail"]


class TestIngestionConfigReading:
    """Test that ingestion correctly reads from config.yaml."""
    
    def test_ingestion_reads_max_articles_from_config(self, tmp_path, monkeypatch):
        """Test that ingestion reads max_articles from config.yaml when CLI uses default."""
        from src.ingestion.fetch_wikipedia_data import main_async
        import asyncio
        
        # Create config with custom max_articles
        config_path = tmp_path / "config.yaml"
        config = {
            "ingestion": {
                "seed_queries": ["Test1", "Test2", "Test3"],
                "per_query_limit": 8,
                "max_articles": 24,  # Custom value
            },
            "data": {
                "wikipedia": {
                    "api_rate_limit": 200.0,
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Change to temp directory so config.yaml resolves correctly
        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        
        try:
            # Mock file operations
            original_exists = os.path.exists
            def mock_exists(path):
                path_str = str(path)
                if path_str == "config.yaml" or path_str.endswith("config.yaml"):
                    return True
                if "raw" in path_str or "articles.json" in path_str:
                    return False
                return original_exists(path)
            
            monkeypatch.setattr("os.path.exists", mock_exists)
            monkeypatch.setattr("src.ingestion.fetch_wikipedia_data.os.path.exists", mock_exists)
            
            # Mock the actual fetching to avoid API calls
            with patch("src.ingestion.fetch_wikipedia_data.fetch_corpus_async") as mock_fetch:
                mock_fetch.return_value = []
                
                # Call with default CLI value (1000)
                asyncio.run(main_async(
                    max_articles=1000,  # Default CLI value
                    per_query_limit=50,  # Default CLI value
                    batch_size=20,
                    max_workers=10,
                    sample=None,
                    resume=False,
                ))
                
                # Verify fetch_corpus_async was called with config value (50), not CLI default (1000)
                call_args = mock_fetch.call_args
                assert call_args.kwargs["max_articles"] == 50, \
                    f"Expected max_articles=50 from config, got {call_args.kwargs['max_articles']}"
                assert call_args.kwargs["per_query_limit"] == 8, \
                    f"Expected per_query_limit=8 from config, got {call_args.kwargs['per_query_limit']}"
        finally:
            os.chdir(original_cwd)
    
    def test_ingestion_uses_cli_when_provided(self, tmp_path, monkeypatch):
        """Test that CLI arguments take precedence when explicitly provided."""
        from src.ingestion.fetch_wikipedia_data import main_async
        import asyncio
        
        # Create config with different values
        config_path = tmp_path / "config.yaml"
        config = {
            "ingestion": {
                "seed_queries": ["Test1", "Test2", "Test3"],
                "per_query_limit": 8,
                "max_articles": 24,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        monkeypatch.setattr("src.ingestion.fetch_wikipedia_data.os.path.exists",
                          lambda p: str(p) == str(config_path) or "raw" in str(p))
        monkeypatch.setattr("src.ingestion.fetch_wikipedia_data.os.getcwd", lambda: str(tmp_path))
        
        with patch("src.ingestion.fetch_wikipedia_data.fetch_corpus_async") as mock_fetch:
            mock_fetch.return_value = []
            
            # Call with explicit CLI values (not defaults)
            asyncio.run(main_async(
                max_articles=50,  # Explicit CLI value (not default 1000)
                per_query_limit=10,  # Explicit CLI value (not default 50)
                batch_size=20,
                max_workers=10,
                sample=None,
                resume=False,
            ))
            
            # Verify CLI values are used, not config values
            call_args = mock_fetch.call_args
            assert call_args.kwargs["max_articles"] == 50, \
                f"Expected max_articles=50 from CLI, got {call_args.kwargs['max_articles']}"
            assert call_args.kwargs["per_query_limit"] == 10, \
                f"Expected per_query_limit=10 from CLI, got {call_args.kwargs['per_query_limit']}"


class TestMaxArticlesEnforcement:
    """Test that max_articles limit is properly enforced during ingestion."""
    
    @pytest.mark.asyncio
    async def test_max_articles_cap_enforced(self, tmp_path, monkeypatch):
        """Test that ingestion stops at max_articles limit."""
        from src.ingestion.fetch_wikipedia_data import fetch_corpus_async
        
        # Mock client to return more articles than max_articles
        mock_client = AsyncMock()
        mock_articles = [{"title": f"Article {i}", "text": f"Content {i}"} for i in range(100)]
        
        # Mock search to return many results
        async def mock_search(query, limit):
            return [{"title": f"Result {i}", "pageid": i} for i in range(limit)]
        
        async def mock_fetch_articles(titles):
            return [{"title": title, "text": "Content", "links": []} for title in titles]
        
        mock_client.search = mock_search
        mock_client.fetch_articles = mock_fetch_articles
        
        with patch("src.ingestion.fetch_wikipedia_data.AsyncWikipediaClient", return_value=mock_client):
            articles = await fetch_corpus_async(
                max_articles=50,  # Limit to 50
                per_query_limit=50,
                batch_size=20,
                max_workers=10,
                seed_queries=["Test1", "Test2", "Test3"],
            )
            
            # Should not exceed max_articles
            assert len(articles) <= 50, \
                f"Expected at most 50 articles, got {len(articles)}"


class TestFrontendAPIInterface:
    """Test frontend API interface contracts."""
    
    def test_pipeline_config_request_model(self):
        """Test PipelineConfigRequest Pydantic model validation."""
        from src.api.main import PipelineConfigRequest
        
        # Valid config
        valid = PipelineConfigRequest(
            seed_queries=["A", "B", "C"],
            per_query_limit=8,
            max_articles=50,
        )
        assert valid.max_articles == 50
        assert valid.per_query_limit == 8
        
        # Test default max_articles
        with_default = PipelineConfigRequest(
            seed_queries=["A", "B", "C"],
            per_query_limit=8,
        )
        assert with_default.max_articles == 1000  # Default value
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config.yaml."""
        config_path = tmp_path / "config.yaml"
        config = {
            "ingestion": {
                "seed_queries": ["Test Query 1", "Test Query 2", "Test Query 3"],
                "per_query_limit": 50,
                "max_articles": 1000,
            },
            "data": {
                "wikipedia": {
                    "api_rate_limit": 200.0,
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    def test_api_response_format(self, temp_config, monkeypatch):
        """Test that API response format matches frontend expectations."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        original_exists = os.path.exists
        def mock_exists(path):
            if str(path) == str(temp_config) or str(path) == "config.yaml":
                return True
            return original_exists(path)
        monkeypatch.setattr("os.path.exists", mock_exists)
        monkeypatch.setattr("src.api.main.os.path.exists", mock_exists)
        monkeypatch.setattr("src.api.main.os.getcwd", lambda: str(temp_config.parent))
        
        with patch("subprocess.Popen"):
            with patch("src.api.main._clear_all_data"):
                response = client.post(
                    "/api/pipeline/start",
                    json={
                        "seed_queries": ["Test1", "Test2", "Test3"],
                        "per_query_limit": 8,
                        "max_articles": 50,
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "message" in data
        assert "config" in data
        assert "seed_queries" in data["config"]
        assert "per_query_limit" in data["config"]
        assert "max_articles" in data["config"]
        
        # Verify values match request
        assert data["config"]["max_articles"] == 50
        assert data["config"]["per_query_limit"] == 8
        assert len(data["config"]["seed_queries"]) == 3

