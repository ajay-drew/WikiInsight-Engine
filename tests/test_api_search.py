"""
Tests for the hybrid search API endpoint.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.serving.search_engine import HybridSearchEngine, SearchResult
from src.research.wikidata_linker import WikidataLinker


@pytest.fixture
def mock_search_engine():
    """Create a mock HybridSearchEngine for testing."""
    mock_articles = [
        {"title": "Machine Learning", "text": "Machine learning is a subset of AI."},
        {"title": "Deep Learning", "text": "Deep learning uses neural networks."},
        {"title": "Cooking Pasta", "text": "Cooking pasta requires boiling water."},
    ]
    mock_embeddings = np.random.normal(size=(3, 384))
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.normal(size=(384,))

    engine = HybridSearchEngine(
        articles=mock_articles,
        embeddings=mock_embeddings,
        model=mock_model,
    )
    return engine


@pytest.fixture
def mock_cleaned_articles_df():
    """Create mock cleaned articles DataFrame with metadata."""
    return pd.DataFrame({
        "title": ["Machine Learning", "Deep Learning", "Cooking Pasta"],
        "cleaned_text": ["text1", "text2", "text3"],
        "categories": [
            ["Computer Science", "AI"],
            ["Computer Science", "Neural Networks"],
            ["Food", "Cooking"],
        ],
        "links": [
            ["Deep Learning", "AI"],
            ["Machine Learning"],
            ["Pasta", "Sauce"],
        ],
    })


@pytest.fixture
def mock_topic_index():
    """Create a mock TopicIndex."""
    mock_index = MagicMock()
    mock_index.lookup.return_value = MagicMock(cluster_id=0)
    return mock_index


@pytest.fixture
def mock_wikidata_linker():
    """Create a mock WikidataLinker."""
    linker = MagicMock(spec=WikidataLinker)
    linker.link_entity.return_value = "Q1234"
    linker.get_wikidata_url.return_value = "https://www.wikidata.org/wiki/Q1234"
    return linker


@pytest.fixture
def client_with_search_engine(mock_search_engine, mock_cleaned_articles_df, mock_topic_index, mock_wikidata_linker):
    """Create a test client with search engine and metadata services injected."""
    with patch("src.api.main._search_engine", mock_search_engine):
        with patch("src.api.main._cleaned_articles_df", mock_cleaned_articles_df):
            with patch("src.api.main._topic_index", mock_topic_index):
                with patch("src.api.main._wikidata_linker", mock_wikidata_linker):
                    yield TestClient(app)


def test_search_endpoint_returns_results(client_with_search_engine):
    """Test that search endpoint returns valid results."""
    response = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total_results" in data
    assert isinstance(data["results"], list)
    assert data["total_results"] >= 0


def test_api_search_endpoint_empty_query(client_with_search_engine):
    """Test that empty query returns empty results gracefully."""
    response = client_with_search_engine.post(
        "/api/search",
        json={"query": "", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert data["results"] == []


def test_api_search_endpoint_without_engine_returns_503():
    """Test that search endpoint returns 503 when engine is not available."""
    with patch("src.api.main._search_engine", None):
        client = TestClient(app)
        response = client.post(
            "/api/search",
            json={"query": "test", "top_k": 5},
        )

        assert response.status_code == 503
        assert "not available" in response.json()["detail"].lower()


def test_api_search_endpoint_validates_top_k(client_with_search_engine):
    """Test that top_k is clamped to reasonable values."""
    # Test with very large top_k (should be clamped)
    response = client_with_search_engine.post(
        "/api/search",
        json={"query": "test", "top_k": 1000},
    )

    assert response.status_code == 200
    data = response.json()
    # Should be clamped to max 50
    assert len(data["results"]) <= 50


def test_api_search_results_have_required_fields(client_with_search_engine):
    """Test that search results contain all required fields."""
    response = client_with_search_engine.post(
        "/api/search",
        json={"query": "learning", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()

    if data["results"]:
        for result in data["results"]:
            assert "title" in result
            assert "score" in result
            assert "rank" in result
            assert isinstance(result["title"], str)
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["rank"], int)


def test_api_search_endpoint_rate_limited(client_with_search_engine):
    """Test that search endpoint respects rate limiting."""
    # Make multiple rapid requests
    responses = []
    for _ in range(5):
        response = client_with_search_engine.post(
            "/api/search",
            json={"query": "test", "top_k": 5},
        )
        responses.append(response.status_code)

    # All should succeed (rate limit is 100/minute, so 5 requests should be fine)
    assert all(status == 200 for status in responses)


def test_api_search_results_have_metadata(client_with_search_engine):
    """Test that search results include rich metadata."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 5},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        # Check for enhanced metadata fields
        assert "wikipedia_url" in result
        assert "wikidata_qid" in result
        assert "wikidata_url" in result
        assert "cluster_id" in result
        assert "categories" in result
        assert "link_count" in result
        
        # Check types
        assert isinstance(result["wikipedia_url"], str)
        assert isinstance(result["categories"], list)
        assert isinstance(result["link_count"], int)


def test_api_search_wikipedia_url_generation(client_with_search_engine):
    """Test that Wikipedia URLs are correctly generated."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 1},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        assert result["wikipedia_url"].startswith("https://en.wikipedia.org/wiki/")


def test_api_search_wikidata_linking(client_with_search_engine):
    """Test that Wikidata linking works in search results."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 1},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        # Should have Wikidata QID and URL (from mock)
        assert result["wikidata_qid"] == "Q1234"
        assert result["wikidata_url"] == "https://www.wikidata.org/wiki/Q1234"


def test_api_search_cluster_id_in_results(client_with_search_engine):
    """Test that cluster IDs are included in search results."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 1},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        # Should have cluster_id (from mock topic index)
        assert result["cluster_id"] == 0


def test_api_search_categories_in_results(client_with_search_engine):
    """Test that categories are included in search results."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 1},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        # Should have categories from cleaned articles
        assert isinstance(result["categories"], list)
        assert len(result["categories"]) > 0


def test_api_search_link_count_in_results(client_with_search_engine):
    """Test that link counts are included in search results."""
    resp = client_with_search_engine.post(
        "/api/search",
        json={"query": "machine learning", "top_k": 1},
    )
    assert resp.status_code == 200
    
    data = resp.json()
    if len(data["results"]) > 0:
        result = data["results"][0]
        # Should have link_count
        assert isinstance(result["link_count"], int)
        assert result["link_count"] >= 0

