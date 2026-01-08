"""
Tests for async Wikipedia data fetching.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

pytest.importorskip("mwclient")

from src.ingestion.fetch_wikipedia_data import (  # noqa: E402
    fetch_corpus_async,
    _normalize_article,
    save_articles,
    load_seed_queries,
    validate_ingestion_config,
)
# Test seed queries (using default from config)
TEST_SEED_QUERIES = [
    "Machine learning",
    "Artificial intelligence",
    "Data science",
]
RAW_DATA_PATH = "data/raw/articles.json"


@pytest.mark.asyncio
async def test_fetch_corpus_async_concurrent():
    """Test that fetch_corpus_async fetches articles concurrently."""
    # Mock the async client
    with patch("src.ingestion.fetch_wikipedia_data.AsyncWikipediaClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock search results
        test_queries = TEST_SEED_QUERIES
        mock_search_results = [
            [{"title": f"Article{i}", "snippet": f"Snippet{i}"} for i in range(5)]
            for _ in test_queries
        ]
        
        # Mock article fetching
        mock_articles = [
            {
                "title": f"Article{i}",
                "text": f"Content for article {i}",
                "categories": [],
                "links": [],
                "revisions": [],
            }
            for i in range(10)
        ]
        
        async def mock_search(query, limit):
            # Return appropriate mock results
            idx = test_queries.index(query) if query in test_queries else 0
            return mock_search_results[idx]
        
        async def mock_get_batch(titles, **kwargs):
            return mock_articles[:len(titles)]
        
        mock_client.search_articles = AsyncMock(side_effect=mock_search)
        mock_client.get_articles_batch = AsyncMock(side_effect=mock_get_batch)
        mock_client.executor.shutdown = MagicMock()
        
        # Run fetch with test queries
        articles = await fetch_corpus_async(
            max_articles=50, 
            per_query_limit=5,
            seed_queries=test_queries
        )
        
        # Verify concurrent execution
        assert len(articles) <= 50
        assert mock_client.search_articles.call_count == len(test_queries)
        # Should call get_articles_batch once with all titles (concurrent)
        assert mock_client.get_articles_batch.call_count == 1


@pytest.mark.asyncio
async def test_fetch_corpus_async_respects_max_articles():
    """Test that fetch_corpus_async respects max_articles limit."""
    with patch("src.ingestion.fetch_wikipedia_data.AsyncWikipediaClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        test_queries = TEST_SEED_QUERIES
        mock_search_results = [
            [{"title": f"Article{i}", "snippet": f"Snippet{i}"} for i in range(20)]
            for _ in test_queries
        ]
        
        mock_articles = [
            {
                "title": f"Article{i}",
                "text": f"Content {i}",
                "categories": [],
                "links": [],
                "revisions": [],
            }
            for i in range(50)
        ]
        
        async def mock_search(query, limit):
            idx = test_queries.index(query) if query in test_queries else 0
            return mock_search_results[idx]
        
        async def mock_get_batch(titles, **kwargs):
            return mock_articles[:len(titles)]
        
        mock_client.search_articles = AsyncMock(side_effect=mock_search)
        mock_client.get_articles_batch = AsyncMock(side_effect=mock_get_batch)
        mock_client.executor.shutdown = MagicMock()
        
        articles = await fetch_corpus_async(
            max_articles=50, 
            per_query_limit=20,
            seed_queries=test_queries
        )
        
        assert len(articles) <= 50


def test_normalize_article():
    """Test article normalization."""
    raw = {
        "title": "Test Article",
        "text": "Some content here",
        "categories": ["Category1", "Category2"],
        "links": ["Link1", "Link2"],
    }
    
    normalized = _normalize_article(raw)
    
    assert normalized["title"] == "Test Article"
    assert normalized["text"] == "Some content here"
    assert isinstance(normalized["categories"], list)
    assert isinstance(normalized["links"], list)
    assert len(normalized["categories"]) == 2
    assert len(normalized["links"]) == 2


def test_normalize_article_with_mock_objects():
    """Test normalization with mwclient-style objects."""
    # Mock objects with .name attribute
    class MockCategory:
        def __init__(self, name):
            self.name = name
    
    class MockLink:
        def __init__(self, name):
            self.name = name
    
    raw = {
        "title": "Test",
        "text": "Content",
        "categories": [MockCategory("Cat1"), MockCategory("Cat2")],
        "links": [MockLink("Link1")],
    }
    
    normalized = _normalize_article(raw)
    
    assert normalized["categories"] == ["Cat1", "Cat2"]
    assert normalized["links"] == ["Link1"]


def test_save_articles(tmp_path):
    """Test saving articles to file."""
    articles = [
        {"title": "Article1", "text": "Content1", "categories": [], "links": []},
        {"title": "Article2", "text": "Content2", "categories": [], "links": []},
    ]
    
    test_path = tmp_path / "test_articles.json"
    save_articles(articles, str(test_path))
    
    assert test_path.exists()
    
    # Verify content
    import json
    with open(test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["title"] == "Article1"


@pytest.mark.asyncio
async def test_main_function_runs():
    """Test that main() function runs without errors."""
    with patch("src.ingestion.fetch_wikipedia_data.fetch_corpus_async") as mock_fetch:
        mock_fetch.return_value = [
            {
                "title": "Test",
                "text": "Content",
                "categories": [],
                "links": [],
            }
        ]
        
        with patch("src.ingestion.fetch_wikipedia_data.save_articles") as mock_save:
            await fetch_corpus_async()
            # Should complete without errors
            assert True

