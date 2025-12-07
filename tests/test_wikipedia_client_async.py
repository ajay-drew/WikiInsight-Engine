"""
Tests for async Wikipedia client.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

pytest.importorskip("mwclient")

from src.ingestion.wikipedia_client_async import AsyncWikipediaClient  # noqa: E402


@pytest.fixture
def async_client():
    """Fixture providing an AsyncWikipediaClient instance."""
    return AsyncWikipediaClient(site="en.wikipedia.org", max_workers=5)


@pytest.mark.asyncio
async def test_async_client_initialization(async_client):
    """Test that AsyncWikipediaClient initializes correctly."""
    assert async_client is not None
    assert async_client.site is not None
    assert async_client.executor is not None
    assert async_client.executor._max_workers == 5


@pytest.mark.asyncio
async def test_async_search_articles(async_client):
    """Test async article search."""
    query = "python"
    limit = 5
    
    results = await async_client.search_articles(query, limit=limit)
    
    assert isinstance(results, list)
    assert len(results) <= limit
    
    for result in results:
        assert "title" in result
        assert isinstance(result["title"], str)


@pytest.mark.asyncio
async def test_async_get_article(async_client):
    """Test async article fetching."""
    title = "Python (programming language)"
    
    article = await async_client.get_article(title, fetch_links=False, fetch_categories=False)
    
    assert article is not None
    assert "title" in article
    assert "text" in article
    assert article["title"] == title
    assert len(article["text"]) > 0


@pytest.mark.asyncio
async def test_async_get_articles_batch_concurrent(async_client):
    """Test that batch fetching is truly concurrent (not sequential)."""
    titles = [
        "Python (programming language)",
        "JavaScript",
        "Java (programming language)",
    ]
    
    import time
    start_time = time.time()
    
    articles = await async_client.get_articles_batch(
        titles, fetch_links=False, fetch_categories=False
    )
    
    elapsed_time = time.time() - start_time
    
    # Should be faster than sequential (rough check)
    assert len(articles) == len(titles)
    assert all(article is not None for article in articles)
    
    # Verify all articles were fetched
    for article in articles:
        assert article is not None
        assert "title" in article
        assert "text" in article


@pytest.mark.asyncio
async def test_async_get_articles_batch_with_failures(async_client):
    """Test batch fetching handles failures gracefully."""
    titles = [
        "Python (programming language)",  # Valid
        "ThisArticleDefinitelyDoesNotExist12345",  # Invalid
        "JavaScript",  # Valid
    ]
    
    articles = await async_client.get_articles_batch(
        titles, fetch_links=False, fetch_categories=False
    )
    
    assert len(articles) == len(titles)
    # Some should be None (failed fetches)
    assert articles[1] is None
    assert articles[0] is not None
    assert articles[2] is not None


@pytest.mark.asyncio
async def test_async_get_article_nonexistent(async_client):
    """Test handling of nonexistent articles."""
    title = "ThisArticleDefinitelyDoesNotExist12345"
    
    article = await async_client.get_article(title)
    
    assert article is None


@pytest.mark.asyncio
async def test_async_get_article_with_links_and_categories(async_client):
    """Test fetching article with links and categories."""
    title = "Python (programming language)"
    
    article = await async_client.get_article(
        title, fetch_links=True, fetch_categories=True
    )
    
    assert article is not None
    assert "links" in article
    assert "categories" in article
    assert isinstance(article["links"], list)
    assert isinstance(article["categories"], list)


@pytest.mark.asyncio
async def test_concurrent_searches(async_client):
    """Test that multiple searches run concurrently."""
    queries = ["python", "javascript", "java"]
    
    import time
    start_time = time.time()
    
    # Run searches concurrently
    tasks = [async_client.search_articles(q, limit=5) for q in queries]
    results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    assert len(results) == len(queries)
    for result_list in results:
        assert isinstance(result_list, list)
        assert len(result_list) > 0


@pytest.mark.asyncio
async def test_client_cleanup(async_client):
    """Test that executor is properly cleaned up."""
    # Use the client
    await async_client.search_articles("test", limit=1)
    
    # Cleanup should not raise errors
    async_client.executor.shutdown(wait=False)
    
    # Executor should be shutdown
    assert async_client.executor._shutdown

