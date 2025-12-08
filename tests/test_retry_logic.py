"""
Tests for retry logic with exponential backoff in Wikipedia clients.
"""

import pytest

pytest.importorskip("mwclient")

from unittest.mock import MagicMock, patch
from src.ingestion.wikipedia_client import WikipediaClient, _sleep_with_backoff
from src.ingestion.wikipedia_client_async import AsyncWikipediaClient


def test_sleep_with_backoff():
    """Test that backoff sleep function works correctly."""
    import time
    start = time.time()
    _sleep_with_backoff(0, base_delay=0.01, max_delay=0.1)
    elapsed = time.time() - start
    # Should sleep for approximately base_delay with some jitter
    assert elapsed >= 0.005  # At least some sleep occurred


def test_wikipedia_client_initializes_with_max_retries():
    """Test that WikipediaClient accepts max_retries parameter."""
    client = WikipediaClient(max_retries=5)
    assert client.max_retries == 5


def test_wikipedia_client_retries_on_failure():
    """Test that WikipediaClient.get_article retries on transient errors."""
    client = WikipediaClient(max_retries=3)
    
    # Mock the site to fail then succeed
    mock_page = MagicMock()
    mock_page.exists = True
    mock_page.name = "Test Article"
    mock_page.text.return_value = "Test content"
    mock_page.categories.return_value = []
    mock_page.links.return_value = []
    mock_page.revisions.return_value = []
    
    call_count = 0
    
    def mock_getitem(key):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Network error")
        return mock_page
    
    client.site.pages.__getitem__ = mock_getitem
    
    # Should retry and eventually succeed
    with patch('src.ingestion.wikipedia_client._sleep_with_backoff'):
        result = client.get_article("Test Article", fetch_links=False, fetch_categories=False)
    assert result is not None
    assert result["title"] == "Test Article"
    assert call_count == 2  # Failed once, then succeeded


def test_max_links_limit():
    """Test that max_links parameter limits link fetching."""
    client = WikipediaClient()
    
    # Create a mock page with many links
    mock_page = MagicMock()
    mock_page.exists = True
    mock_page.name = "Test Article"
    mock_page.text.return_value = "Test content"
    mock_page.categories.return_value = []
    mock_page.revisions.return_value = []
    
    # Create 100 links but limit to 50
    many_links = [f"Link{i}" for i in range(100)]
    mock_page.links.return_value = many_links
    
    client.site.pages = MagicMock()
    client.site.pages.__getitem__ = lambda key: mock_page
    
    result = client.get_article("Test Article", fetch_links=True, max_links=50)
    assert result is not None
    assert len(result["links"]) == 50


@pytest.mark.asyncio
async def test_async_client_retry():
    """Test that AsyncWikipediaClient uses retry logic."""
    client = AsyncWikipediaClient(max_workers=2, max_retries=3)
    
    # Mock the site
    mock_page = MagicMock()
    mock_page.exists = True
    mock_page.name = "Test Article"
    mock_page.text.return_value = "Test content"
    mock_page.categories.return_value = []
    mock_page.links.return_value = []
    mock_page.revisions.return_value = []
    
    call_count = 0
    
    def mock_getitem(key):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Network error")
        return mock_page
    
    client.site.pages.__getitem__ = mock_getitem
    
    with patch('asyncio.sleep'):  # Mock sleep to speed up test
        result = await client.get_article("Test Article", fetch_links=False, fetch_categories=False)
    assert result is not None
    assert result["title"] == "Test Article"
    
    # Cleanup
    client.executor.shutdown(wait=True)

