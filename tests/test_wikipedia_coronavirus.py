"""
Test script for fetching coronavirus-related data from Wikipedia using mwclient.

This script tests:
- WikipediaClient initialization
- Searching for coronavirus-related articles
- Fetching specific article content
- Verifying data structure and content
"""

import pytest

# Skip tests if mwclient is not installed
pytest.importorskip("mwclient")

import logging
from typing import Dict, List

from src.ingestion.wikipedia_client import WikipediaClient  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


@pytest.fixture
def wikipedia_client():
    """Fixture providing a WikipediaClient instance."""
    return WikipediaClient(site="en.wikipedia.org")


def test_wikipedia_client_initialization(wikipedia_client):
    """Test that WikipediaClient initializes correctly."""
    assert wikipedia_client is not None
    assert wikipedia_client.site is not None
    logger.info("✓ WikipediaClient initialized successfully")


def test_search_coronavirus_articles(wikipedia_client):
    """Test searching for coronavirus-related articles."""
    query = "coronavirus"
    limit = 10
    
    logger.info(f"Searching for articles matching '{query}' (limit={limit})...")
    results = wikipedia_client.search_articles(query, limit=limit)
    
    assert isinstance(results, list)
    assert len(results) > 0, "Should find at least one coronavirus-related article"
    
    # Verify structure
    for result in results:
        assert "title" in result
        assert "snippet" in result
        assert isinstance(result["title"], str)
        assert len(result["title"]) > 0
    
    logger.info(f"✓ Found {len(results)} articles matching '{query}'")
    for i, result in enumerate(results[:5], 1):
        logger.info(f"  {i}. {result['title']}")


def test_fetch_coronavirus_main_article(wikipedia_client):
    """Test fetching the main 'Coronavirus' article."""
    title = "Coronavirus"
    
    logger.info(f"Fetching article: '{title}'...")
    article = wikipedia_client.get_article(title)
    
    assert article is not None, f"Article '{title}' should exist"
    assert "title" in article
    assert "text" in article
    assert "categories" in article
    assert "links" in article
    
    # Verify content
    assert article["title"] == title
    assert len(article["text"]) > 0, "Article text should not be empty"
    assert len(article["text"]) > 100, "Article should have substantial content"
    
    # Check for coronavirus-related keywords in text
    text_lower = article["text"].lower()
    assert any(keyword in text_lower for keyword in ["virus", "coronavirus", "covid", "disease"]), \
        "Article should contain virus-related content"
    
    logger.info(f"✓ Successfully fetched article '{title}'")
    logger.info(f"  Text length: {len(article['text'])} characters")
    logger.info(f"  Categories: {len(article['categories'])}")
    logger.info(f"  Links: {len(article['links'])}")


def test_fetch_covid19_article(wikipedia_client):
    """Test fetching the 'COVID-19' article."""
    title = "COVID-19"
    
    logger.info(f"Fetching article: '{title}'...")
    article = wikipedia_client.get_article(title)
    
    assert article is not None, f"Article '{title}' should exist"
    assert len(article["text"]) > 0
    
    logger.info(f"✓ Successfully fetched article '{title}'")
    logger.info(f"  Text length: {len(article['text'])} characters")


def test_fetch_multiple_coronavirus_articles(wikipedia_client):
    """Test fetching multiple coronavirus-related articles."""
    titles = [
        "Coronavirus",
        "COVID-19",
        "SARS-CoV-2",
        "Pandemic",
    ]
    
    fetched_articles: List[Dict] = []
    
    for title in titles:
        logger.info(f"Fetching article: '{title}'...")
        article = wikipedia_client.get_article(title)
        
        if article:
            fetched_articles.append(article)
            logger.info(f"  ✓ Fetched '{title}' ({len(article['text'])} chars)")
        else:
            logger.warning(f"  ✗ Article '{title}' not found or failed to fetch")
    
    assert len(fetched_articles) >= 2, "Should fetch at least 2 coronavirus-related articles"
    logger.info(f"✓ Successfully fetched {len(fetched_articles)}/{len(titles)} articles")


def test_article_data_structure(wikipedia_client):
    """Test that article data has the expected structure."""
    title = "Coronavirus"
    article = wikipedia_client.get_article(title)
    
    assert article is not None
    
    # Required fields
    required_fields = ["title", "text", "categories", "links"]
    for field in required_fields:
        assert field in article, f"Article should have '{field}' field"
    
    # Type checks
    assert isinstance(article["title"], str)
    assert isinstance(article["text"], str)
    assert isinstance(article["categories"], list)
    assert isinstance(article["links"], list)
    
    # Content checks
    assert len(article["title"]) > 0
    assert len(article["text"]) > 0
    
    logger.info("✓ Article data structure is valid")
    logger.info(f"  Title: {article['title']}")
    logger.info(f"  Text length: {len(article['text'])}")
    logger.info(f"  Categories count: {len(article['categories'])}")
    logger.info(f"  Links count: {len(article['links'])}")


def test_nonexistent_article_handling(wikipedia_client):
    """Test handling of nonexistent articles."""
    title = "ThisArticleDefinitelyDoesNotExist12345"
    
    logger.info(f"Testing nonexistent article: '{title}'...")
    article = wikipedia_client.get_article(title)
    
    assert article is None, "Nonexistent article should return None"
    logger.info("✓ Nonexistent article handled correctly")


if __name__ == "__main__":
    """
    Run this script directly to test coronavirus data fetching.
    
    Usage:
        python -m pytest tests/test_wikipedia_coronavirus.py -v -s
        OR
        python tests/test_wikipedia_coronavirus.py
    """
    # Setup logging for standalone execution
    setup_logging()
    
    # Create client
    client = WikipediaClient(site="en.wikipedia.org")
    
    print("\n" + "="*70)
    print("Testing Wikipedia Coronavirus Data Fetching")
    print("="*70 + "\n")
    
    # Run tests manually
    try:
        # Test 1: Search
        print("Test 1: Searching for coronavirus articles...")
        results = client.search_articles("coronavirus", limit=5)
        print(f"✓ Found {len(results)} articles")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']}")
        
        # Test 2: Fetch main article
        print("\nTest 2: Fetching 'Coronavirus' article...")
        article = client.get_article("Coronavirus")
        if article:
            print(f"✓ Successfully fetched")
            print(f"  Title: {article['title']}")
            print(f"  Text length: {len(article['text'])} characters")
            print(f"  Categories: {len(article['categories'])}")
            print(f"  Links: {len(article['links'])}")
            print(f"  First 200 chars: {article['text'][:200]}...")
        else:
            print("✗ Failed to fetch article")
        
        # Test 3: Fetch COVID-19 article
        print("\nTest 3: Fetching 'COVID-19' article...")
        covid_article = client.get_article("COVID-19")
        if covid_article:
            print(f"✓ Successfully fetched")
            print(f"  Title: {covid_article['title']}")
            print(f"  Text length: {len(covid_article['text'])} characters")
        else:
            print("✗ Failed to fetch article")
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        raise

