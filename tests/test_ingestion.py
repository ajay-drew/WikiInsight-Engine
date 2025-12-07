"""
Tests for data ingestion module.
"""

import pytest

pytest.importorskip("mwclient")

from src.ingestion.fetch_wikipedia_data import _normalize_article  # noqa: E402
from src.ingestion.wikipedia_client import WikipediaClient  # noqa: E402


def test_wikipedia_client_init():
    """Test Wikipedia client initialization."""
    client = WikipediaClient()
    assert client is not None
    assert client.site is not None


def test_normalize_article_schema():
    """Ensure _normalize_article produces the expected minimal schema."""
    raw = {
        "title": "Example",
        "text": "Some text",
        "categories": ["Cat1", "Cat2"],
        "links": ["Link1"],
    }
    norm = _normalize_article(raw)
    assert norm["title"] == "Example"
    assert norm["text"] == "Some text"
    assert isinstance(norm["categories"], list)
    assert isinstance(norm["links"], list)

