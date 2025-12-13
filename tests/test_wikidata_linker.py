"""
Tests for Wikidata entity linker (src/research/wikidata_linker.py).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.research.wikidata_linker import WikidataLinker


@pytest.fixture
def temp_cache_path():
    """Create a temporary cache path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "wikidata_mappings.parquet")


def test_wikidata_linker_init(temp_cache_path):
    """Test WikidataLinker initialization."""
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    assert linker.cache_path == temp_cache_path
    assert linker.enabled is True
    assert isinstance(linker.cache, dict)


def test_wikidata_linker_init_disabled():
    """Test WikidataLinker initialization with disabled flag."""
    linker = WikidataLinker(enabled=False)
    assert linker.enabled is False
    result = linker.link_entity("Test Article")
    assert result is None


def test_wikidata_linker_cache_loading(temp_cache_path):
    """Test loading cache from disk."""
    # Create a cache file
    cache_df = pd.DataFrame({
        "title": ["article a", "article b"],
        "qid": ["Q1", "Q2"],
    })
    os.makedirs(os.path.dirname(temp_cache_path), exist_ok=True)
    cache_df.to_parquet(temp_cache_path, index=False)
    
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    assert "article a" in linker.cache
    assert linker.cache["article a"] == "Q1"
    assert linker.cache["article b"] == "Q2"


def test_wikidata_linker_cache_saving(temp_cache_path):
    """Test saving cache to disk."""
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    linker.cache["test article"] = "Q123"
    linker._save_cache()
    
    # Load and verify
    if os.path.exists(temp_cache_path):
        df = pd.read_parquet(temp_cache_path)
        assert "test article" in df["title"].values
        assert "Q123" in df["qid"].values


@patch("src.research.wikidata_linker.requests.get")
def test_link_entity_success(mock_get, temp_cache_path):
    """Test successful entity linking."""
    # Mock Wikidata API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search": [
            {
                "id": "Q1234",
                "label": "Test Article",
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    qid = linker.link_entity("Test Article")
    
    assert qid == "Q1234"
    assert "test article" in linker.cache
    assert linker.cache["test article"] == "Q1234"


@patch("src.research.wikidata_linker.requests.get")
def test_link_entity_not_found(mock_get, temp_cache_path):
    """Test entity linking when entity not found."""
    # Mock Wikidata API response with no results
    mock_response = MagicMock()
    mock_response.json.return_value = {"search": []}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    qid = linker.link_entity("NonExistent Article")
    
    assert qid is None
    assert "nonexistent article" in linker.cache
    assert linker.cache["nonexistent article"] is None


@patch("src.research.wikidata_linker.requests.get")
def test_link_entity_api_error(mock_get, temp_cache_path):
    """Test entity linking when API request fails."""
    # Mock API error
    mock_get.side_effect = requests.exceptions.RequestException("API Error")
    
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    qid = linker.link_entity("Test Article")
    
    # Should return None on error
    assert qid is None


@patch("src.research.wikidata_linker.requests.get")
def test_link_entity_uses_cache(mock_get, temp_cache_path):
    """Test that cached results are used without API call."""
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    linker.cache["cached article"] = "Q999"
    
    qid = linker.link_entity("Cached Article")
    
    assert qid == "Q999"
    # Should not make API call
    mock_get.assert_not_called()


def test_get_wikidata_url():
    """Test Wikidata URL generation."""
    linker = WikidataLinker(enabled=False)
    url = linker.get_wikidata_url("Q1234")
    assert url == "https://www.wikidata.org/wiki/Q1234"


@patch("src.research.wikidata_linker.requests.get")
def test_link_batch(mock_get, temp_cache_path):
    """Test batch entity linking."""
    # Mock API responses
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search": [{"id": "Q1", "label": "Article 1"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    results = linker.link_batch(["Article 1", "Article 2"], delay=0.01)
    
    assert isinstance(results, dict)
    assert "Article 1" in results
    # Should have made API calls for both articles
    assert mock_get.call_count == 2


def test_finalize(temp_cache_path):
    """Test finalize method saves cache."""
    linker = WikidataLinker(cache_path=temp_cache_path, enabled=True)
    linker.cache["test"] = "Q123"
    linker.finalize()
    
    # Cache should be saved
    if os.path.exists(temp_cache_path):
        df = pd.read_parquet(temp_cache_path)
        assert "test" in df["title"].values

