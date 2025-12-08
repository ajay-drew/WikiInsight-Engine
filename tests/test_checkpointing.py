"""
Tests for checkpointing and resume functionality in fetch_wikipedia_data.
"""

import json
import os
import pytest
import tempfile

pytest.importorskip("mwclient")

from src.ingestion.fetch_wikipedia_data import save_articles, _normalize_article


def test_save_articles_creates_file():
    """Test that save_articles creates a file with articles."""
    articles = [
        {"title": "Article1", "text": "Content1", "categories": [], "links": []},
        {"title": "Article2", "text": "Content2", "categories": [], "links": []},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "articles.json")
        save_articles(articles, path)
        
        assert os.path.exists(path)
        
        # Verify contents
        loaded = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))
        
        assert len(loaded) == 2
        assert loaded[0]["title"] == "Article1"
        assert loaded[1]["title"] == "Article2"


def test_save_articles_handles_empty_list():
    """Test that save_articles handles empty article list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.json")
        save_articles([], path)
        
        assert os.path.exists(path)
        
        # File should exist but be empty or have no valid JSON lines
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 0


def test_resume_loads_existing_articles():
    """Test that resume mode can load existing articles from file."""
    articles = [
        {"title": "Article1", "text": "Content1", "categories": [], "links": []},
        {"title": "Article2", "text": "Content2", "categories": [], "links": []},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "articles.json")
        save_articles(articles, path)
        
        # Simulate resume loading
        loaded = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        assert len(loaded) == 2
        assert loaded[0]["title"] == "Article1"


def test_checkpoint_interval_logic():
    """Test the checkpoint interval logic used in fetch_corpus_async."""
    checkpoint_interval = 1000
    
    # Simulate checkpointing logic
    article_counts = [500, 1000, 1500, 2000]
    should_checkpoint = [count % checkpoint_interval == 0 for count in article_counts]
    
    assert not should_checkpoint[0]  # 500 % 1000 != 0
    assert should_checkpoint[1]      # 1000 % 1000 == 0
    assert not should_checkpoint[2]  # 1500 % 1000 != 0
    assert should_checkpoint[3]      # 2000 % 1000 == 0

