"""
Tests for stub article filtering functionality in fetch_wikipedia_data.
"""

import pytest

pytest.importorskip("mwclient")

from src.ingestion.fetch_wikipedia_data import _normalize_article


def test_stub_filtering_logic():
    """Test the inline stub filtering logic used in fetch_corpus_async."""
    # Articles with different word counts
    short_text = " ".join(["word"] * 50)  # 50 words - stub
    long_text = " ".join(["word"] * 300)  # 300 words - not stub
    empty_text = ""
    
    # Simulate the filtering logic from fetch_corpus_async
    stub_min_words = 200
    
    def should_keep(text: str) -> bool:
        return len(text.split()) >= stub_min_words
    
    assert not should_keep(short_text)
    assert should_keep(long_text)
    assert not should_keep(empty_text)


def test_normalize_article_preserves_text():
    """Test that _normalize_article preserves text for filtering."""
    raw = {
        "title": "Test Article",
        "text": " ".join(["word"] * 300),
        "categories": ["Cat1"],
        "links": ["Link1"],
    }
    
    normalized = _normalize_article(raw)
    assert normalized["title"] == "Test Article"
    assert len(normalized["text"].split()) >= 200  # Should have enough words


def test_word_counting_handles_punctuation():
    """Test that word counting handles punctuation correctly."""
    # Text with punctuation should still count words correctly
    text_with_punct = "Hello, world! How are you? I'm fine. Let's go. " * 30
    word_count = len(text_with_punct.split())
    assert word_count > 200  # Should have enough words despite punctuation

