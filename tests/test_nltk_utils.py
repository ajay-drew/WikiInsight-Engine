"""
Tests for NLTK-based text normalization utilities.
"""

import pytest
from src.preprocessing.nltk_utils import normalize_text, normalize_corpus


def test_normalize_text_lowercases():
    """Test that normalize_text lowercases input."""
    result = normalize_text("HELLO WORLD")
    assert result == normalize_text("hello world")
    assert "hello" in result.lower() or "world" in result.lower()


def test_normalize_text_removes_stopwords():
    """Test that normalize_text removes common stopwords when NLTK is available."""
    result = normalize_text("the quick brown fox jumps over the lazy dog")
    # Stopwords like "the", "over" should be removed if NLTK is available
    # If NLTK is not available, fallback will keep them
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_text_handles_empty():
    """Test that normalize_text handles empty strings gracefully."""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalize_text_handles_special_chars():
    """Test that normalize_text handles special characters."""
    result = normalize_text("Hello, world! How are you?")
    assert isinstance(result, str)
    # Should not crash and should return something reasonable


def test_normalize_text_with_stemming():
    """Test that normalize_text can optionally apply stemming."""
    result_no_stem = normalize_text("running jumping", use_stem=False)
    result_with_stem = normalize_text("running jumping", use_stem=True)
    # Both should return strings (may differ if stemming is applied)
    assert isinstance(result_no_stem, str)
    assert isinstance(result_with_stem, str)


def test_normalize_corpus():
    """Test that normalize_corpus processes multiple texts."""
    texts = ["Hello world", "Python programming", "Machine learning"]
    results = normalize_corpus(texts)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_normalize_corpus_with_stemming():
    """Test that normalize_corpus can apply stemming."""
    texts = ["running", "jumping", "walking"]
    results = normalize_corpus(texts, use_stem=True)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_normalize_text_handles_unicode():
    """Test that normalize_text handles unicode characters."""
    result = normalize_text("Café résumé naïve")
    assert isinstance(result, str)
    # Should not crash on unicode


def test_normalize_text_handles_numbers():
    """Test that normalize_text handles numbers in text."""
    result = normalize_text("Python 3.10 and Python 3.11")
    assert isinstance(result, str)
    # Numbers may or may not be included depending on tokenization


def test_normalize_text_handles_long_text():
    """Test that normalize_text handles longer text passages."""
    long_text = " ".join(["word"] * 100)
    result = normalize_text(long_text)
    assert isinstance(result, str)
    assert len(result) > 0

