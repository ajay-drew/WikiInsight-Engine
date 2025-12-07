"""
Tests for preprocessing module.
"""

from src.preprocessing.process_data import clean_articles
from src.preprocessing.text_processor import TextProcessor


def test_text_processor_clean_text():
    """Test text cleaning."""
    processor = TextProcessor()
    text = "This is [[a link]] and {{template}}"
    cleaned = processor.clean_text(text)
    assert "[[a link]]" not in cleaned
    assert "{{template}}" not in cleaned


def test_clean_articles_creates_expected_columns():
    """Ensure clean_articles produces expected columns."""
    articles = [
        {"title": "Example", "text": "Some [[text]]", "categories": ["Cat"], "links": ["Link"]},
    ]
    df = clean_articles(articles)
    assert {"title", "raw_text", "cleaned_text", "categories", "links"}.issubset(df.columns)
    assert len(df) == 1

