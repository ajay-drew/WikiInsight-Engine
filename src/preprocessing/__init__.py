"""
Data preprocessing and feature engineering module.
"""

from .text_processor import TextProcessor
from .embeddings import EmbeddingGenerator
from .nltk_utils import normalize_text, normalize_corpus

__all__ = [
    "TextProcessor",
    "EmbeddingGenerator",
    "normalize_text",
    "normalize_corpus",
]

