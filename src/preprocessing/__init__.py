"""
Data preprocessing and feature engineering module.
"""

from .text_processor import TextProcessor
from .nltk_utils import normalize_text, normalize_corpus

# Lazy import of EmbeddingGenerator to avoid torch DLL issues during pytest collection
def __getattr__(name: str):
    """Lazy import for EmbeddingGenerator to defer torch import."""
    if name == "EmbeddingGenerator":
        from .embeddings import EmbeddingGenerator
        return EmbeddingGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TextProcessor",
    "EmbeddingGenerator",
    "normalize_text",
    "normalize_corpus",
]

