"""
Embedding generation using sentence transformers.
"""

import logging
from typing import List, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # sentence-transformers is optional for some tests
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.

        Args:
            model_name: Sentence transformer model name
        """
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it to use EmbeddingGenerator."
            )

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embedding array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            Embedding array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        return embeddings

