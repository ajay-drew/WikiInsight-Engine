"""
Embedding generation using sentence transformers.
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np

# Set environment variables early to prevent PyTorch from trying to load CUDA DLLs
# This must be done before importing sentence_transformers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError) as e:  # Handle both ImportError and Windows DLL errors
    # OSError can occur on Windows when torch DLL fails to load
    SentenceTransformer = None  # type: ignore[assignment]
    _import_error = str(e)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize embedding generator.

        Args:
            model_name: Sentence transformer model name
            device: Device to use (always "cpu" - GPU support removed)
        """
        if SentenceTransformer is None:
            error_msg = (
                "sentence-transformers is not installed or failed to import. "
                "Install it to use EmbeddingGenerator."
            )
            if '_import_error' in globals():
                error_msg += f" Original error: {_import_error}"
            raise RuntimeError(error_msg)

        # Always use CPU - GPU support removed
        if device is None:
            device = "cpu"
        # Force CPU regardless of what was passed
        device = "cpu"

        logger.info("Loading embedding model: %s on CPU", model_name)
        
        # Always use CPU - GPU support removed
        # Set environment variables to prevent PyTorch from trying to load CUDA DLLs
        import os
        original_env = {}
        try:
            # Prevent PyTorch from trying to load CUDA DLLs
            original_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
            original_env["PYTORCH_ENABLE_MPS_FALLBACK"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            original_env["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK")
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            try:
                self.model = SentenceTransformer(model_name, device="cpu")
                self.model_name = model_name
                self.device = "cpu"
                logger.info("Successfully loaded model on CPU")
            except (OSError, RuntimeError, Exception) as exc:
                error_msg = str(exc)
                error_type = type(exc).__name__
                logger.error("Failed to load embedding model on CPU: %s (%s)", error_msg, error_type)
                
                # If it's a DLL error, provide a helpful message
                if "DLL" in error_msg or "WinError" in error_msg or "c10.dll" in error_msg:
                    raise RuntimeError(
                        f"Failed to load embedding model due to PyTorch DLL error: {error_msg}. "
                        f"This is a Windows-specific PyTorch installation issue. "
                        f"Try reinstalling PyTorch CPU-only version: pip install torch --index-url https://download.pytorch.org/whl/cpu"
                    ) from exc
                else:
                    raise RuntimeError(f"Failed to load embedding model on CPU: {error_msg}") from exc
        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
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

