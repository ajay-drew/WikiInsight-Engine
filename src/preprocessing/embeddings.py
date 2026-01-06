"""
Embedding generation using sentence transformers.

Supports NVIDIA GPU (CUDA) acceleration with automatic CPU fallback.
Device selection: auto (default) | cuda | cpu
"""

import logging
import os
from time import perf_counter
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Track import status
_import_error: Optional[str] = None
SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SentenceTransformer = _SentenceTransformer
except (ImportError, OSError) as e:
    _import_error = str(e)
    logger.warning("Failed to import sentence_transformers: %s", e)


def detect_device(preferred: Optional[str] = None) -> str:
    """
    Detect the best available device for embedding generation.
    
    Note: sentence-transformers requires PyTorch, so this checks PyTorch CUDA.
    For clustering, PyTorch/CuPy GPU clustering is available when CuPy is installed.
    
    Args:
        preferred: Preferred device ('auto', 'cuda', 'cpu', or None)
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if preferred == "cpu":
        logger.info("Using CPU (explicitly requested)")
        return "cpu"
    
    # Try to detect CUDA availability via PyTorch (required for sentence-transformers)
    try:
        import torch
        
        if preferred == "cuda":
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("Using CUDA GPU: %s (explicitly requested)", device_name)
                return "cuda"
            else:
                logger.warning("CUDA requested but not available via PyTorch, falling back to CPU")
                logger.warning("Note: Clustering may still use GPU via PyTorch/CuPy if available")
                return "cpu"
        
        # Auto-detect
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("CUDA GPU detected via PyTorch: %s (%.1f GB)", device_name, gpu_memory)
            return "cuda"
        else:
            logger.info("No CUDA GPU detected via PyTorch, using CPU for embeddings")
            logger.info("Note: Clustering may still use GPU via PyTorch/CuPy if available")
            return "cpu"
            
    except ImportError:
        logger.warning("PyTorch not installed, using CPU for embeddings")
        logger.info("Note: Clustering may still use GPU via PyTorch/CuPy if available")
        return "cpu"
    except Exception as e:
        logger.warning("Error detecting GPU via PyTorch: %s, using CPU for embeddings", e)
        logger.info("Note: Clustering may still use GPU via PyTorch/CuPy if available")
        return "cpu"


def get_gpu_info() -> dict:
    """
    Get information about available GPU(s).
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    
    try:
        import torch
        
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            
            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except Exception as e:
        info["error"] = str(e)
    
    return info


class EmbeddingGenerator:
    """
    Generate embeddings using sentence transformers.
    
    Supports NVIDIA GPU acceleration with automatic CPU fallback.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ('auto', 'cuda', 'cpu', or None for auto)
        """
        if SentenceTransformer is None:
            error_msg = (
                "sentence-transformers is not installed or failed to import. "
                "Install it to use EmbeddingGenerator."
            )
            if _import_error:
                error_msg += f" Original error: {_import_error}"
            raise RuntimeError(error_msg)

        # Detect best device
        self.device = detect_device(device)
        self.model_name = model_name
        self.model = None
        
        # Set environment variables for stability
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        
        # Try to load model on preferred device with fallback
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model with automatic fallback."""
        devices_to_try = [self.device]
        
        # Add fallback to CPU if trying GPU
        if self.device == "cuda":
            devices_to_try.append("cpu")
        
        last_error = None
        
        for device in devices_to_try:
            try:
                logger.info("=" * 80)
                logger.info("Loading embedding model '%s' on %s...", self.model_name, device.upper())
                if device == "cuda":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_name = torch.cuda.get_device_name(0)
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            logger.info("  - GPU: %s (%.1f GB)", gpu_name, gpu_memory)
                    except Exception as e:
                        logger.debug("Could not get GPU info: %s", e)
                logger.info("=" * 80)
                
                load_start = perf_counter()
                self.model = SentenceTransformer(self.model_name, device=device)
                load_time = perf_counter() - load_start
                self.device = device
                logger.info("Successfully loaded model on %s in %.2f seconds", device.upper(), load_time)
                logger.info("Model embedding dimension: %d", self.model.get_sentence_embedding_dimension())
                return
                
            except Exception as exc:
                last_error = exc
                error_msg = str(exc)
                error_type = type(exc).__name__
                logger.warning(
                    "Failed to load model on %s (%s): %s",
                    device.upper(), error_type, error_msg
                )
                
                # Check for specific Windows DLL errors
                if "DLL" in error_msg or "WinError" in error_msg:
                    logger.warning(
                        "Windows DLL error detected. Try reinstalling PyTorch: "
                        "pip install torch --index-url https://download.pytorch.org/whl/cu118"
                    )
        
        # All attempts failed
        raise RuntimeError(
            f"Failed to load embedding model on any device. Last error: {last_error}"
        )
    
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
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing (larger for GPU)
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding array
        """
        # Use larger batch size for GPU
        if self.device == "cuda" and batch_size < 64:
            old_batch_size = batch_size
            batch_size = 64
            logger.info("Increased batch size from %d to %d for GPU", old_batch_size, batch_size)
        
        logger.info("Generating embeddings:")
        logger.info("  - Device: %s", self.device.upper())
        logger.info("  - Batch size: %d", batch_size)
        logger.info("  - Total texts: %d", len(texts))
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated(0) / (1024**3)
                    logger.info("  - GPU memory before: %.2f GB", gpu_memory_before)
            except Exception:
                pass
        
        encode_start = perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        encode_time = perf_counter() - encode_start
        
        logger.info("Embedding generation completed:")
        logger.info("  - Time: %.2f seconds", encode_time)
        logger.info("  - Throughput: %.1f texts/second", len(texts) / encode_time if encode_time > 0 else 0)
        logger.info("  - Output shape: %s", embeddings.shape)
        
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated(0) / (1024**3)
                    logger.info("  - GPU memory after: %.2f GB", gpu_memory_after)
            except Exception:
                pass
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return f"EmbeddingGenerator(model='{self.model_name}', device='{self.device}')"
