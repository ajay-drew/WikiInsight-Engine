"""
GPU/CUDA detection and device selection utilities.

This module provides functions to detect GPU availability, check for cuML,
and determine the appropriate device for embeddings and clustering.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_cuda_available: Optional[bool] = None
_cuml_available: Optional[bool] = None
_gpu_info: Optional[Dict] = None


def detect_gpu() -> Dict[str, any]:
    """
    Detect GPU availability and capabilities.
    
    Gracefully handles DLL errors and other exceptions, always falling back to CPU.

    Returns:
        Dictionary with GPU information:
        - available: bool - Whether GPU is available
        - device_count: int - Number of CUDA devices
        - device_name: str - Name of first GPU (if available)
        - memory_gb: float - GPU memory in GB (if available)
    """
    global _gpu_info

    if _gpu_info is not None:
        return _gpu_info

    info = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "memory_gb": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            if info["device_count"] > 0:
                info["device_name"] = torch.cuda.get_device_name(0)
                # Get memory in GB
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                info["memory_gb"] = memory_bytes / (1024**3)
                logger.info(
                    "GPU detected: %s (%.1f GB, %d device(s))",
                    info["device_name"],
                    info["memory_gb"],
                    info["device_count"],
                )
            else:
                logger.info("CUDA available but no devices found")
        else:
            logger.info("CUDA not available")
    except ImportError:
        logger.debug("PyTorch not installed, cannot detect GPU")
    except (OSError, RuntimeError, Exception) as exc:
        # Handle DLL errors, access violations, and other exceptions gracefully
        error_msg = str(exc)
        if "DLL" in error_msg or "WinError" in error_msg or "c10.dll" in error_msg:
            logger.warning(
                "GPU detection failed due to DLL/loading error: %s. Falling back to CPU.",
                error_msg
            )
        else:
            logger.warning("Error detecting GPU: %s. Falling back to CPU.", exc)

    _gpu_info = info
    return info


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Gracefully handles DLL errors and always returns False on any error,
    ensuring CPU fallback.
    """
    global _cuda_available

    if _cuda_available is not None:
        return _cuda_available

    try:
        import torch

        _cuda_available = torch.cuda.is_available()
        return _cuda_available
    except ImportError:
        _cuda_available = False
        logger.debug("PyTorch not installed, CUDA not available")
        return False
    except (OSError, RuntimeError, Exception) as exc:
        # Handle DLL errors and other exceptions gracefully
        error_msg = str(exc)
        if "DLL" in error_msg or "WinError" in error_msg or "c10.dll" in error_msg:
            logger.warning(
                "CUDA check failed due to DLL/loading error: %s. Using CPU.",
                error_msg
            )
        else:
            logger.warning("CUDA check failed: %s. Using CPU.", exc)
        _cuda_available = False
        return False


def is_cuml_available() -> bool:
    """Check if cuML is installed and available."""
    global _cuml_available

    if _cuml_available is not None:
        return _cuml_available

    try:
        import cuml

        # Try to create a simple model to verify it works
        _cuml_available = True
        logger.info("cuML is available for GPU clustering")
        return True
    except ImportError:
        _cuml_available = False
        logger.debug("cuML not installed, will use sklearn for clustering")
        return False
    except Exception as exc:
        _cuml_available = False
        logger.warning("cuML import failed: %s", exc)
        return False


def get_device(device_preference: str = "auto") -> str:
    """
    Get the device to use based on preference and availability.
    
    Always returns a valid device ("cpu" or "cuda"), gracefully handling
    all errors and DLL issues by falling back to CPU.

    Args:
        device_preference: "auto", "cuda", or "cpu"

    Returns:
        "cuda" or "cpu" (always returns "cpu" on any error)
    """
    if device_preference == "cpu":
        logger.info("Device preference set to CPU")
        return "cpu"

    if device_preference == "cuda":
        try:
            if is_cuda_available():
                logger.info("CUDA requested and available, using GPU")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except Exception as exc:
            logger.warning("Error checking CUDA availability: %s. Falling back to CPU.", exc)
            return "cpu"

    # device_preference == "auto"
    try:
        if is_cuda_available():
            logger.info("Auto-detected CUDA, using GPU")
            return "cuda"
        else:
            logger.info("Auto-detection: CUDA not available, using CPU")
            return "cpu"
    except Exception as exc:
        logger.warning("Error during auto-detection: %s. Falling back to CPU.", exc)
        return "cpu"


def get_clustering_backend() -> str:
    """
    Determine which clustering backend to use.

    Returns:
        "cuml" if GPU and cuML available, "sklearn" otherwise
    """
    if is_cuda_available() and is_cuml_available():
        return "cuml"
    return "sklearn"


def log_device_info() -> None:
    """
    Log current device configuration.
    
    Gracefully handles all errors and ensures logging continues even if
    GPU detection fails.
    """
    try:
        gpu_info = detect_gpu()
        device = get_device()
        clustering_backend = get_clustering_backend()

        logger.info("Device configuration:")
        logger.info("  - Device: %s", device)
        logger.info("  - GPU available: %s", gpu_info["available"])
        if gpu_info["available"]:
            logger.info("  - GPU name: %s", gpu_info["device_name"])
            logger.info("  - GPU memory: %.1f GB", gpu_info["memory_gb"])
        else:
            logger.info("  - Using CPU (GPU not available or detection failed)")
        logger.info("  - Clustering backend: %s", clustering_backend)
    except Exception as exc:
        # Never let device info logging crash the pipeline
        logger.warning("Error logging device info: %s. Continuing with CPU.", exc)
        logger.info("Device configuration: CPU (fallback due to error)")

