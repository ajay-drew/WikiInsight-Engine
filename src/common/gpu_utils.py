"""
GPU/CUDA detection and device selection utilities.

This module provides functions to detect GPU availability and determine
the appropriate device for embeddings and clustering using PyTorch/CuPy.
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# #region agent log
DEBUG_LOG_PATH = r"x:\majorProjects\WikiInsight-Engine\.cursor\debug.log"
def _debug_log(hypothesis_id, location, message, data=None):
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            log_entry = {
                "sessionId": "gpu-debug",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": __import__("time").time() * 1000
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
# #endregion

_cuda_available: Optional[bool] = None
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
    Check if CUDA is available via PyTorch or CuPy.
    
    Checks both PyTorch and CuPy for CUDA availability, as they can work independently.
    Gracefully handles DLL errors and always returns False on any error,
    ensuring CPU fallback.
    """
    global _cuda_available

    if _cuda_available is not None:
        return _cuda_available

    # #region agent log
    _debug_log("A", "gpu_utils.py:94", "is_cuda_available() called", {"cached": _cuda_available is not None})
    # #endregion
    
    # Try PyTorch first
    try:
        # #region agent log
        _debug_log("A", "gpu_utils.py:98", "Attempting PyTorch import")
        # #endregion
        import torch
        # #region agent log
        _debug_log("A", "gpu_utils.py:101", "PyTorch imported, checking CUDA", {"torch_available": torch.cuda.is_available()})
        # #endregion
        if torch.cuda.is_available():
            _cuda_available = True
            logger.info("CUDA available via PyTorch")
            # #region agent log
            _debug_log("A", "gpu_utils.py:106", "CUDA available via PyTorch - RETURNING TRUE")
            # #endregion
            return True
    except ImportError as e:
        # #region agent log
        _debug_log("A", "gpu_utils.py:110", "PyTorch import failed", {"error": str(e)})
        # #endregion
        logger.debug("PyTorch not installed, checking CuPy for CUDA")
    except (OSError, RuntimeError, Exception) as exc:
        error_msg = str(exc)
        # #region agent log
        _debug_log("A", "gpu_utils.py:115", "PyTorch CUDA check exception", {"error": error_msg, "error_type": type(exc).__name__})
        # #endregion
        if "DLL" in error_msg or "WinError" in error_msg or "c10.dll" in error_msg:
            logger.debug("PyTorch CUDA check failed (DLL error), checking CuPy: %s", error_msg)
        else:
            logger.debug("PyTorch CUDA check failed, checking CuPy: %s", exc)
    
    # Try CuPy as fallback (CuPy can work even if PyTorch CUDA doesn't)
    try:
        # #region agent log
        _debug_log("A", "gpu_utils.py:123", "Attempting CuPy import")
        # #endregion
        import cupy as cp
        # #region agent log
        _debug_log("A", "gpu_utils.py:126", "CuPy imported, testing GPU array creation")
        # #endregion
        # Try to create a small array on GPU to verify CUDA works
        test_array = cp.array([1, 2, 3])
        # #region agent log
        _debug_log("A", "gpu_utils.py:130", "CuPy GPU array created successfully - RETURNING TRUE", {"test_array": str(test_array)})
        # #endregion
        _cuda_available = True
        logger.info("CUDA available via CuPy (PyTorch CUDA not available)")
        return True
    except ImportError as e:
        # #region agent log
        _debug_log("A", "gpu_utils.py:136", "CuPy import failed", {"error": str(e)})
        # #endregion
        logger.debug("CuPy not installed")
    except Exception as exc:
        # #region agent log
        _debug_log("A", "gpu_utils.py:140", "CuPy CUDA check failed", {"error": str(exc), "error_type": type(exc).__name__})
        # #endregion
        logger.debug("CuPy CUDA check failed: %s", exc)
    
    # Neither PyTorch nor CuPy CUDA is available
    _cuda_available = False
    # #region agent log
    _debug_log("A", "gpu_utils.py:146", "CUDA not available - RETURNING FALSE")
    # #endregion
    logger.debug("CUDA not available via PyTorch or CuPy")
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
        "pytorch" if GPU and CuPy available, "sklearn" otherwise
        
    Uses PyTorch/CuPy for GPU clustering (Windows compatible).
    """
    # #region agent log
    _debug_log("C", "gpu_utils.py:204", "get_clustering_backend() called")
    # #endregion
    
    # Check if CUDA is available
    cuda_avail = is_cuda_available()
    # #region agent log
    _debug_log("C", "gpu_utils.py:211", "CUDA availability check", {"cuda_available": cuda_avail})
    # #endregion
    
    if cuda_avail:
        # Check if CuPy is available for GPU operations
        try:
            # #region agent log
            _debug_log("C", "gpu_utils.py:220", "Testing CuPy for GPU clustering")
            # #endregion
            import cupy as cp
            
            # Test 1: Create array on GPU
            test = cp.array([1])
            # Synchronize to ensure operation completes
            cp.cuda.Stream.null.synchronize()
            
            # Test 2: Verify it's actually on GPU (not NumPy fallback)
            if not isinstance(test, cp.ndarray):
                logger.debug("CuPy test array is not a CuPy array - might be using CPU fallback")
                return "sklearn"
            
            # Test 3: Perform actual GPU computation
            test_result = cp.sum(test * 2)
            cp.cuda.Stream.null.synchronize()
            result_value = float(cp.asnumpy(test_result))
            if result_value != 2.0:
                logger.debug("CuPy GPU computation test failed - result incorrect")
                return "sklearn"
            
            # Test 4: Verify CUDA device is accessible
            try:
                device_id = cp.cuda.Device().id
                logger.debug("CuPy using CUDA device %d", device_id)
            except Exception as e:
                logger.debug("CuPy cannot access CUDA device: %s", e)
                return "sklearn"
            
            # #region agent log
            _debug_log("C", "gpu_utils.py:224", "CuPy test successful - Returning 'pytorch'")
            # #endregion
            logger.info("CuPy CUDA available - using PyTorch/CuPy GPU backend")
            return "pytorch"
        except ImportError as e:
            # #region agent log
            _debug_log("C", "gpu_utils.py:229", "CuPy import failed", {"error": str(e)})
            # #endregion
            logger.debug("CuPy not installed - using CPU backend")
            return "sklearn"
        except Exception as e:
            # #region agent log
            _debug_log("C", "gpu_utils.py:229", "CuPy test failed", {"error": str(e), "error_type": type(e).__name__})
            # #endregion
            logger.debug("CUDA available but CuPy not working - using CPU backend: %s", e)
            # #region agent log
            _debug_log("C", "gpu_utils.py:232", "Returning 'sklearn' (CuPy test failed)")
            # #endregion
            return "sklearn"
    
    # #region agent log
    _debug_log("C", "gpu_utils.py:236", "Returning 'sklearn' (CUDA not available)")
    # #endregion
    return "sklearn"


def verify_gpu_usage() -> Dict[str, bool]:
    """
    Verify GPU usage by checking all available methods.
    
    Returns:
        Dictionary with verification results:
        - pytorch_cuda: bool - PyTorch CUDA available
        - cupy_cuda: bool - CuPy CUDA available
        - clustering_can_use_gpu: bool - Clustering can use GPU
    """
    results = {
        "pytorch_cuda": False,
        "cupy_cuda": False,
        "clustering_can_use_gpu": False,
    }
    
    try:
        import torch
        results["pytorch_cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    
    try:
        import cupy as cp
        cp.array([1, 2, 3])
        results["cupy_cuda"] = True
    except Exception:
        pass
    
    backend = get_clustering_backend()
    results["clustering_can_use_gpu"] = backend == "pytorch"
    
    return results


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
        gpu_verification = verify_gpu_usage()

        logger.info("=" * 80)
        logger.info("Device Configuration")
        logger.info("=" * 80)
        logger.info("Embedding device: %s", device)
        logger.info("Clustering backend: %s", clustering_backend)
        logger.info("")
        logger.info("GPU Availability:")
        logger.info("  - PyTorch CUDA: %s", "✅ Available" if gpu_verification["pytorch_cuda"] else "❌ Not available")
        logger.info("  - CuPy CUDA: %s", "✅ Available" if gpu_verification["cupy_cuda"] else "❌ Not available")
        logger.info("  - Clustering can use GPU: %s", 
                   "✅ Yes" if gpu_verification["clustering_can_use_gpu"] else "❌ No")
        logger.info("")
        
        if gpu_info["available"]:
            logger.info("GPU Details:")
            logger.info("  - GPU name: %s", gpu_info["device_name"])
            logger.info("  - GPU memory: %.1f GB", gpu_info["memory_gb"])
        else:
            logger.info("Using CPU (GPU not available or detection failed)")
        logger.info("=" * 80)
    except Exception as exc:
        logger.warning("Error logging device info: %s. Continuing with CPU.", exc)
        logger.info("Device configuration: CPU (fallback due to error)")

