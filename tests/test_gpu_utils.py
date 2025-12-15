"""
Tests for GPU detection utilities.
"""

from unittest.mock import patch

import pytest

from src.common.gpu_utils import (
    detect_gpu,
    get_clustering_backend,
    get_device,
    is_cuda_available,
    is_cuml_available,
)


def test_get_device_cpu():
    """Test that get_device returns 'cpu' when requested."""
    assert get_device("cpu") == "cpu"


def test_get_device_auto_no_cuda():
    """Test auto device selection when CUDA is not available."""
    with patch("src.common.gpu_utils.is_cuda_available", return_value=False):
        assert get_device("auto") == "cpu"


def test_get_device_auto_with_cuda():
    """Test auto device selection when CUDA is available."""
    with patch("src.common.gpu_utils.is_cuda_available", return_value=True):
        assert get_device("auto") == "cuda"


def test_get_device_cuda_fallback():
    """Test that CUDA request falls back to CPU if unavailable."""
    with patch("src.common.gpu_utils.is_cuda_available", return_value=False):
        device = get_device("cuda")
        assert device == "cpu"


def test_get_clustering_backend_sklearn():
    """Test that clustering backend returns sklearn when cuML unavailable."""
    with patch("src.common.gpu_utils.is_cuda_available", return_value=False):
        with patch("src.common.gpu_utils.is_cuml_available", return_value=False):
            assert get_clustering_backend() == "sklearn"


def test_get_clustering_backend_cuml():
    """Test that clustering backend returns cuml when available."""
    with patch("src.common.gpu_utils.is_cuda_available", return_value=True):
        with patch("src.common.gpu_utils.is_cuml_available", return_value=True):
            assert get_clustering_backend() == "cuml"


def test_detect_gpu_no_torch():
    """Test GPU detection when PyTorch is not installed."""
    with patch("builtins.__import__", side_effect=ImportError("No module named torch")):
        gpu_info = detect_gpu()
        assert gpu_info["available"] is False
        assert gpu_info["device_count"] == 0


def test_is_cuda_available_no_torch():
    """Test CUDA check when PyTorch is not installed."""
    with patch("builtins.__import__", side_effect=ImportError("No module named torch")):
        assert is_cuda_available() is False


def test_is_cuml_available_no_cuml():
    """Test cuML check when cuML is not installed."""
    with patch("builtins.__import__", side_effect=ImportError("No module named cuml")):
        assert is_cuml_available() is False

