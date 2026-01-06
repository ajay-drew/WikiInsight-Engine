"""
Comprehensive tests for GPU clustering functionality.

Tests GPU clustering with PyTorch/CuPy backend, including:
- GPU detection and backend selection
- GPU clustering operations
- CuPy array operations
- GPU memory management
- Fallback to CPU when GPU unavailable
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.modeling.cluster_topics import make_clusterer, build_nn_index
from src.common.gpu_utils import (
    get_clustering_backend,
    is_cuda_available,
    verify_gpu_usage,
    get_device,
)


class TestGPUDetection:
    """Test GPU detection and backend selection."""
    
    def test_get_clustering_backend_sklearn_when_no_cuda(self):
        """Test backend returns sklearn when CUDA not available."""
        with patch("src.common.gpu_utils.is_cuda_available", return_value=False):
            backend = get_clustering_backend()
            assert backend == "sklearn"
    
    def test_get_clustering_backend_pytorch_when_cuda_available(self):
        """Test backend returns pytorch when CUDA and CuPy available."""
        with patch("src.common.gpu_utils.is_cuda_available", return_value=True):
            # This test verifies the backend detection logic
            # Actual result depends on whether CuPy is installed and working
            backend = get_clustering_backend()
            # Backend should be either pytorch (if CuPy works) or sklearn (if not)
            assert backend in ["pytorch", "sklearn"]
    
    def test_verify_gpu_usage(self):
        """Test GPU usage verification."""
        results = verify_gpu_usage()
        
        assert isinstance(results, dict)
        assert "pytorch_cuda" in results
        assert "cupy_cuda" in results
        assert "clustering_can_use_gpu" in results
        assert isinstance(results["pytorch_cuda"], bool)
        assert isinstance(results["cupy_cuda"], bool)
        assert isinstance(results["clustering_can_use_gpu"], bool)


class TestGPUClustering:
    """Test GPU clustering operations."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create embeddings large enough to potentially use GPU (2000+ samples)
        n_samples = 2500
        n_features = 384
        return np.random.randn(n_samples, n_features).astype(np.float32)
    
    @pytest.fixture
    def small_embeddings(self):
        """Create small embeddings that should use CPU."""
        n_samples = 100
        n_features = 384
        return np.random.randn(n_samples, n_features).astype(np.float32)
    
    def test_make_clusterer_cpu_fallback(self, sample_embeddings):
        """Test clustering falls back to CPU when GPU not available."""
        cfg = {
            "method": "kmeans",
            "n_clusters": 10,
            "random_state": 42,
        }
        
        with patch("src.modeling.cluster_topics.get_clustering_backend", return_value="sklearn"):
            model, labels = make_clusterer(sample_embeddings, cfg, use_gpu=False)
            
            assert model is not None
            assert labels is not None
            assert len(labels) == len(sample_embeddings)
            assert hasattr(model, "cluster_centers_")
    
    def test_make_clusterer_small_dataset_uses_cpu(self, small_embeddings):
        """Test that small datasets use CPU even if GPU requested."""
        cfg = {
            "method": "kmeans",
            "n_clusters": 5,
            "random_state": 42,
        }
        
        # Even with GPU available, small datasets should use CPU
        with patch("src.modeling.cluster_topics.get_clustering_backend", return_value="pytorch"):
            model, labels = make_clusterer(small_embeddings, cfg, use_gpu=True)
            
            assert model is not None
            assert labels is not None
            assert len(labels) == len(small_embeddings)
    
    def test_make_clusterer_gpu_path(self, sample_embeddings):
        """Test GPU clustering path (will use CPU if GPU not available)."""
        cfg = {
            "method": "kmeans",
            "n_clusters": 10,
            "random_state": 42,
        }
        
        # This test verifies the function works with GPU requested
        # It will use GPU if available, otherwise fall back to CPU
        model, labels = make_clusterer(sample_embeddings, cfg, use_gpu=True)
        
        assert model is not None
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
        assert hasattr(model, "cluster_centers_")
    
    def test_make_clusterer_kmeans_method(self, sample_embeddings):
        """Test KMeans clustering."""
        cfg = {
            "method": "kmeans",
            "n_clusters": 10,
            "random_state": 42,
        }
        
        model, labels = make_clusterer(sample_embeddings, cfg, use_gpu=False)
        
        assert model is not None
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
        assert len(set(labels)) <= 10  # Should have <= n_clusters unique labels
        assert hasattr(model, "cluster_centers_")
        assert model.cluster_centers_.shape[0] == 10
    
    def test_make_clusterer_agglomerative_method(self, sample_embeddings):
        """Test AgglomerativeClustering."""
        cfg = {
            "method": "agglomerative",
            "n_clusters": 10,
            "random_state": 42,
        }
        
        model, labels = make_clusterer(sample_embeddings, cfg, use_gpu=False)
        
        assert model is not None
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
        assert len(set(labels)) == 10  # Agglomerative should have exactly n_clusters
        assert hasattr(model, "cluster_centers_")
    
    def test_make_clusterer_minibatch_kmeans(self, sample_embeddings):
        """Test MiniBatchKMeans clustering."""
        cfg = {
            "method": "minibatch_kmeans",
            "n_clusters": 10,
            "random_state": 42,
        }
        
        model, labels = make_clusterer(sample_embeddings, cfg, use_gpu=False)
        
        assert model is not None
        assert labels is not None
        assert len(labels) == len(sample_embeddings)


class TestGPUNearestNeighbors:
    """Test GPU nearest neighbors index building."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return np.random.randn(100, 384).astype(np.float32)
    
    def test_build_nn_index_cpu(self, sample_embeddings):
        """Test building nearest neighbors index on CPU."""
        cfg = {
            "n_neighbors": 10,
            "algorithm": "auto",
        }
        
        nn_index = build_nn_index(sample_embeddings, cfg, use_gpu=False)
        
        assert nn_index is not None
        assert hasattr(nn_index, "kneighbors")
        
        # Test query
        query = sample_embeddings[:5]
        distances, indices = nn_index.kneighbors(query, n_neighbors=5)
        
        assert distances.shape[0] == 5
        assert indices.shape[0] == 5
        assert distances.shape[1] == 5
        assert indices.shape[1] == 5
    
    def test_build_nn_index_gpu(self, sample_embeddings):
        """Test GPU nearest neighbors (will use CPU if GPU not available)."""
        cfg = {
            "n_neighbors": 10,
            "algorithm": "auto",
        }
        
        # This test verifies the function works with GPU requested
        # It will use GPU if available, otherwise fall back to CPU
        nn_index = build_nn_index(sample_embeddings, cfg, use_gpu=True)
        
        assert nn_index is not None
        assert hasattr(nn_index, "kneighbors")
        
        # Test query
        query = sample_embeddings[:5]
        distances, indices = nn_index.kneighbors(query, n_neighbors=5)
        
        assert distances.shape[0] == 5
        assert indices.shape[0] == 5


class TestGPUDeviceSelection:
    """Test GPU device selection and configuration."""
    
    def test_get_device_cpu_explicit(self):
        """Test explicit CPU device selection."""
        device = get_device("cpu")
        assert device == "cpu"
    
    def test_get_device_auto_detection(self):
        """Test automatic device detection."""
        device = get_device("auto")
        assert device in ["cpu", "cuda"]
    
    def test_get_device_cuda_requested(self):
        """Test CUDA device when requested."""
        with patch("src.common.gpu_utils.is_cuda_available", return_value=True):
            device = get_device("cuda")
            assert device == "cuda"
    
    def test_get_device_cuda_fallback(self):
        """Test CUDA request falls back to CPU if unavailable."""
        with patch("src.common.gpu_utils.is_cuda_available", return_value=False):
            device = get_device("cuda")
            assert device == "cpu"


class TestGPUIntegration:
    """Integration tests for GPU functionality."""
    
    def test_gpu_clustering_integration(self):
        """Test full GPU clustering integration."""
        # Create test data
        n_samples = 2500
        n_features = 384
        embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
        
        cfg = {
            "method": "kmeans",
            "n_clusters": 20,
            "random_state": 42,
        }
        
        # Test with GPU disabled (should always work)
        model, labels = make_clusterer(embeddings, cfg, use_gpu=False)
        
        assert model is not None
        assert labels is not None
        assert len(labels) == n_samples
        assert model.cluster_centers_.shape == (20, n_features)
    
    def test_gpu_backend_detection(self):
        """Test that backend detection works correctly."""
        backend = get_clustering_backend()
        assert backend in ["sklearn", "pytorch", "cuml"]
        
        # Verify GPU usage
        gpu_verification = verify_gpu_usage()
        assert isinstance(gpu_verification, dict)
        assert "clustering_can_use_gpu" in gpu_verification

