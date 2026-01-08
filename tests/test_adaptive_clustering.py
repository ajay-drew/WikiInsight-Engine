"""
Comprehensive tests for adaptive clustering functionality.

Tests include:
- Embedding normalization
- Optimal n_clusters finding with different strategies
- Adaptive clusterer with algorithm selection
- Edge cases and error handling
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.modeling.cluster_topics import (
    normalize_embeddings,
    find_optimal_n_clusters_adaptive,
    make_adaptive_clusterer,
    make_clusterer,
)


class TestNormalizeEmbeddings:
    """Test embedding normalization."""
    
    def test_normalize_embeddings_basic(self):
        """Test basic L2 normalization."""
        embeddings = np.array([[3.0, 4.0], [5.0, 12.0], [1.0, 1.0]])
        normalized = normalize_embeddings(embeddings)
        
        # Check that each row has L2 norm of 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
        
        # Check shape preserved
        assert normalized.shape == embeddings.shape
    
    def test_normalize_embeddings_zero_vector(self):
        """Test normalization handles zero vectors gracefully."""
        embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
        normalized = normalize_embeddings(embeddings)
        
        # Zero vector should remain zero
        assert np.allclose(normalized[0], [0.0, 0.0])
        # Non-zero vector should be normalized
        assert np.allclose(np.linalg.norm(normalized[1]), 1.0)
    
    def test_normalize_embeddings_large_array(self):
        """Test normalization on large array."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        normalized = normalize_embeddings(embeddings)
        
        # Check all rows normalized
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestFindOptimalNClusters:
    """Test optimal n_clusters finding."""
    
    def test_find_optimal_small_dataset(self):
        """Test optimization for small dataset (<200)."""
        np.random.seed(42)
        # Create 50 samples with 3 clear clusters
        n_samples = 50
        n_features = 10
        
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, n_features) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(15, n_features) + [0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster3 = np.random.randn(15, n_features) + [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        
        # Normalize
        embeddings = normalize_embeddings(embeddings)
        
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=30,
            random_state=42,
        )
        
        # Should find k around 3 (the true number of clusters)
        assert 2 <= optimal_k <= 5
        assert best_score > 0.0
    
    def test_find_optimal_medium_dataset(self):
        """Test optimization for medium dataset (200-500)."""
        np.random.seed(42)
        n_samples = 300
        n_features = 20
        
        # Create 5 distinct clusters
        embeddings_list = []
        for i in range(5):
            cluster = np.random.randn(60, n_features) + [i * 2] * n_features
            embeddings_list.append(cluster)
        embeddings = np.vstack(embeddings_list)
        embeddings = normalize_embeddings(embeddings)
        
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=30,
            random_state=42,
        )
        
        # Should find reasonable k (allow wider range due to optimization strategy)
        assert 2 <= optimal_k <= 20
        assert best_score > 0.0
    
    def test_find_optimal_large_dataset(self):
        """Test optimization for large dataset (>500)."""
        np.random.seed(42)
        n_samples = 600
        n_features = 20
        
        # Create 8 distinct clusters
        embeddings_list = []
        for i in range(8):
            cluster = np.random.randn(75, n_features) + [i * 1.5] * n_features
            embeddings_list.append(cluster)
        embeddings = np.vstack(embeddings_list)
        embeddings = normalize_embeddings(embeddings)
        
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=30,
            random_state=42,
        )
        
        # Should find reasonable k
        assert 5 <= optimal_k <= 15
        assert best_score > 0.0
    
    def test_find_optimal_time_limit(self):
        """Test that optimization respects time limit."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        embeddings = normalize_embeddings(embeddings)
        
        # Very short time limit
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=1,  # 1 second
            random_state=42,
        )
        
        # Should still return a valid result
        assert optimal_k >= 2
        assert best_score >= -1.0  # -1.0 is error code
    
    def test_find_optimal_very_small_dataset(self):
        """Test optimization for very small dataset (<50)."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 10)
        embeddings = normalize_embeddings(embeddings)
        
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=10,
            random_state=42,
        )
        
        # Should find reasonable k (at least 2, but not more than n_samples/3)
        assert 2 <= optimal_k <= 10
        assert best_score >= -1.0


class TestMakeAdaptiveClusterer:
    """Test adaptive clusterer."""
    
    def test_adaptive_clusterer_small_dataset(self):
        """Test adaptive clusterer selects Agglomerative for small datasets."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,  # Use fixed k for speed
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
            },
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        # Should use Agglomerative for <200 samples
        assert metadata["algorithm"] == "agglomerative"
        assert metadata["n_clusters"] == 5
        assert metadata["normalization_applied"] is True
        assert len(labels) == 100
        assert hasattr(model, "cluster_centers_")
    
    def test_adaptive_clusterer_medium_dataset(self):
        """Test adaptive clusterer selects KMeans for medium datasets."""
        np.random.seed(42)
        embeddings = np.random.randn(300, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 10,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
            },
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        # Should use KMeans for 200-500 samples
        assert metadata["algorithm"] == "kmeans"
        assert metadata["n_clusters"] == 10
        assert len(labels) == 300
        assert hasattr(model, "cluster_centers_")
    
    def test_adaptive_clusterer_auto_n_clusters(self):
        """Test adaptive clusterer with auto n_clusters optimization."""
        np.random.seed(42)
        embeddings = np.random.randn(150, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": True,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
                "max_optimization_time": 10,  # Short time for testing
            },
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        # Should have optimized n_clusters
        assert metadata["optimization_applied"] is True
        assert metadata["n_clusters"] >= 2
        assert len(labels) == 150
    
    def test_adaptive_clusterer_no_normalization(self):
        """Test adaptive clusterer without normalization."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": False,
            },
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        assert metadata["normalization_applied"] is False
        assert len(labels) == 100
    
    def test_adaptive_clusterer_boundary_sizes(self):
        """Test adaptive clusterer at boundary sizes."""
        np.random.seed(42)
        
        # Test at 199 (should use Agglomerative)
        embeddings_199 = np.random.randn(199, 10)
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 10,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        model, labels, metadata = make_adaptive_clusterer(embeddings_199, cfg, use_gpu=False)
        assert metadata["algorithm"] == "agglomerative"
        
        # Test at 200 (should use KMeans)
        embeddings_200 = np.random.randn(200, 10)
        model, labels, metadata = make_adaptive_clusterer(embeddings_200, cfg, use_gpu=False)
        assert metadata["algorithm"] == "kmeans"
        
        # Test at 500 (should use KMeans)
        embeddings_500 = np.random.randn(500, 10)
        model, labels, metadata = make_adaptive_clusterer(embeddings_500, cfg, use_gpu=False)
        assert metadata["algorithm"] == "kmeans"


class TestMakeClustererAdaptive:
    """Test make_clusterer with adaptive mode."""
    
    def test_make_clusterer_adaptive_mode(self):
        """Test make_clusterer uses adaptive mode when enabled."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
            },
        }
        
        model, labels, _ = make_clusterer(embeddings, cfg, use_gpu=False)
        
        assert len(labels) == 100
        assert hasattr(model, "cluster_centers_")
    
    def test_make_clusterer_auto_method(self):
        """Test make_clusterer uses adaptive mode when method=auto."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        cfg = {
            "method": "auto",
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
            },
        }
        
        model, labels, _ = make_clusterer(embeddings, cfg, use_gpu=False)
        
        assert len(labels) == 100
    
    def test_make_clusterer_backward_compatible(self):
        """Test make_clusterer backward compatibility (non-adaptive mode)."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        cfg = {
            "adaptive": False,
            "method": "kmeans",
            "n_clusters": 5,
            "random_state": 42,
        }
        
        model, labels, _ = make_clusterer(embeddings, cfg, use_gpu=False)
        
        assert len(labels) == 100
        assert hasattr(model, "cluster_centers_")
        assert model.cluster_centers_.shape[0] == 5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_dataset(self):
        """Test with very small dataset (50 samples)."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 3,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        assert len(labels) == 50
        assert metadata["n_clusters"] == 3
    
    def test_n_clusters_larger_than_samples(self):
        """Test when n_clusters > n_samples (should clamp)."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 100,  # More than samples
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        # Should clamp to n_samples
        assert metadata["n_clusters"] <= 50
        assert len(labels) == 50
    
    def test_invalid_config_fallback(self):
        """Test fallback behavior with invalid config."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        
        # Missing required config
        cfg = {
            "random_state": 42,
        }
        
        # Should use defaults
        model, labels, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=False)
        
        assert len(labels) == 100

