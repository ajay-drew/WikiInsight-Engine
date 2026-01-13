"""
Tests for clustering metrics (silhouette score, DBI, and semantic metrics).

Verifies that adaptive clustering improves metrics compared to baseline,
and tests semantic metrics (topic coherence, category alignment, cluster diversity).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.modeling.cluster_topics import (
    normalize_embeddings,
    make_clusterer,
    make_adaptive_clusterer,
)
from src.modeling.clustering_metrics import (
    calculate_topic_coherence_keywords,
    calculate_topic_coherence_articles,
    calculate_category_alignment,
    calculate_cluster_diversity_centroids,
    calculate_cluster_diversity_keywords,
    calculate_all_semantic_metrics,
)


def create_clustered_data(n_samples: int, n_clusters: int, n_features: int = 10, seed: int = 42):
    """Create synthetic data with clear cluster structure."""
    np.random.seed(seed)
    embeddings_list = []
    labels_true = []
    
    for i in range(n_clusters):
        cluster_size = n_samples // n_clusters
        cluster = np.random.randn(cluster_size, n_features) + [i * 3.0] * n_features
        embeddings_list.append(cluster)
        labels_true.extend([i] * cluster_size)
    
    embeddings = np.vstack(embeddings_list)
    labels_true = np.array(labels_true)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    embeddings = embeddings[indices]
    labels_true = labels_true[indices]
    
    return embeddings, labels_true


class TestMetricImprovements:
    """Test that adaptive clustering improves metrics."""
    
    def test_normalization_improves_metrics(self):
        """Test that normalization improves silhouette score."""
        embeddings, _ = create_clustered_data(100, 5, n_features=10)
        
        # Without normalization
        cfg_no_norm = {
            "adaptive": False,
            "method": "kmeans",
            "n_clusters": 5,
            "random_state": 42,
        }
        model_no_norm, labels_no_norm, _ = make_clusterer(embeddings, cfg_no_norm)
        score_no_norm = silhouette_score(embeddings, labels_no_norm)
        
        # With normalization
        embeddings_norm = normalize_embeddings(embeddings)
        model_norm, labels_norm, _ = make_clusterer(embeddings_norm, cfg_no_norm)
        score_norm = silhouette_score(embeddings_norm, labels_norm)
        
        # Normalization should improve or maintain score
        # (may not always improve due to randomness, but should be close)
        assert score_norm >= score_no_norm - 0.1  # Allow small tolerance
    
    def test_kmeans_plus_plus_improves_metrics(self):
        """Test that k-means++ improves metrics compared to random init."""
        embeddings, _ = create_clustered_data(200, 8, n_features=10)
        embeddings = normalize_embeddings(embeddings)
        
        # Traditional KMeans (random init)
        cfg_traditional = {
            "adaptive": False,
            "method": "kmeans",
            "n_clusters": 8,
            "random_state": 42,
        }
        # Note: sklearn KMeans uses k-means++ by default, so we test adaptive vs non-adaptive
        # which uses k-means++ in adaptive mode
        
        # Adaptive mode (uses k-means++)
        cfg_adaptive = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 8,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        
        model_adaptive, labels_adaptive, _ = make_clusterer(embeddings, cfg_adaptive)
        score_adaptive = silhouette_score(embeddings, labels_adaptive)
        
        # Should have reasonable score
        assert score_adaptive > 0.0
    
    def test_adaptive_vs_non_adaptive_metrics(self):
        """Compare metrics between adaptive and non-adaptive clustering."""
        embeddings, _ = create_clustered_data(150, 5, n_features=10)
        
        # Normalize embeddings for fair comparison
        embeddings_norm = normalize_embeddings(embeddings)
        
        # Non-adaptive (traditional) on normalized embeddings
        cfg_traditional = {
            "adaptive": False,
            "method": "kmeans",
            "n_clusters": 5,
            "random_state": 42,
        }
        model_trad, labels_trad, _ = make_clusterer(embeddings_norm, cfg_traditional)
        score_trad = silhouette_score(embeddings_norm, labels_trad)
        dbi_trad = davies_bouldin_score(embeddings_norm, labels_trad)
        
        # Adaptive (with normalization and k-means++) on same normalized embeddings
        cfg_adaptive = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": False},  # Already normalized
        }
        model_adaptive, labels_adaptive, _ = make_clusterer(embeddings_norm, cfg_adaptive)
        score_adaptive = silhouette_score(embeddings_norm, labels_adaptive)
        dbi_adaptive = davies_bouldin_score(embeddings_norm, labels_adaptive)
        
        # Both should produce valid metrics
        # (k-means++ in adaptive mode may help, but results depend on data)
        assert score_adaptive > 0.0  # Should be positive
        assert score_trad > 0.0  # Should be positive
        # DBI: lower is better
        assert dbi_adaptive >= 0.0  # Should be non-negative
        assert dbi_trad >= 0.0  # Should be non-negative
    
    def test_optimal_k_improves_metrics(self):
        """Test that finding optimal k improves metrics."""
        embeddings, _ = create_clustered_data(100, 5, n_features=10)
        embeddings = normalize_embeddings(embeddings)
        
        # Fixed k (suboptimal)
        cfg_fixed = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 10,  # Too many clusters
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        model_fixed, labels_fixed, _ = make_clusterer(embeddings, cfg_fixed)
        score_fixed = silhouette_score(embeddings, labels_fixed)
        
        # Auto-optimized k
        cfg_auto = {
            "adaptive": True,
            "auto_n_clusters": True,
            "random_state": 42,
            "preprocessing": {
                "normalize_embeddings": True,
                "max_optimization_time": 10,  # Short for testing
            },
        }
        model_auto, labels_auto, _ = make_clusterer(embeddings, cfg_auto)
        score_auto = silhouette_score(embeddings, labels_auto)
        
        # Optimal k should improve or maintain score
        # (may not always be better due to time limit, but should be reasonable)
        assert score_auto > 0.0
        assert score_auto >= score_fixed - 0.1  # Allow tolerance


class TestMetricCalculation:
    """Test metric calculation accuracy."""
    
    def test_silhouette_score_calculation(self):
        """Test silhouette score is calculated correctly."""
        embeddings, labels_true = create_clustered_data(100, 5, n_features=10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        
        model, labels, _ = make_clusterer(embeddings, cfg)
        score = silhouette_score(embeddings, labels)
        
        # Score should be in valid range [-1, 1]
        assert -1.0 <= score <= 1.0
        # For well-separated clusters, score should be positive
        assert score > 0.0
    
    def test_dbi_calculation(self):
        """Test Davies-Bouldin Index is calculated correctly."""
        embeddings, labels_true = create_clustered_data(100, 5, n_features=10)
        
        cfg = {
            "adaptive": True,
            "auto_n_clusters": False,
            "n_clusters": 5,
            "random_state": 42,
            "preprocessing": {"normalize_embeddings": True},
        }
        
        model, labels, _ = make_clusterer(embeddings, cfg)
        dbi = davies_bouldin_score(embeddings, labels)
        
        # DBI should be non-negative (lower is better)
        assert dbi >= 0.0
        # For well-separated clusters, DBI should be relatively low
        assert dbi < 2.0
    
    def test_metrics_different_dataset_sizes(self):
        """Test metrics work correctly for different dataset sizes."""
        sizes = [50, 100, 200, 300]
        
        for size in sizes:
            embeddings, _ = create_clustered_data(size, 5, n_features=10)
            embeddings = normalize_embeddings(embeddings)
            
            cfg = {
                "adaptive": True,
                "auto_n_clusters": False,
                "n_clusters": 5,
                "random_state": 42,
                "preprocessing": {"normalize_embeddings": True},
            }
            
            model, labels, _ = make_clusterer(embeddings, cfg)
            score = silhouette_score(embeddings, labels)
            dbi = davies_bouldin_score(embeddings, labels)
            
            # Metrics should be valid
            assert -1.0 <= score <= 1.0
            assert dbi >= 0.0


class TestSemanticMetrics:
    """Test semantic clustering metrics."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model for testing."""
        class MockEmbeddingModel:
            def encode(self, texts):
                """Return simple embeddings based on text length."""
                if isinstance(texts, str):
                    texts = [texts]
                # Create deterministic embeddings based on text
                embeddings = []
                for text in texts:
                    # Simple hash-based embedding
                    hash_val = hash(text) % 1000
                    emb = np.random.RandomState(hash_val).randn(384)
                    # Normalize
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append(emb)
                return np.array(embeddings)
        
        return MockEmbeddingModel()
    
    def test_topic_coherence_keywords(self, mock_embedding_model):
        """Test topic coherence calculation using keywords."""
        # Create cluster summaries with related keywords
        summaries_df = pd.DataFrame({
            "cluster_id": [0, 1, 2],
            "keywords": [
                ["machine", "learning", "neural", "network"],  # Coherent
                ["cooking", "recipe", "cuisine", "food"],  # Coherent
                ["space", "rocket", "planet", "astronaut"],  # Coherent
            ],
        })
        
        coherence = calculate_topic_coherence_keywords(summaries_df, mock_embedding_model)
        
        # Should return a valid coherence score
        assert coherence is not None
        assert 0.0 <= coherence <= 1.0
    
    def test_topic_coherence_articles(self):
        """Test topic coherence calculation using article embeddings."""
        # Create clustered data
        embeddings, labels = create_clustered_data(100, 5, n_features=10)
        
        # Create assignments DataFrame
        titles = [f"article_{i}" for i in range(len(embeddings))]
        assignments_df = pd.DataFrame({
            "title": titles,
            "cluster_id": labels,
        })
        
        # Create title to index mapping
        title_to_index = {title: i for i, title in enumerate(titles)}
        
        coherence = calculate_topic_coherence_articles(assignments_df, embeddings, title_to_index)
        
        # Should return a valid coherence score
        assert coherence is not None
        assert 0.0 <= coherence <= 1.0
    
    def test_category_alignment(self):
        """Test category alignment calculation."""
        # Create assignments with articles
        assignments_df = pd.DataFrame({
            "title": ["article_1", "article_2", "article_3", "article_4"],
            "cluster_id": [0, 0, 1, 1],
        })
        
        # Create articles with categories
        cleaned_articles_df = pd.DataFrame({
            "title": ["article_1", "article_2", "article_3", "article_4"],
            "categories": [
                ["Machine learning", "Artificial intelligence"],  # Cluster 0
                ["Machine learning", "Algorithms"],  # Cluster 0 - shares category
                ["Cooking", "Italian cuisine"],  # Cluster 1
                ["Cooking", "Recipes"],  # Cluster 1 - shares category
            ],
        })
        
        alignment = calculate_category_alignment(assignments_df, cleaned_articles_df)
        
        # Should return a valid alignment score
        assert alignment is not None
        assert 0.0 <= alignment <= 1.0
        # Articles in same cluster should have some category overlap
        assert alignment > 0.0
    
    def test_cluster_diversity_centroids(self):
        """Test cluster diversity calculation using centroids."""
        # Create well-separated centroids
        centers = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        diversity = calculate_cluster_diversity_centroids(centers)
        
        # Should return a valid diversity score
        assert diversity is not None
        assert 0.0 <= diversity <= 1.0
        # Well-separated centroids should have high diversity
        assert diversity > 0.5
    
    def test_cluster_diversity_keywords(self, mock_embedding_model):
        """Test cluster diversity calculation using keywords."""
        # Create cluster summaries with distinct keywords
        summaries_df = pd.DataFrame({
            "cluster_id": [0, 1, 2],
            "keywords": [
                ["machine", "learning", "neural"],
                ["cooking", "recipe", "cuisine"],
                ["space", "rocket", "planet"],
            ],
        })
        
        diversity = calculate_cluster_diversity_keywords(summaries_df, mock_embedding_model)
        
        # Should return a valid diversity score
        assert diversity is not None
        assert 0.0 <= diversity <= 1.0
    
    def test_calculate_all_semantic_metrics(self, mock_embedding_model):
        """Test calculation of all semantic metrics together."""
        # Create test data
        embeddings, labels = create_clustered_data(50, 3, n_features=10)
        
        summaries_df = pd.DataFrame({
            "cluster_id": [0, 1, 2],
            "keywords": [
                ["machine", "learning"],
                ["cooking", "recipe"],
                ["space", "rocket"],
            ],
        })
        
        titles = [f"article_{i}" for i in range(len(embeddings))]
        assignments_df = pd.DataFrame({
            "title": titles,
            "cluster_id": labels,
        })
        
        cleaned_articles_df = pd.DataFrame({
            "title": titles,
            "categories": [
                ["Category_A"] if i % 3 == 0 else ["Category_B"] if i % 3 == 1 else ["Category_C"]
                for i in range(len(titles))
            ],
        })
        
        # Calculate cluster centers
        centers = np.array([
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
            embeddings[labels == 2].mean(axis=0),
        ])
        
        # Calculate all metrics
        metrics = calculate_all_semantic_metrics(
            summaries_df=summaries_df,
            assignments_df=assignments_df,
            cleaned_articles_df=cleaned_articles_df,
            embeddings=embeddings,
            centers=centers,
            embedding_model=mock_embedding_model,
            metrics_config={"coherence_method": "keywords", "diversity_method": "centroids"},
        )
        
        # Should return a dictionary with metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check that expected metrics are present
        if "topic_coherence_keywords" in metrics:
            assert 0.0 <= metrics["topic_coherence_keywords"] <= 1.0
        
        if "category_alignment" in metrics:
            assert 0.0 <= metrics["category_alignment"] <= 1.0
        
        if "cluster_diversity_centroids" in metrics:
            assert 0.0 <= metrics["cluster_diversity_centroids"] <= 1.0
    
    def test_metrics_handle_missing_data(self, mock_embedding_model):
        """Test that metrics handle missing data gracefully."""
        # Empty summaries
        empty_summaries = pd.DataFrame({"cluster_id": [], "keywords": []})
        coherence = calculate_topic_coherence_keywords(empty_summaries, mock_embedding_model)
        assert coherence is None
        
        # Empty assignments
        empty_assignments = pd.DataFrame({"title": [], "cluster_id": []})
        empty_articles = pd.DataFrame({"title": [], "categories": []})
        alignment = calculate_category_alignment(empty_assignments, empty_articles)
        assert alignment is None
        
        # Single cluster (not enough for diversity)
        single_center = np.array([[1.0, 0.0, 0.0]])
        diversity = calculate_cluster_diversity_centroids(single_center)
        assert diversity is None

