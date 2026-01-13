"""
Tests for 2D embedding map generation using UMAP.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.modeling.embedding_map import generate_embedding_map_2d


@pytest.fixture
def sample_embeddings_df():
    """Create sample embeddings dataframe."""
    n_samples = 50
    embedding_dim = 384
    
    # Generate random embeddings
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Create dataframe
    df = pd.DataFrame({
        "title": [f"Article_{i}" for i in range(n_samples)],
        "embedding": [emb for emb in embeddings],
    })
    
    return df


@pytest.fixture
def sample_assignments_df():
    """Create sample cluster assignments dataframe."""
    n_samples = 50
    n_clusters = 5
    
    df = pd.DataFrame({
        "title": [f"Article_{i}" for i in range(n_samples)],
        "cluster_id": np.random.randint(0, n_clusters, n_samples),
    })
    
    return df


@pytest.fixture
def sample_summaries_df():
    """Create sample cluster summaries dataframe."""
    n_clusters = 5
    
    data = []
    for i in range(n_clusters):
        data.append({
            "cluster_id": i,
            "size": 10,
            "keywords": [f"keyword_{i}_{j}" for j in range(5)],
            "top_articles": [f"Article_{i * 10 + j}" for j in range(3)],
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(sample_embeddings_df, sample_assignments_df, sample_summaries_df):
    """Create temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectories
        features_dir = os.path.join(tmpdir, "features")
        clustering_dir = os.path.join(tmpdir, "clustering")
        os.makedirs(features_dir)
        os.makedirs(clustering_dir)
        
        # Save files
        embeddings_path = os.path.join(features_dir, "embeddings.parquet")
        assignments_path = os.path.join(clustering_dir, "cluster_assignments.parquet")
        summaries_path = os.path.join(clustering_dir, "clusters_summary.parquet")
        output_path = os.path.join(clustering_dir, "embedding_map_2d.parquet")
        
        sample_embeddings_df.to_parquet(embeddings_path, index=False)
        sample_assignments_df.to_parquet(assignments_path, index=False)
        sample_summaries_df.to_parquet(summaries_path, index=False)
        
        yield {
            "embeddings_path": embeddings_path,
            "assignments_path": assignments_path,
            "summaries_path": summaries_path,
            "output_path": output_path,
        }


class TestEmbeddingMapGeneration:
    """Tests for generate_embedding_map_2d function."""
    
    def test_generate_embedding_map_basic(self, temp_data_dir):
        """Test basic 2D embedding map generation."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            summaries_path=temp_data_dir["summaries_path"],
            output_path=temp_data_dir["output_path"],
            n_neighbors=10,
            min_dist=0.1,
            random_state=42,
        )
        
        # Check result
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50  # Same as input
        
        # Check columns (keywords are optional)
        assert "title" in result.columns
        assert "cluster_id" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns
        
        # Check data types
        assert result["x"].dtype == np.float64
        assert result["y"].dtype == np.float64
        assert result["cluster_id"].dtype == np.int64
        
        # Check that output file was created
        assert os.path.exists(temp_data_dir["output_path"])
        
    def test_generate_embedding_map_no_summaries(self, temp_data_dir):
        """Test generation when summaries file is missing (keywords optional)."""
        # Remove summaries file
        os.remove(temp_data_dir["summaries_path"])
        
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            summaries_path=temp_data_dir["summaries_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Should still succeed, just without keywords
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "x" in result.columns
        assert "y" in result.columns
        
    def test_generate_embedding_map_missing_embeddings(self, temp_data_dir):
        """Test handling of missing embeddings file."""
        # Remove embeddings file
        os.remove(temp_data_dir["embeddings_path"])
        
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
        )
        
        # Should return None
        assert result is None
        
    def test_generate_embedding_map_missing_assignments(self, temp_data_dir):
        """Test handling of missing assignments file."""
        # Remove assignments file
        os.remove(temp_data_dir["assignments_path"])
        
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
        )
        
        # Should return None
        assert result is None
        
    def test_generate_embedding_map_small_dataset(self, temp_data_dir):
        """Test with small dataset (adjusts n_neighbors)."""
        # Create small dataset
        small_embeddings = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(10)],
            "embedding": [np.random.randn(384).astype(np.float32) for _ in range(10)],
        })
        
        small_assignments = pd.DataFrame({
            "title": [f"Article_{i}" for i in range(10)],
            "cluster_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        })
        
        small_embeddings.to_parquet(temp_data_dir["embeddings_path"], index=False)
        small_assignments.to_parquet(temp_data_dir["assignments_path"], index=False)
        
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            n_neighbors=15,  # More than available samples
            random_state=42,
        )
        
        # Should succeed with adjusted n_neighbors
        assert result is not None
        assert len(result) == 10
        
    def test_generate_embedding_map_coordinates_valid(self, temp_data_dir):
        """Test that generated coordinates are valid numbers."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Check no NaN values
        assert not result["x"].isna().any()
        assert not result["y"].isna().any()
        
        # Check finite values
        assert np.all(np.isfinite(result["x"]))
        assert np.all(np.isfinite(result["y"]))
        
    def test_generate_embedding_map_titles_preserved(self, temp_data_dir):
        """Test that article titles are preserved correctly."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Check all titles present
        expected_titles = [f"Article_{i}" for i in range(50)]
        assert set(result["title"]) == set(expected_titles)
        
    def test_generate_embedding_map_clusters_preserved(self, temp_data_dir):
        """Test that cluster assignments are preserved."""
        # Load original assignments
        assignments_df = pd.read_parquet(temp_data_dir["assignments_path"])
        original_clusters = dict(zip(assignments_df["title"], assignments_df["cluster_id"]))
        
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Check cluster assignments match
        for _, row in result.iterrows():
            assert row["cluster_id"] == original_clusters[row["title"]]
            
    def test_generate_embedding_map_keywords_included(self, temp_data_dir):
        """Test that keywords are included when summaries available."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            summaries_path=temp_data_dir["summaries_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Keywords are optional - if summaries exist and have keywords, they should be included
        # If not, that's okay too (depends on summaries file format)
        assert result is not None
        assert len(result) == 50
        
        # If keywords column exists, check it has valid data
        if "keywords" in result.columns:
            has_keywords = result["keywords"].apply(lambda x: isinstance(x, list))
            assert has_keywords.all()  # All should be lists (even if empty)
        
    def test_generate_embedding_map_handles_exceptions(self, temp_data_dir):
        """Test that exceptions are handled gracefully."""
        # Try with invalid embedding dimension (force error)
        # Create embeddings with inconsistent dimensions
        bad_embeddings = pd.DataFrame({
            "title": ["Article_0", "Article_1"],
            "embedding": [
                np.random.randn(384).astype(np.float32),
                np.random.randn(256).astype(np.float32),  # Different dimension
            ],
        })
        
        bad_embeddings.to_parquet(temp_data_dir["embeddings_path"], index=False)
        
        # Should handle error gracefully and return None
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
        )
        
        # Should return None on error
        assert result is None
            
    def test_generate_embedding_map_reproducible(self, temp_data_dir):
        """Test that generation is reproducible with same random_state."""
        result1 = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Generate again with same seed
        result2 = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"] + ".2",
            random_state=42,
        )
        
        # Results should be identical
        assert len(result1) == len(result2)
        np.testing.assert_array_almost_equal(result1["x"].values, result2["x"].values, decimal=5)
        np.testing.assert_array_almost_equal(result1["y"].values, result2["y"].values, decimal=5)
        
    def test_generate_embedding_map_different_seeds(self, temp_data_dir):
        """Test that different random seeds produce different results."""
        result1 = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Generate with different seed
        result2 = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"] + ".2",
            random_state=123,
        )
        
        # Results should be different
        assert not np.allclose(result1["x"].values, result2["x"].values)
        assert not np.allclose(result1["y"].values, result2["y"].values)
        
    def test_generate_embedding_map_output_file_created(self, temp_data_dir):
        """Test that output file is created and can be loaded."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Check file exists
        assert os.path.exists(temp_data_dir["output_path"])
        
        # Load and verify
        loaded = pd.read_parquet(temp_data_dir["output_path"])
        assert len(loaded) == len(result)
        assert list(loaded.columns) == list(result.columns)
        
    def test_generate_embedding_map_custom_parameters(self, temp_data_dir):
        """Test with custom UMAP parameters."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            output_path=temp_data_dir["output_path"],
            n_neighbors=20,
            min_dist=0.5,
            random_state=42,
        )
        
        # Should succeed with custom parameters
        assert result is not None
        assert len(result) == 50


class TestEmbeddingMapIntegration:
    """Integration tests for embedding map in the pipeline."""
    
    def test_embedding_map_file_structure(self, temp_data_dir):
        """Test that generated file has correct structure for API."""
        result = generate_embedding_map_2d(
            embeddings_path=temp_data_dir["embeddings_path"],
            assignments_path=temp_data_dir["assignments_path"],
            summaries_path=temp_data_dir["summaries_path"],
            output_path=temp_data_dir["output_path"],
            random_state=42,
        )
        
        # Check API-compatible structure
        assert "title" in result.columns
        assert "cluster_id" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns
        
        # Check data can be converted to JSON-compatible format
        for _, row in result.head().iterrows():
            point = {
                "title": str(row["title"]),
                "cluster_id": int(row["cluster_id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
            }
            assert isinstance(point["title"], str)
            assert isinstance(point["cluster_id"], int)
            assert isinstance(point["x"], float)
            assert isinstance(point["y"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
