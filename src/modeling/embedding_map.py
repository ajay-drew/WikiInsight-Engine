"""
Generate 2D embedding map using UMAP for visualization.

This module reduces high-dimensional embeddings to 2D using UMAP,
making it possible to visualize cluster topology and semantic relationships
in an interactive scatter plot.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_embedding_map_2d(
    embeddings_path: str = "data/features/embeddings.parquet",
    assignments_path: str = "models/clustering/cluster_assignments.parquet",
    summaries_path: str = "models/clustering/clusters_summary.parquet",
    output_path: str = "models/clustering/embedding_map_2d.parquet",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Generate 2D UMAP projection of article embeddings.
    
    Args:
        embeddings_path: Path to embeddings parquet file
        assignments_path: Path to cluster assignments parquet file
        summaries_path: Path to cluster summaries parquet file
        output_path: Path to save 2D embedding map
        n_neighbors: UMAP n_neighbors parameter (controls local structure)
        min_dist: UMAP min_dist parameter (controls point spacing)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: title, cluster_id, x, y, keywords (if available)
        Returns None if generation fails
    """
    try:
        # Check if required files exist
        if not os.path.exists(embeddings_path):
            logger.warning("Embeddings file not found: %s", embeddings_path)
            return None
        
        if not os.path.exists(assignments_path):
            logger.warning("Assignments file not found: %s", assignments_path)
            return None
        
        logger.info("=" * 80)
        logger.info("Generating 2D embedding map with UMAP")
        logger.info("  - Embeddings: %s", embeddings_path)
        logger.info("  - Assignments: %s", assignments_path)
        logger.info("  - Output: %s", output_path)
        logger.info("  - UMAP params: n_neighbors=%d, min_dist=%.2f", n_neighbors, min_dist)
        logger.info("=" * 80)
        
        # Load embeddings
        logger.info("Loading embeddings...")
        embeddings_df = pd.read_parquet(embeddings_path)
        
        if "embedding" not in embeddings_df.columns:
            logger.error("No 'embedding' column in embeddings file")
            return None
        
        # Extract embedding matrix
        embeddings = np.vstack(embeddings_df["embedding"].to_list())
        titles = embeddings_df["title"].tolist()
        n_samples = embeddings.shape[0]
        logger.info("  - Loaded %d embeddings with shape %s", n_samples, embeddings.shape)
        
        # Load cluster assignments
        logger.info("Loading cluster assignments...")
        assignments_df = pd.read_parquet(assignments_path)
        
        # Merge to align titles
        merged = embeddings_df[["title"]].merge(
            assignments_df[["title", "cluster_id"]], 
            on="title", 
            how="left"
        )
        cluster_ids = merged["cluster_id"].fillna(-1).astype(int).tolist()
        
        # Load cluster summaries for keywords (optional)
        keywords_map = {}
        if os.path.exists(summaries_path):
            try:
                summaries_df = pd.read_parquet(summaries_path)
                for _, row in summaries_df.iterrows():
                    cluster_id = int(row["cluster_id"])
                    kw = row.get("keywords", [])
                    if isinstance(kw, list) and len(kw) > 0:
                        # Take top 5 keywords
                        keywords_map[cluster_id] = kw[:5]
            except Exception as e:
                logger.warning("Failed to load keywords from summaries: %s", e)
        
        # Run UMAP
        logger.info("Running UMAP dimensionality reduction...")
        logger.info("  - This may take a few moments for large datasets...")
        
        try:
            import umap
        except ImportError:
            logger.error("UMAP library not installed. Install with: pip install umap-learn")
            return None
        
        # Adjust n_neighbors if dataset is too small
        effective_n_neighbors = min(n_neighbors, n_samples - 1)
        if effective_n_neighbors < n_neighbors:
            logger.info("  - Adjusted n_neighbors from %d to %d (dataset size)", n_neighbors, effective_n_neighbors)
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric="cosine",  # Use cosine for semantic embeddings
            verbose=False,
        )
        
        embedding_2d = reducer.fit_transform(embeddings)
        logger.info("  - UMAP projection complete: shape %s", embedding_2d.shape)
        
        # Create output dataframe
        output_data = {
            "title": titles,
            "cluster_id": cluster_ids,
            "x": embedding_2d[:, 0].tolist(),
            "y": embedding_2d[:, 1].tolist(),
        }
        
        # Add keywords if available
        if keywords_map:
            output_data["keywords"] = [
                keywords_map.get(cid, []) for cid in cluster_ids
            ]
        
        output_df = pd.DataFrame(output_data)
        
        # Save to parquet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_parquet(output_path, index=False)
        
        logger.info("=" * 80)
        logger.info("2D embedding map saved successfully")
        logger.info("  - Output: %s", output_path)
        logger.info("  - Total points: %d", len(output_df))
        logger.info("  - Unique clusters: %d", output_df["cluster_id"].nunique())
        logger.info("=" * 80)
        
        return output_df
        
    except Exception as e:
        logger.exception("Failed to generate 2D embedding map: %s", e)
        return None


if __name__ == "__main__":
    # For standalone testing
    from src.common.logging_utils import setup_logging
    setup_logging()
    
    result = generate_embedding_map_2d()
    if result is not None:
        print(f"Successfully generated 2D embedding map with {len(result)} points")
    else:
        print("Failed to generate 2D embedding map")
