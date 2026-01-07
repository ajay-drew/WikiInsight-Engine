"""
Graph construction pipeline script.

Builds knowledge graph from articles, clusters, and embeddings.
This script is designed to be called via:
    python -m src.graph.build_graph
and is referenced from `dvc.yaml` as the `build_graph` stage.
"""

import logging
import os
from time import perf_counter
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.common.logging_utils import setup_logging
from src.common.pipeline_progress import update_progress, mark_stage_completed, mark_stage_error
from src.modeling.cluster_topics import (
    CLEANED_ARTICLES_PATH,
    CLUSTER_ASSIGNMENTS_PATH,
    EMBEDDINGS_PATH,
)
from .knowledge_graph import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
GRAPH_PATH = os.path.join("data", "graph", "knowledge_graph.pkl")


def load_config(path: str = CONFIG_PATH) -> Dict:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles() -> pd.DataFrame:
    """Load cleaned articles with links and categories."""
    if not os.path.exists(CLEANED_ARTICLES_PATH):
        raise FileNotFoundError(
            f"Cleaned articles file not found at {CLEANED_ARTICLES_PATH}. "
            "Run preprocessing stage first."
        )
    df = pd.read_parquet(CLEANED_ARTICLES_PATH)
    logger.info("Loaded %d articles from %s", len(df), CLEANED_ARTICLES_PATH)
    return df


def load_cluster_assignments() -> pd.DataFrame:
    """Load cluster assignments."""
    if not os.path.exists(CLUSTER_ASSIGNMENTS_PATH):
        raise FileNotFoundError(
            f"Cluster assignments file not found at {CLUSTER_ASSIGNMENTS_PATH}. "
            "Run clustering stage first."
        )
    df = pd.read_parquet(CLUSTER_ASSIGNMENTS_PATH)
    logger.info("Loaded %d cluster assignments from %s", len(df), CLUSTER_ASSIGNMENTS_PATH)
    return df


def load_embeddings() -> np.ndarray:
    """Load embeddings matrix."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found at {EMBEDDINGS_PATH}. "
            "Run preprocessing stage first."
        )
    df = pd.read_parquet(EMBEDDINGS_PATH)
    if "embedding" not in df.columns:
        raise ValueError("Embeddings parquet must contain an 'embedding' column")
    embeddings = np.vstack(df["embedding"].to_list())
    logger.info("Loaded embeddings matrix: %s", embeddings.shape)
    return embeddings


def main() -> None:
    """Main graph construction pipeline."""
    setup_logging()
    pipeline_start = perf_counter()
    
    logger.info("=" * 80)
    logger.info("Starting knowledge graph construction pipeline")
    logger.info("=" * 80)
    
    update_progress("build_graph", "running", 0.0, "Loading articles, clusters, and embeddings...")

    try:
        # Load configuration
        logger.info("Loading configuration from %s...", CONFIG_PATH)
        config = load_config(CONFIG_PATH)
        graph_cfg = config.get("graph", {})
        semantic_threshold = float(graph_cfg.get("semantic_similarity_threshold", 0.7))
        enable_cluster_edges = bool(graph_cfg.get("enable_cluster_edges", True))
        
        logger.info("Graph configuration:")
        logger.info("  - Semantic similarity threshold: %.2f", semantic_threshold)
        logger.info("  - Enable cluster edges: %s", enable_cluster_edges)

        # Load data
        logger.info("=" * 80)
        logger.info("Loading input data...")
        logger.info("=" * 80)
        
        update_progress("build_graph", "running", 20.0, "Loading articles...")
        load_start = perf_counter()
        articles_df = load_articles()
        load_time = perf_counter() - load_start
        logger.info("Articles loaded in %.2f seconds", load_time)
        
        update_progress("build_graph", "running", 40.0, "Loading cluster assignments...")
        load_start = perf_counter()
        cluster_assignments = load_cluster_assignments()
        load_time = perf_counter() - load_start
        logger.info("Cluster assignments loaded in %.2f seconds", load_time)
        
        update_progress("build_graph", "running", 60.0, "Loading embeddings...")
        load_start = perf_counter()
        embeddings = load_embeddings()
        load_time = perf_counter() - load_start
        logger.info("Embeddings loaded in %.2f seconds", load_time)

        # Validate data consistency
        logger.info("Validating data consistency...")
        if len(articles_df) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(articles_df)} articles but {len(embeddings)} embeddings"
            )
        if len(articles_df) != len(cluster_assignments):
            raise ValueError(
                f"Mismatch: {len(articles_df)} articles but {len(cluster_assignments)} cluster assignments"
            )
        logger.info("Data validation passed: %d articles, %d embeddings, %d cluster assignments",
                   len(articles_df), len(embeddings), len(cluster_assignments))

        # Build graph
        logger.info("=" * 80)
        logger.info("Building knowledge graph...")
        logger.info("=" * 80)
        update_progress("build_graph", "running", 80.0, "Building knowledge graph...")
        
        build_start = perf_counter()
        builder = KnowledgeGraphBuilder(
            semantic_threshold=semantic_threshold,
            enable_cluster_edges=enable_cluster_edges,
        )
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)
        build_time = perf_counter() - build_start
        
        logger.info("Graph building completed in %.2f seconds", build_time)
        logger.info("Graph statistics:")
        logger.info("  - Nodes: %d", len(graph.nodes()) if hasattr(graph, 'nodes') else 'N/A')
        logger.info("  - Edges: %d", len(graph.edges()) if hasattr(graph, 'edges') else 'N/A')

        # Save graph
        logger.info("=" * 80)
        logger.info("Saving knowledge graph...")
        logger.info("=" * 80)
        update_progress("build_graph", "running", 95.0, "Saving knowledge graph...")
        
        save_start = perf_counter()
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        builder.save_graph(GRAPH_PATH)
        save_time = perf_counter() - save_start
        
        logger.info("Graph saved to %s in %.2f seconds", GRAPH_PATH, save_time)
        
        total_time = perf_counter() - pipeline_start
        logger.info("=" * 80)
        logger.info("Knowledge graph construction complete!")
        logger.info("Total time: %.2f seconds (%.1f minutes)", total_time, total_time / 60)
        logger.info("=" * 80)
        
        mark_stage_completed("build_graph", "Knowledge graph construction complete")
    except FileNotFoundError as exc:
        error_msg = f"Graph construction failed: {exc}"
        logger.error(error_msg)
        mark_stage_error("build_graph", error_msg)
        raise
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Graph construction failed: {exc}"
        logger.exception(error_msg)
        mark_stage_error("build_graph", error_msg)
        raise


if __name__ == "__main__":
    main()

