"""
Graph construction pipeline script.

Builds knowledge graph from articles, clusters, and embeddings.
This script is designed to be called via:
    python -m src.graph.build_graph
and is referenced from `dvc.yaml` as the `build_graph` stage.
"""

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.common.logging_utils import setup_logging
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
    logger.info("Starting knowledge graph construction pipeline")

    try:
        # Load configuration
        config = load_config(CONFIG_PATH)
        graph_cfg = config.get("graph", {})
        semantic_threshold = float(graph_cfg.get("semantic_similarity_threshold", 0.7))
        enable_cluster_edges = bool(graph_cfg.get("enable_cluster_edges", True))

        # Load data
        articles_df = load_articles()
        cluster_assignments = load_cluster_assignments()
        embeddings = load_embeddings()

        # Validate data consistency
        if len(articles_df) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(articles_df)} articles but {len(embeddings)} embeddings"
            )

        # Build graph
        builder = KnowledgeGraphBuilder(
            semantic_threshold=semantic_threshold,
            enable_cluster_edges=enable_cluster_edges,
        )
        graph = builder.build_graph(articles_df, cluster_assignments, embeddings)

        # Save graph
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        builder.save_graph(GRAPH_PATH)

        logger.info("Knowledge graph construction complete")
    except FileNotFoundError as exc:
        logger.error("Graph construction failed: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph construction failed with error: %s", exc)
        raise


if __name__ == "__main__":
    main()

