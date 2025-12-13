"""
Knowledge graph construction using NetworkX.

Builds a multi-layer graph from Wikipedia articles with:
- Layer 2: Cluster relationships (same-cluster connections)
- Layer 3: Semantic relationships (embedding similarity)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds a multi-layer knowledge graph from articles, clusters, and embeddings."""

    def __init__(
        self,
        semantic_threshold: float = 0.7,
        enable_cluster_edges: bool = True,
    ):
        """
        Initialize graph builder.

        Args:
            semantic_threshold: Minimum cosine similarity for semantic edges (Layer 3)
            enable_cluster_edges: Whether to add cluster relationship edges (Layer 2)
        """
        self.semantic_threshold = semantic_threshold
        self.enable_cluster_edges = enable_cluster_edges
        self.graph: Optional[nx.DiGraph] = None

    def build_graph(
        self,
        articles_df: pd.DataFrame,
        cluster_assignments: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> nx.DiGraph:
        """
        Build multi-layer knowledge graph.

        Args:
            articles_df: DataFrame with columns: title, links, categories
            cluster_assignments: DataFrame with columns: title, cluster_id
            embeddings: Embedding matrix (n_articles, embedding_dim)

        Returns:
            NetworkX DiGraph with nodes (articles) and edges (relationships)
        """
        logger.info("Building knowledge graph from %d articles", len(articles_df))

        # Create graph
        self.graph = nx.DiGraph()

        # Create title to index mapping
        title_to_idx: Dict[str, int] = {
            str(title).lower(): idx for idx, title in enumerate(articles_df["title"])
        }
        titles = articles_df["title"].astype(str).tolist()

        # Create cluster mapping
        cluster_map: Dict[str, int] = {}
        if "title" in cluster_assignments.columns and "cluster_id" in cluster_assignments.columns:
            for _, row in cluster_assignments.iterrows():
                title = str(row["title"]).lower()
                cluster_id = int(row["cluster_id"])
                cluster_map[title] = cluster_id

        # Add nodes
        logger.info("Adding %d nodes to graph", len(titles))
        for idx, title in enumerate(titles):
            cluster_id = cluster_map.get(title.lower(), -1)
            self.graph.add_node(
                title,
                index=idx,
                cluster_id=cluster_id,
            )

        # Layer 2: Cluster relationships
        if self.enable_cluster_edges:
            logger.info("Adding Layer 2 edges (cluster relationships)...")
            layer2_count = self._add_cluster_edges(cluster_map, titles)
            logger.info("Added %d Layer 2 edges", layer2_count)
        else:
            logger.info("Skipping Layer 2 edges (disabled)")

        # Layer 3: Semantic relationships
        logger.info("Adding Layer 3 edges (semantic similarity)...")
        layer3_count = self._add_semantic_edges(embeddings, titles, title_to_idx)
        logger.info("Added %d Layer 3 edges", layer3_count)

        # Compute basic graph metrics
        total_edges = self.graph.number_of_edges()
        total_nodes = self.graph.number_of_nodes()
        logger.info(
            "Graph construction complete: %d nodes, %d edges (L2: %d, L3: %d)",
            total_nodes,
            total_edges,
            layer2_count if self.enable_cluster_edges else 0,
            layer3_count,
        )

        return self.graph

    def _add_cluster_edges(
        self,
        cluster_map: Dict[str, int],
        titles: List[str],
    ) -> int:
        """Add Layer 2 edges: same-cluster connections."""
        edge_count = 0

        # Group articles by cluster using original titles
        cluster_to_titles: Dict[int, List[str]] = {}
        for title in titles:
            title_lower = title.lower()
            if title_lower in cluster_map:
                cluster_id = cluster_map[title_lower]
                if cluster_id not in cluster_to_titles:
                    cluster_to_titles[cluster_id] = []
                cluster_to_titles[cluster_id].append(title)

        # For each cluster, connect all articles to each other
        for cluster_id, titles in cluster_to_titles.items():
            if len(titles) < 2:
                continue
            # Connect each article to all others in the same cluster
            for i, source in enumerate(titles):
                for target in titles[i + 1 :]:
                    if not self.graph.has_edge(source, target):
                        self.graph.add_edge(
                            source,
                            target,
                            layer=2,
                            weight=1.0,
                            type="cluster",
                        )
                        edge_count += 1
                    # Add reverse edge for undirected cluster relationships
                    if not self.graph.has_edge(target, source):
                        self.graph.add_edge(
                            target,
                            source,
                            layer=2,
                            weight=1.0,
                            type="cluster",
                        )
                        edge_count += 1

        return edge_count

    def _add_semantic_edges(
        self,
        embeddings: np.ndarray,
        titles: List[str],
        title_to_idx: Dict[str, int],
    ) -> int:
        """Add Layer 3 edges: semantic similarity connections."""
        edge_count = 0

        # Handle empty embeddings
        if len(embeddings) == 0 or len(titles) == 0:
            return 0

        # Compute pairwise cosine similarity
        logger.info("Computing pairwise cosine similarities...")
        similarity_matrix = cosine_similarity(embeddings)

        # Add edges for high similarity pairs
        n = len(titles)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= self.semantic_threshold:
                    source = titles[i]
                    target = titles[j]
                    # Add bidirectional edges for semantic relationships
                    if not self.graph.has_edge(source, target):
                        self.graph.add_edge(
                            source,
                            target,
                            layer=3,
                            weight=float(similarity),
                            type="semantic",
                        )
                        edge_count += 1
                    if not self.graph.has_edge(target, source):
                        self.graph.add_edge(
                            target,
                            source,
                            layer=3,
                            weight=float(similarity),
                            type="semantic",
                        )
                        edge_count += 1

        return edge_count

    def get_graph(self) -> Optional[nx.DiGraph]:
        """Get the constructed graph."""
        return self.graph

    def save_graph(self, path: str) -> None:
        """Save graph to disk using pickle."""
        import pickle

        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")

        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("Saved graph to %s (%d nodes, %d edges)", path, self.graph.number_of_nodes(), self.graph.number_of_edges())

    @staticmethod
    def load_graph(path: str) -> nx.DiGraph:
        """Load graph from disk."""
        import pickle

        with open(path, "rb") as f:
            graph = pickle.load(f)
        logger.info("Loaded graph from %s (%d nodes, %d edges)", path, graph.number_of_nodes(), graph.number_of_edges())
        return graph

