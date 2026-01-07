"""
Knowledge graph construction using NetworkX.

Builds a multi-layer graph from Wikipedia articles with:
- Layer 1: Link relationships (Wikipedia article links)
- Layer 2: Cluster relationships (same-cluster connections)
- Layer 3: Semantic relationships (embedding similarity)
"""

import logging
from time import perf_counter
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds a multi-layer knowledge graph from articles, clusters, and embeddings."""

    def __init__(
        self,
        semantic_threshold: float = 0.7,
        enable_cluster_edges: bool = True,
        enable_link_edges: bool = True,
    ):
        """
        Initialize graph builder.

        Args:
            semantic_threshold: Minimum cosine similarity for semantic edges (Layer 3)
            enable_cluster_edges: Whether to add cluster relationship edges (Layer 2)
            enable_link_edges: Whether to add Wikipedia link edges (Layer 1)
        """
        self.semantic_threshold = semantic_threshold
        self.enable_cluster_edges = enable_cluster_edges
        self.enable_link_edges = enable_link_edges
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

        # Create a set of valid article titles for fast lookup (case-insensitive)
        valid_titles_lower = {str(title).lower() for title in titles}
        title_lower_to_original: Dict[str, str] = {
            str(title).lower(): title for title in titles
        }

        # Layer 1: Link relationships
        if self.enable_link_edges:
            logger.info("=" * 80)
            logger.info("Adding Layer 1 edges (Wikipedia links)...")
            logger.info("=" * 80)
            layer1_start = perf_counter()
            layer1_count = self._add_link_edges(articles_df, valid_titles_lower, title_lower_to_original)
            layer1_time = perf_counter() - layer1_start
            logger.info("Added %d Layer 1 edges in %.2f seconds", layer1_count, layer1_time)
        else:
            logger.info("Skipping Layer 1 edges (disabled)")
            layer1_count = 0

        # Layer 2: Cluster relationships
        if self.enable_cluster_edges:
            logger.info("=" * 80)
            logger.info("Adding Layer 2 edges (cluster relationships)...")
            logger.info("=" * 80)
            layer2_start = perf_counter()
            layer2_count = self._add_cluster_edges(cluster_map, titles)
            layer2_time = perf_counter() - layer2_start
            logger.info("Added %d Layer 2 edges in %.2f seconds", layer2_count, layer2_time)
        else:
            logger.info("Skipping Layer 2 edges (disabled)")
            layer2_count = 0

        # Layer 3: Semantic relationships
        logger.info("=" * 80)
        logger.info("Adding Layer 3 edges (semantic similarity)...")
        logger.info("=" * 80)
        layer3_start = perf_counter()
        layer3_count = self._add_semantic_edges(embeddings, titles, title_to_idx)
        layer3_time = perf_counter() - layer3_start
        logger.info("Added %d Layer 3 edges in %.2f seconds", layer3_count, layer3_time)

        # Compute basic graph metrics
        total_edges = self.graph.number_of_edges()
        total_nodes = self.graph.number_of_nodes()
        logger.info(
            "Graph construction complete: %d nodes, %d edges (L1: %d, L2: %d, L3: %d)",
            total_nodes,
            total_edges,
            layer1_count if self.enable_link_edges else 0,
            layer2_count if self.enable_cluster_edges else 0,
            layer3_count,
        )

        return self.graph

    def _add_link_edges(
        self,
        articles_df: pd.DataFrame,
        valid_titles_lower: set,
        title_lower_to_original: Dict[str, str],
    ) -> int:
        """
        Add Layer 1 edges: Wikipedia article links.
        
        Creates directed edges from each article to the articles it links to,
        but only if both articles exist in the graph.
        
        Args:
            articles_df: DataFrame with columns: title, links
            valid_titles_lower: Set of lowercase article titles that exist in the graph
            title_lower_to_original: Mapping from lowercase to original title
            
        Returns:
            Number of edges added
        """
        edge_count = 0
        
        if "links" not in articles_df.columns:
            logger.warning("Articles DataFrame missing 'links' column. Skipping Layer 1 edges.")
            return 0
        
        logger.info("Processing Wikipedia links for %d articles...", len(articles_df))
        
        edges_to_add = []
        for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing links", unit="article"):
            source_title = str(row["title"])
            links = row.get("links", [])
            
            # Handle different link formats (list, array, etc.)
            if links is None:
                links = []
            elif isinstance(links, np.ndarray):
                # Handle numpy arrays - check size first to avoid ambiguous truth value
                if links.size == 0:
                    links = []
                else:
                    links = links.tolist()
            elif not isinstance(links, (list, tuple)):
                # Handle other types (pandas Series, strings, etc.)
                try:
                    # Check for NaN/None values safely
                    if pd.isna(links):
                        links = []
                    elif isinstance(links, str):
                        links = [links]
                    else:
                        # Try to convert to list (e.g., pandas Series)
                        links = list(links) if hasattr(links, '__iter__') else []
                except (ValueError, TypeError):
                    # Fallback: empty list if conversion fails
                    links = []
            
            # Process each link
            for link_title in links:
                if not link_title or pd.isna(link_title):
                    continue
                
                link_title_str = str(link_title).strip()
                if not link_title_str:
                    continue
                
                # Check if the linked article exists in our graph (case-insensitive)
                link_lower = link_title_str.lower()
                if link_lower in valid_titles_lower:
                    target_title = title_lower_to_original[link_lower]
                    
                    # Only add edge if it doesn't already exist
                    if not self.graph.has_edge(source_title, target_title):
                        edges_to_add.append((
                            source_title,
                            target_title,
                            {"layer": 1, "weight": 1.0, "type": "link"}
                        ))
        
        # Batch add all edges
        if edges_to_add:
            logger.info("Adding %d link edges to graph (batch operation)...", len(edges_to_add))
            add_start = perf_counter()
            self.graph.add_edges_from(edges_to_add)
            add_time = perf_counter() - add_start
            logger.info("  - Edges added in %.2f seconds", add_time)
            edge_count = len(edges_to_add)
        else:
            logger.info("No valid link edges to add")
        
        return edge_count

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
        total_clusters = len(cluster_to_titles)
        logger.info("Processing %d clusters for cluster edges...", total_clusters)
        
        for cluster_id, cluster_titles in tqdm(cluster_to_titles.items(), desc="Adding cluster edges", unit="cluster"):
            if len(cluster_titles) < 2:
                continue
            # Connect each article to all others in the same cluster
            for i, source in enumerate(cluster_titles):
                for target in cluster_titles[i + 1 :]:
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
        """
        Add Layer 3 edges: semantic similarity connections.
        
        Uses vectorized numpy operations for performance.
        """
        edge_count = 0

        # Handle empty embeddings
        if len(embeddings) == 0 or len(titles) == 0:
            return 0

        n = len(titles)
        
        # Compute pairwise cosine similarity (vectorized)
        logger.info("Computing pairwise cosine similarities for %d articles...", n)
        logger.info("  - Embeddings shape: %s", embeddings.shape)
        logger.info("  - Similarity matrix will be %dx%d", n, n)
        
        sim_start = perf_counter()
        similarity_matrix = cosine_similarity(embeddings)
        sim_time = perf_counter() - sim_start
        logger.info("  - Similarity matrix computed in %.2f seconds", sim_time)
        
        # Use numpy to find pairs above threshold (vectorized)
        # Only look at upper triangle to avoid duplicates
        logger.info("Finding high-similarity pairs (threshold=%.2f)...", self.semantic_threshold)
        
        # Get indices where similarity >= threshold (upper triangle only)
        upper_tri_indices = np.triu_indices(n, k=1)
        similarities = similarity_matrix[upper_tri_indices]
        
        # Find pairs above threshold
        filter_start = perf_counter()
        high_sim_mask = similarities >= self.semantic_threshold
        high_sim_i = upper_tri_indices[0][high_sim_mask]
        high_sim_j = upper_tri_indices[1][high_sim_mask]
        high_sim_values = similarities[high_sim_mask]
        filter_time = perf_counter() - filter_start
        logger.info("  - Filtered pairs in %.2f seconds", filter_time)
        logger.info("  - Found %d high-similarity pairs to add as edges", len(high_sim_i))
        
        # Batch add edges (much faster than individual adds)
        logger.info("Preparing edges for batch addition...")
        edges_to_add = []
        for idx in tqdm(range(len(high_sim_i)), desc="Preparing semantic edges", unit="pair"):
            i, j = high_sim_i[idx], high_sim_j[idx]
            source, target = titles[i], titles[j]
            sim = float(high_sim_values[idx])
            
            # Add bidirectional edges
            edges_to_add.append((source, target, {"layer": 3, "weight": sim, "type": "semantic"}))
            edges_to_add.append((target, source, {"layer": 3, "weight": sim, "type": "semantic"}))
        
        # Bulk add edges
        logger.info("Adding %d edges to graph (batch operation)...", len(edges_to_add))
        add_start = perf_counter()
        self.graph.add_edges_from(edges_to_add)
        add_time = perf_counter() - add_start
        logger.info("  - Edges added in %.2f seconds", add_time)
        edge_count = len(edges_to_add)
        
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

