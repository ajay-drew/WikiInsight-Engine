"""
Graph service for querying the knowledge graph.

Provides methods to query the graph for neighbors, paths, subgraphs,
and convert to frontend-friendly formats.
"""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
from .knowledge_graph import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

GRAPH_PATH = "data/graph/knowledge_graph.pkl"


class GraphService:
    """Service for querying the knowledge graph."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize graph service.

        Args:
            graph: NetworkX graph (if None, will load from default path)
        """
        if graph is not None:
            self.graph = graph
        else:
            self.graph = None
            self._load_graph()

    def _load_graph(self) -> None:
        """Load graph from default path."""
        try:
            self.graph = KnowledgeGraphBuilder.load_graph(GRAPH_PATH)
            logger.info("Graph service initialized with %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        except FileNotFoundError:
            logger.warning("Graph file not found at %s. Graph queries will be unavailable.", GRAPH_PATH)
            self.graph = None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load graph: %s", exc)
            self.graph = None

    def get_neighbors(
        self,
        article_title: str,
        max_neighbors: int = 20,
        layer_filter: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Get graph neighbors of an article.

        Args:
            article_title: Article title
            max_neighbors: Maximum number of neighbors to return
            layer_filter: Optional list of layers to include (2, 3)

        Returns:
            List of neighbor dictionaries with title and edge metadata
        """
        if self.graph is None:
            return []

        if article_title not in self.graph:
            logger.warning("Article '%s' not found in graph", article_title)
            return []

        neighbors = []
        for target in self.graph.successors(article_title):
            edge_data = self.graph[article_title][target]
            layer = edge_data.get("layer", 0)
            edge_type = edge_data.get("type", "unknown")
            weight = edge_data.get("weight", 1.0)

            # Apply layer filter if specified
            if layer_filter is not None and layer not in layer_filter:
                continue

            neighbors.append(
                {
                    "title": target,
                    "layer": layer,
                    "type": edge_type,
                    "weight": weight,
                }
            )

        # Sort by weight (descending) and limit
        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        return neighbors[:max_neighbors]

    def find_path(
        self,
        from_title: str,
        to_title: str,
        max_path_length: int = 5,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two articles.

        Args:
            from_title: Source article title
            to_title: Target article title
            max_path_length: Maximum path length to consider

        Returns:
            List of article titles forming the path, or None if no path found
        """
        if self.graph is None:
            return None

        if from_title not in self.graph or to_title not in self.graph:
            return None

        try:
            path = nx.shortest_path(self.graph, from_title, to_title)
            if len(path) > max_path_length:
                return None
            return path
        except nx.NetworkXNoPath:
            return None

    def get_cluster_subgraph(
        self,
        cluster_id: int,
        cluster_assignments: Dict[str, int],
        max_nodes: int = 100,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get subgraph for a specific cluster.

        Args:
            cluster_id: Cluster ID
            cluster_assignments: Dictionary mapping article titles to cluster IDs
            max_nodes: Maximum number of nodes to include

        Returns:
            Tuple of (nodes, edges) in frontend-friendly format
        """
        if self.graph is None:
            return [], []

        # Find all articles in this cluster (match by lowercase)
        cluster_articles = []
        for node_title in self.graph.nodes():
            if node_title.lower() in cluster_assignments:
                if cluster_assignments[node_title.lower()] == cluster_id:
                    cluster_articles.append(node_title)

        if len(cluster_articles) > max_nodes:
            # Sample articles if cluster is too large
            import random

            random.seed(42)
            cluster_articles = random.sample(cluster_articles, max_nodes)
            logger.info("Sampled %d articles from cluster %d (original size: %d)", max_nodes, cluster_id, len(cluster_articles))

        # Build subgraph
        # Start with cluster articles
        subgraph_nodes = set(cluster_articles)
        
        # For Layer 3 (semantic), include connected articles
        # even if they're outside the cluster (but limit to prevent huge graphs)
        connected_articles = set()
        for article in cluster_articles:
            if article in self.graph:
                for neighbor in self.graph.successors(article):
                    edge_data = self.graph[article][neighbor]
                    layer = edge_data.get("layer", 0)
                    # Include neighbors for Layer 3 edges
                    # (Layer 2 edges are already within cluster)
                    if layer == 3:
                        connected_articles.add(neighbor)
        
        # Limit external connections to prevent huge graphs
        # Include up to 50 additional nodes from Layer 3 connections
        if len(connected_articles) > 50:
            import random
            random.seed(42)
            connected_articles = set(random.sample(list(connected_articles), 50))
        
        # Add connected articles to subgraph
        subgraph_nodes.update(connected_articles)

        # Convert to frontend format
        nodes = []
        for title in subgraph_nodes:
            node_data = self.graph.nodes[title]
            # Get cluster_id for external nodes
            cluster_id = node_data.get("cluster_id", -1)
            if cluster_id == -1 and title.lower() in cluster_assignments:
                cluster_id = cluster_assignments[title.lower()]
            
            nodes.append(
                {
                    "id": title,
                    "label": title[:50] + "..." if len(title) > 50 else title,
                    "cluster_id": cluster_id,
                }
            )

        edges = []
        cluster_articles_set = set(cluster_articles)
        for source in subgraph_nodes:
            if source not in self.graph:
                continue
            for target in self.graph.successors(source):
                if target in subgraph_nodes:
                    edge_data = self.graph[source][target]
                    layer = edge_data.get("layer", 0)
                    
                    # Include all edges where source is in cluster
                    # This ensures Layer 3 edges are included
                    if source in cluster_articles_set:
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "layer": int(layer),  # Ensure layer is an integer
                                "weight": float(edge_data.get("weight", 1.0)),
                                "type": str(edge_data.get("type", "unknown")),
                            }
                        )
                    # Also include Layer 2 edges (cluster relationships) where both are in cluster
                    elif layer == 2 and source in cluster_articles_set and target in cluster_articles_set:
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "layer": int(layer),
                                "weight": float(edge_data.get("weight", 1.0)),
                                "type": str(edge_data.get("type", "unknown")),
                            }
                        )

        return nodes, edges

    def get_article_graph(
        self,
        article_title: str,
        max_neighbors: int = 20,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get graph centered on a specific article.

        Args:
            article_title: Center article title
            max_neighbors: Maximum number of neighbors to include

        Returns:
            Tuple of (nodes, edges) in frontend-friendly format
        """
        if self.graph is None:
            return [], []

        if article_title not in self.graph:
            return [], []

        # Get neighbors
        neighbors = self.get_neighbors(article_title, max_neighbors=max_neighbors)
        neighbor_titles = [n["title"] for n in neighbors]

        # Build node set
        nodes_set = {article_title} | set(neighbor_titles)

        # Convert to frontend format
        nodes = []
        for title in nodes_set:
            node_data = self.graph.nodes[title]
            nodes.append(
                {
                    "id": title,
                    "label": title[:50] + "..." if len(title) > 50 else title,
                    "cluster_id": node_data.get("cluster_id", -1),
                }
            )

        # Add center node first
        center_node = {
            "id": article_title,
            "label": article_title[:50] + "..." if len(article_title) > 50 else article_title,
            "cluster_id": self.graph.nodes[article_title].get("cluster_id", -1),
        }
        if center_node not in nodes:
            nodes.insert(0, center_node)

        # Get edges
        edges = []
        for source in nodes_set:
            if source not in self.graph:
                continue
            for target in self.graph.successors(source):
                if target in nodes_set:
                    edge_data = self.graph[source][target]
                    edges.append(
                        {
                            "source": source,
                            "target": target,
                            "layer": edge_data.get("layer", 0),
                            "weight": edge_data.get("weight", 1.0),
                            "type": edge_data.get("type", "unknown"),
                        }
                    )

        return nodes, edges

    def to_visualization_format(
        self,
        nodes: List[str],
        max_nodes: int = 100,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert graph nodes to frontend visualization format.

        Args:
            nodes: List of article titles to include
            max_nodes: Maximum number of nodes

        Returns:
            Tuple of (nodes, edges) in frontend format
        """
        if self.graph is None:
            return [], []

        # Limit nodes if needed
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]

        # Build node list
        viz_nodes = []
        for title in nodes:
            if title not in self.graph:
                continue
            node_data = self.graph.nodes[title]
            viz_nodes.append(
                {
                    "id": title,
                    "label": title[:50] + "..." if len(title) > 50 else title,
                    "cluster_id": node_data.get("cluster_id", -1),
                }
            )

        # Build edge list (only edges between selected nodes)
        viz_edges = []
        nodes_set = set(nodes)
        for source in nodes_set:
            if source not in self.graph:
                continue
            for target in self.graph.successors(source):
                if target in nodes_set:
                    edge_data = self.graph[source][target]
                    viz_edges.append(
                        {
                            "source": source,
                            "target": target,
                            "layer": edge_data.get("layer", 0),
                            "weight": edge_data.get("weight", 1.0),
                            "type": edge_data.get("type", "unknown"),
                        }
                    )

        return viz_nodes, viz_edges

