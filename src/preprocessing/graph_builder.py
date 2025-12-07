"""
Graph construction for network analysis (article similarity, citation networks).
"""

import networkx as nx
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build graphs from Wikipedia data."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph = nx.DiGraph()
    
    def build_citation_graph(self, articles: List[Dict]) -> nx.DiGraph:
        """
        Build citation graph from articles.
        
        Args:
            articles: List of article dictionaries with links
            
        Returns:
            Directed graph
        """
        graph = nx.DiGraph()
        
        for article in articles:
            title = article.get("title")
            links = article.get("links", [])
            
            graph.add_node(title, **article)
            
            for link in links:
                if isinstance(link, dict):
                    link_title = link.get("title", link)
                else:
                    link_title = link
                graph.add_edge(title, link_title)
        
        logger.info(f"Built citation graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def build_similarity_graph(self, embeddings: Dict[str, List[float]], threshold: float = 0.7) -> nx.Graph:
        """
        Build similarity graph from embeddings.
        
        Args:
            embeddings: Dictionary of title -> embedding vector
            threshold: Similarity threshold for edges
            
        Returns:
            Undirected graph
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        graph = nx.Graph()
        titles = list(embeddings.keys())
        vectors = np.array([embeddings[title] for title in titles])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Add edges above threshold
        for i, title1 in enumerate(titles):
            graph.add_node(title1)
            for j, title2 in enumerate(titles[i+1:], start=i+1):
                if similarity_matrix[i][j] >= threshold:
                    graph.add_edge(title1, title2, weight=similarity_matrix[i][j])
        
        logger.info(f"Built similarity graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph

