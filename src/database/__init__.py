"""
Database module for WikiInsight Engine.

Provides PostgreSQL + pgvector integration for:
- Article storage with vector embeddings
- Cluster metadata
- Graph edges
- Efficient similarity search
"""

from .models import Article, Cluster, GraphEdge, Base
from .connection import DatabaseManager, get_db_manager
from .repository import ArticleRepository, ClusterRepository, GraphRepository

__all__ = [
    "Article",
    "Cluster", 
    "GraphEdge",
    "Base",
    "DatabaseManager",
    "get_db_manager",
    "ArticleRepository",
    "ClusterRepository",
    "GraphRepository",
]

