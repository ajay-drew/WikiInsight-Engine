"""
SQLAlchemy models for WikiInsight Engine database.

Uses pgvector for efficient vector similarity search on embeddings.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.postgresql import ARRAY

# pgvector support
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    # Fallback for environments without pgvector
    Vector = None
    PGVECTOR_AVAILABLE = False


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Article(Base):
    """
    Wikipedia article with embedding vector.
    
    Stores article content, metadata, and 384-dimensional embedding
    for semantic similarity search using pgvector.
    """
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), unique=True, nullable=False, index=True)
    raw_text = Column(Text, nullable=True)
    cleaned_text = Column(Text, nullable=True)
    nltk_cleaned_text = Column(Text, nullable=True)
    
    # Metadata
    categories = Column(ARRAY(String), default=[])
    links = Column(ARRAY(String), default=[])
    word_count = Column(Integer, default=0)
    
    # Embedding vector (384 dimensions for all-MiniLM-L6-v2)
    # Using pgvector's Vector type for efficient similarity search
    embedding = Column(Vector(384) if PGVECTOR_AVAILABLE else Text, nullable=True)
    
    # Cluster assignment
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="articles")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_articles_cluster_id", "cluster_id"),
        Index("idx_articles_title_trgm", "title", postgresql_using="gin",
              postgresql_ops={"title": "gin_trgm_ops"}),
    )
    
    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:50]}...')>"


class Cluster(Base):
    """
    Topic cluster with centroid embedding.
    
    Stores cluster metadata, keywords, and centroid vector
    for cluster-level similarity operations.
    """
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Cluster metadata
    size = Column(Integer, default=0)
    keywords = Column(ARRAY(String), default=[])
    top_articles = Column(ARRAY(String), default=[])
    
    # Centroid embedding (384 dimensions)
    centroid = Column(Vector(384) if PGVECTOR_AVAILABLE else Text, nullable=True)
    
    # Quality metrics
    silhouette_score = Column(Float, nullable=True)
    cohesion = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    articles = relationship("Article", back_populates="cluster")
    
    def __repr__(self):
        return f"<Cluster(id={self.id}, size={self.size})>"


class GraphEdge(Base):
    """
    Knowledge graph edge connecting two articles.
    
    Stores edge metadata including layer type (link, cluster, semantic)
    and weight for graph traversal operations.
    """
    __tablename__ = "graph_edges"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Edge endpoints (article titles for flexibility)
    source_title = Column(String(500), nullable=False, index=True)
    target_title = Column(String(500), nullable=False, index=True)
    
    # Edge properties
    layer = Column(String(50), nullable=False)  # 'link', 'cluster', 'semantic'
    weight = Column(Float, default=1.0)
    
    # Additional edge metadata (renamed from 'metadata' to avoid SQLAlchemy reserved word)
    edge_metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for graph queries
    __table_args__ = (
        Index("idx_edges_source", "source_title"),
        Index("idx_edges_target", "target_title"),
        Index("idx_edges_layer", "layer"),
        Index("idx_edges_source_target", "source_title", "target_title"),
    )
    
    def __repr__(self):
        return f"<GraphEdge(source='{self.source_title}', target='{self.target_title}', layer='{self.layer}')>"


class PipelineRun(Base):
    """
    Pipeline execution record for tracking runs.
    
    Stores metadata about each pipeline execution for
    monitoring and debugging purposes.
    """
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Run metadata
    run_id = Column(String(100), unique=True, nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    
    # Configuration snapshot
    config_snapshot = Column(JSON, default={})
    
    # Metrics
    articles_ingested = Column(Integer, default=0)
    clusters_created = Column(Integer, default=0)
    edges_created = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<PipelineRun(id={self.run_id}, status='{self.status}')>"

