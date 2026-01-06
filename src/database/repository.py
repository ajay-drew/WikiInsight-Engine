"""
Data access layer for WikiInsight Engine.

Provides repository classes for CRUD operations on articles,
clusters, and graph edges with pgvector similarity search.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Article, Cluster, GraphEdge, PipelineRun

logger = logging.getLogger(__name__)


class ArticleRepository:
    """Repository for article CRUD operations and similarity search."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, article_data: Dict[str, Any]) -> Article:
        """Create a new article."""
        article = Article(**article_data)
        self.session.add(article)
        await self.session.flush()
        return article
    
    async def create_batch(self, articles_data: List[Dict[str, Any]]) -> List[Article]:
        """Create multiple articles in batch."""
        articles = [Article(**data) for data in articles_data]
        self.session.add_all(articles)
        await self.session.flush()
        return articles
    
    async def get_by_id(self, article_id: int) -> Optional[Article]:
        """Get article by ID."""
        result = await self.session.execute(
            select(Article).where(Article.id == article_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_title(self, title: str) -> Optional[Article]:
        """Get article by title."""
        result = await self.session.execute(
            select(Article).where(Article.title == title)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 1000, offset: int = 0) -> List[Article]:
        """Get all articles with pagination."""
        result = await self.session.execute(
            select(Article).limit(limit).offset(offset)
        )
        return list(result.scalars().all())
    
    async def get_by_cluster(self, cluster_id: int) -> List[Article]:
        """Get all articles in a cluster."""
        result = await self.session.execute(
            select(Article).where(Article.cluster_id == cluster_id)
        )
        return list(result.scalars().all())
    
    async def update(self, article_id: int, data: Dict[str, Any]) -> Optional[Article]:
        """Update an article."""
        await self.session.execute(
            update(Article).where(Article.id == article_id).values(**data)
        )
        return await self.get_by_id(article_id)
    
    async def update_embedding(self, title: str, embedding: List[float]) -> bool:
        """Update article embedding by title."""
        result = await self.session.execute(
            update(Article)
            .where(Article.title == title)
            .values(embedding=embedding)
        )
        return result.rowcount > 0
    
    async def update_cluster_assignment(self, title: str, cluster_id: int) -> bool:
        """Update article cluster assignment."""
        result = await self.session.execute(
            update(Article)
            .where(Article.title == title)
            .values(cluster_id=cluster_id)
        )
        return result.rowcount > 0
    
    async def delete(self, article_id: int) -> bool:
        """Delete an article."""
        result = await self.session.execute(
            delete(Article).where(Article.id == article_id)
        )
        return result.rowcount > 0
    
    async def delete_all(self) -> int:
        """Delete all articles. Returns count of deleted rows."""
        result = await self.session.execute(delete(Article))
        return result.rowcount
    
    async def count(self) -> int:
        """Count total articles."""
        result = await self.session.execute(select(func.count(Article.id)))
        return result.scalar() or 0
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        cluster_id: Optional[int] = None,
    ) -> List[Tuple[Article, float]]:
        """
        Search for similar articles using pgvector cosine similarity.
        
        Args:
            query_embedding: Query embedding vector (384 dimensions)
            top_k: Number of results to return
            cluster_id: Optional cluster filter
            
        Returns:
            List of (article, similarity_score) tuples
        """
        # Build query with pgvector cosine distance
        # Note: pgvector uses <=> for cosine distance (1 - similarity)
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        if cluster_id is not None:
            query = text("""
                SELECT id, title, cleaned_text, cluster_id,
                       1 - (embedding <=> :embedding::vector) AS similarity
                FROM articles
                WHERE embedding IS NOT NULL AND cluster_id = :cluster_id
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """)
            result = await self.session.execute(
                query,
                {"embedding": embedding_str, "cluster_id": cluster_id, "limit": top_k}
            )
        else:
            query = text("""
                SELECT id, title, cleaned_text, cluster_id,
                       1 - (embedding <=> :embedding::vector) AS similarity
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """)
            result = await self.session.execute(
                query,
                {"embedding": embedding_str, "limit": top_k}
            )
        
        rows = result.fetchall()
        
        # Convert to Article objects with similarity scores
        results = []
        for row in rows:
            article = Article(
                id=row.id,
                title=row.title,
                cleaned_text=row.cleaned_text,
                cluster_id=row.cluster_id,
            )
            results.append((article, float(row.similarity)))
        
        return results
    
    async def full_text_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Article, float]]:
        """
        Full-text search using PostgreSQL tsvector.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (article, rank) tuples
        """
        search_query = text("""
            SELECT id, title, cleaned_text, cluster_id,
                   ts_rank(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(cleaned_text, '')),
                          plainto_tsquery('english', :query)) AS rank
            FROM articles
            WHERE to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(cleaned_text, ''))
                  @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            search_query,
            {"query": query, "limit": top_k}
        )
        
        rows = result.fetchall()
        results = []
        for row in rows:
            article = Article(
                id=row.id,
                title=row.title,
                cleaned_text=row.cleaned_text,
                cluster_id=row.cluster_id,
            )
            results.append((article, float(row.rank)))
        
        return results


class ClusterRepository:
    """Repository for cluster CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, cluster_data: Dict[str, Any]) -> Cluster:
        """Create a new cluster."""
        cluster = Cluster(**cluster_data)
        self.session.add(cluster)
        await self.session.flush()
        return cluster
    
    async def create_batch(self, clusters_data: List[Dict[str, Any]]) -> List[Cluster]:
        """Create multiple clusters in batch."""
        clusters = [Cluster(**data) for data in clusters_data]
        self.session.add_all(clusters)
        await self.session.flush()
        return clusters
    
    async def get_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        result = await self.session.execute(
            select(Cluster).where(Cluster.id == cluster_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self) -> List[Cluster]:
        """Get all clusters."""
        result = await self.session.execute(select(Cluster))
        return list(result.scalars().all())
    
    async def update(self, cluster_id: int, data: Dict[str, Any]) -> Optional[Cluster]:
        """Update a cluster."""
        await self.session.execute(
            update(Cluster).where(Cluster.id == cluster_id).values(**data)
        )
        return await self.get_by_id(cluster_id)
    
    async def delete(self, cluster_id: int) -> bool:
        """Delete a cluster."""
        result = await self.session.execute(
            delete(Cluster).where(Cluster.id == cluster_id)
        )
        return result.rowcount > 0
    
    async def delete_all(self) -> int:
        """Delete all clusters. Returns count of deleted rows."""
        result = await self.session.execute(delete(Cluster))
        return result.rowcount
    
    async def count(self) -> int:
        """Count total clusters."""
        result = await self.session.execute(select(func.count(Cluster.id)))
        return result.scalar() or 0
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[Cluster, float]]:
        """Search for similar clusters by centroid."""
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        query = text("""
            SELECT id, size, keywords, top_articles,
                   1 - (centroid <=> :embedding::vector) AS similarity
            FROM clusters
            WHERE centroid IS NOT NULL
            ORDER BY centroid <=> :embedding::vector
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            query,
            {"embedding": embedding_str, "limit": top_k}
        )
        
        rows = result.fetchall()
        results = []
        for row in rows:
            cluster = Cluster(
                id=row.id,
                size=row.size,
                keywords=row.keywords,
                top_articles=row.top_articles,
            )
            results.append((cluster, float(row.similarity)))
        
        return results


class GraphRepository:
    """Repository for graph edge CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, edge_data: Dict[str, Any]) -> GraphEdge:
        """Create a new edge."""
        edge = GraphEdge(**edge_data)
        self.session.add(edge)
        await self.session.flush()
        return edge
    
    async def create_batch(self, edges_data: List[Dict[str, Any]]) -> int:
        """Create multiple edges in batch. Returns count of created edges."""
        edges = [GraphEdge(**data) for data in edges_data]
        self.session.add_all(edges)
        await self.session.flush()
        return len(edges)
    
    async def get_neighbors(
        self,
        title: str,
        layer: Optional[str] = None,
        limit: int = 50,
    ) -> List[GraphEdge]:
        """Get neighboring edges for an article."""
        query = select(GraphEdge).where(
            (GraphEdge.source_title == title) | (GraphEdge.target_title == title)
        )
        
        if layer:
            query = query.where(GraphEdge.layer == layer)
        
        query = query.limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_edges_by_layer(self, layer: str) -> List[GraphEdge]:
        """Get all edges of a specific layer."""
        result = await self.session.execute(
            select(GraphEdge).where(GraphEdge.layer == layer)
        )
        return list(result.scalars().all())
    
    async def delete_all(self) -> int:
        """Delete all edges. Returns count of deleted rows."""
        result = await self.session.execute(delete(GraphEdge))
        return result.rowcount
    
    async def count(self) -> int:
        """Count total edges."""
        result = await self.session.execute(select(func.count(GraphEdge.id)))
        return result.scalar() or 0
    
    async def count_by_layer(self) -> Dict[str, int]:
        """Count edges by layer."""
        result = await self.session.execute(
            select(GraphEdge.layer, func.count(GraphEdge.id))
            .group_by(GraphEdge.layer)
        )
        return {row[0]: row[1] for row in result.fetchall()}


class PipelineRunRepository:
    """Repository for pipeline run tracking."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, run_data: Dict[str, Any]) -> PipelineRun:
        """Create a new pipeline run record."""
        run = PipelineRun(**run_data)
        self.session.add(run)
        await self.session.flush()
        return run
    
    async def get_by_run_id(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run by run_id."""
        result = await self.session.execute(
            select(PipelineRun).where(PipelineRun.run_id == run_id)
        )
        return result.scalar_one_or_none()
    
    async def update(self, run_id: str, data: Dict[str, Any]) -> Optional[PipelineRun]:
        """Update a pipeline run."""
        await self.session.execute(
            update(PipelineRun).where(PipelineRun.run_id == run_id).values(**data)
        )
        return await self.get_by_run_id(run_id)
    
    async def get_recent(self, limit: int = 10) -> List[PipelineRun]:
        """Get recent pipeline runs."""
        result = await self.session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

