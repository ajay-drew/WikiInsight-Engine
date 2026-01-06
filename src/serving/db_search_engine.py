"""
PostgreSQL + pgvector backed search engine.

Provides hybrid search using:
- pgvector for semantic similarity search
- PostgreSQL full-text search for keyword matching
- Reciprocal Rank Fusion (RRF) to combine results
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.database.connection import DatabaseManager, get_db_manager
from src.database.repository import ArticleRepository

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for a single search result."""
    title: str
    score: float
    rank: int
    cluster_id: Optional[int] = None
    snippet: Optional[str] = None


class DatabaseSearchEngine:
    """
    PostgreSQL-backed hybrid search engine.
    
    Combines:
    - pgvector cosine similarity search
    - PostgreSQL full-text search
    - Reciprocal Rank Fusion (RRF) for result merging
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_model=None,
        rrf_k: int = 60,
    ):
        """
        Initialize database search engine.
        
        Args:
            db_manager: Database manager instance (uses global if None)
            embedding_model: Model with .encode() method for query embedding
            rrf_k: RRF constant (default 60)
        """
        self.db_manager = db_manager or get_db_manager()
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        if not self._initialized:
            await self.db_manager.initialize()
            self._initialized = True
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        cluster_id: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            cluster_id: Optional cluster filter
            
        Returns:
            List of SearchResult objects
        """
        await self.initialize()
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            keyword_weight /= total_weight
        
        # Get semantic results
        semantic_results = []
        if semantic_weight > 0 and self.embedding_model is not None:
            semantic_results = await self._semantic_search(query, top_k * 2, cluster_id)
        
        # Get keyword results
        keyword_results = []
        if keyword_weight > 0:
            keyword_results = await self._keyword_search(query, top_k * 2)
        
        # Combine with RRF
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight,
        )
        
        # Return top_k results
        return combined[:top_k]
    
    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        cluster_id: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Perform semantic search using pgvector."""
        if self.embedding_model is None:
            return []
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_model.encode(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
        except Exception as e:
            logger.error("Failed to encode query: %s", e)
            return []
        
        # Search in database
        async with self.db_manager.session() as session:
            repo = ArticleRepository(session)
            results = await repo.search_similar(
                query_embedding,
                top_k=top_k,
                cluster_id=cluster_id,
            )
        
        return [(article.title, score) for article, score in results]
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Perform full-text keyword search."""
        async with self.db_manager.session() as session:
            repo = ArticleRepository(session)
            results = await repo.full_text_search(query, top_k=top_k)
        
        return [(article.title, score) for article, score in results]
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
        semantic_weight: float,
        keyword_weight: float,
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(weight / (k + rank)) for each ranking
        """
        scores: Dict[str, float] = {}
        
        # Process semantic results
        for rank, (title, sim_score) in enumerate(semantic_results, start=1):
            rrf_score = semantic_weight / (self.rrf_k + rank)
            scores[title] = scores.get(title, 0) + rrf_score
        
        # Process keyword results
        for rank, (title, bm25_score) in enumerate(keyword_results, start=1):
            rrf_score = keyword_weight / (self.rrf_k + rank)
            scores[title] = scores.get(title, 0) + rrf_score
        
        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to SearchResult objects
        return [
            SearchResult(title=title, score=score, rank=rank)
            for rank, (title, score) in enumerate(sorted_results, start=1)
        ]
    
    async def get_article_count(self) -> int:
        """Get total number of articles in database."""
        await self.initialize()
        async with self.db_manager.session() as session:
            repo = ArticleRepository(session)
            return await repo.count()


async def create_db_search_engine(
    embedding_model=None,
) -> DatabaseSearchEngine:
    """
    Factory function to create and initialize a database search engine.
    
    Args:
        embedding_model: Model with .encode() method
        
    Returns:
        Initialized DatabaseSearchEngine
    """
    engine = DatabaseSearchEngine(embedding_model=embedding_model)
    await engine.initialize()
    return engine

