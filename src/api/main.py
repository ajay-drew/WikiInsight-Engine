"""
FastAPI main application for the WikiInsight Engine.

Exposes endpoints for:
- Hybrid search (semantic + keyword search with RRF).
- Health and metadata.
- Topic cluster lookup for a given article title.
- Cluster-level explanations/summaries.
"""

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.common.logging_utils import setup_logging
from src.modeling.topic_index import TopicIndex
from src.serving.search_engine import HybridSearchEngine
from src.preprocessing.embeddings import EmbeddingGenerator

setup_logging()
logger = logging.getLogger(__name__)

_topic_index: Optional[TopicIndex] = None
_search_engine: Optional[HybridSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global _topic_index, _search_engine
    # Startup
    try:
        _topic_index = TopicIndex.load_default()
        logger.info("Topic index loaded successfully")
    except FileNotFoundError as exc:
        # Provide helpful message for missing artifacts
        logger.warning(
            "Topic index artifacts not found. API will run in degraded mode.\n"
            "To generate the required artifacts, run:\n"
            "  dvc repro\n"
            "  OR\n"
            "  python -m src.ingestion.fetch_wikipedia_data\n"
            "  python -m src.preprocessing.process_data\n"
            "  python -m src.modeling.cluster_topics\n"
            f"Original error: {exc}"
        )
        _topic_index = None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load topic index: %s", exc)
        _topic_index = None

    # Load HybridSearchEngine
    try:
        import os
        import pandas as pd
        import numpy as np
        import yaml

        EMBEDDINGS_PATH = os.path.join("data", "features", "embeddings.parquet")
        CLEANED_ARTICLES_PATH = os.path.join("data", "processed", "cleaned_articles.parquet")
        CONFIG_PATH = "config.yaml"

        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CLEANED_ARTICLES_PATH):
            # Load articles
            cleaned_df = pd.read_parquet(CLEANED_ARTICLES_PATH)
            articles = [
                {
                    "title": row["title"],
                    "text": row.get("cleaned_text", row.get("raw_text", "")),
                }
                for _, row in cleaned_df.iterrows()
            ]

            # Load embeddings
            emb_df = pd.read_parquet(EMBEDDINGS_PATH)
            embeddings = np.vstack(emb_df["embedding"].to_list())

            # Load embedding model
            config = yaml.safe_load(open(CONFIG_PATH, "r"))
            model_name = config.get("preprocessing", {}).get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
            embedding_model = EmbeddingGenerator(model_name=model_name)

            _search_engine = HybridSearchEngine(
                articles=articles,
                embeddings=embeddings,
                model=embedding_model.model,
            )
            logger.info("HybridSearchEngine loaded successfully with %d articles", len(articles))
        else:
            logger.warning(
                "Search engine artifacts not found. Search endpoint will be unavailable.\n"
                "To generate the required artifacts, run:\n"
                "  dvc repro"
            )
            _search_engine = None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load HybridSearchEngine: %s", exc)
        _search_engine = None

    yield
    # Shutdown (if needed)
    _topic_index = None
    _search_engine = None


app = FastAPI(
    title="WikiInsight Engine API",
    description="API for unsupervised Wikipedia topic clustering and exploration",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopicQueryRequest(BaseModel):
    """Request model for topic/cluster exploration."""

    article_title: str
    # Reserved for potential future use (e.g., providing raw text for unseen articles)
    features: Optional[dict] = None


class TopicClusterResponse(BaseModel):
    """Response model for topic cluster lookup."""

    article_title: str
    cluster_id: Optional[int] = None
    similar_articles: List[str] = []
    keywords: Optional[List[str]] = None
    explanation: Optional[dict] = None  # Optional cluster summary / importance-style info


class ClusterSummary(BaseModel):
    """Summary of a single topic cluster."""

    cluster_id: int
    size: int
    keywords: List[str] = []
    top_articles: List[str] = []


class SearchRequest(BaseModel):
    """Request model for hybrid search."""

    query: str
    top_k: int = 10


class SearchResultResponse(BaseModel):
    """Response model for search results."""

    title: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    query: str
    results: List[SearchResultResponse]
    total_results: int


_topic_index: Optional[TopicIndex] = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses with latency for easier debugging."""
    start = perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        logger.info(
            "HTTP %s %s -> %s in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception:  # noqa: BLE001
        duration_ms = (perf_counter() - start) * 1000
        logger.exception(
            "Unhandled error during request %s %s after %.2fms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "WikiInsight Engine API", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    status = "degraded" if _topic_index is None else "healthy"
    return {"status": status}


def _ensure_index_available() -> TopicIndex:
    if _topic_index is None:
        raise HTTPException(
            status_code=503,
            detail="Topic index is not available. Run the clustering pipeline first.",
        )
    return _topic_index


@app.post("/topics/lookup", response_model=TopicClusterResponse)
@limiter.limit("100/minute")
async def lookup_topic_cluster(request: Request, body: TopicQueryRequest):
    """
    Look up the topic cluster for an article and return similar articles.
    
    Args:
        request: FastAPI request object (for rate limiting)
        body: Topic/cluster lookup request
        
    Returns:
        Topic cluster information with similar articles and optional explanation
    """
    try:
        index = _ensure_index_available()
        result = index.lookup(body.article_title)
        return TopicClusterResponse(
            article_title=result.article_title,
            cluster_id=result.cluster_id,
            similar_articles=result.similar_articles,
            keywords=result.keywords,
            explanation=result.explanation,
        )
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Article '{body.article_title}' not found in topic index.",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Error looking up topic cluster for '%s'", body.article_title)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/explain/{article_title}")
@limiter.limit("100/minute")
async def explain_prediction(request: Request, article_title: str):
    """
    Get explanation/summary for an article's topic cluster.
    
    Args:
        article_title: Article title
        
    Returns:
        Explanation with feature importance / cluster summary
    """
    try:
        index = _ensure_index_available()
        idx_result = index.lookup(article_title)
        summary = index.get_cluster_summary(idx_result.cluster_id)  # type: ignore[arg-type]
        explanation = {
            "cluster_id": summary["cluster_id"],
            "cluster_size": summary["size"],
            "keywords": summary["keywords"],
            "top_articles": summary["top_articles"],
        }
        return {"article_title": idx_result.article_title, "explanation": explanation}
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Article '{article_title}' not found in topic index.",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Error explaining topic cluster for '%s'", article_title)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/clusters/overview", response_model=List[ClusterSummary])
@limiter.limit("60/minute")
async def clusters_overview(request: Request):
    """
    Return a list of all clusters with basic summary info.
    """
    try:
        index = _ensure_index_available()
        overview_df = index.get_clusters_overview()
        summaries: List[ClusterSummary] = []
        for _, row in overview_df.iterrows():
            cid = int(row["cluster_id"])
            full = index.get_cluster_summary(cid)
            summaries.append(
                ClusterSummary(
                    cluster_id=cid,
                    size=int(full.get("size", 0)),
                    keywords=list(full.get("keywords", [])),
                    top_articles=list(full.get("top_articles", [])),
                )
            )
        return summaries
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Error fetching clusters overview")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/clusters/{cluster_id}", response_model=ClusterSummary)
@limiter.limit("100/minute")
async def cluster_detail(request: Request, cluster_id: int):
    """
    Get detailed summary for a single cluster.
    """
    try:
        index = _ensure_index_available()
        summary = index.get_cluster_summary(cluster_id)
        return ClusterSummary(
            cluster_id=int(summary.get("cluster_id", cluster_id)),
            size=int(summary.get("size", 0)),
            keywords=list(summary.get("keywords", [])),
            top_articles=list(summary.get("top_articles", [])),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Error fetching cluster %s", cluster_id)
        raise HTTPException(status_code=500, detail="Internal server error")


# Convenience /api-prefixed routes for the React frontend


@app.post("/api/topics/lookup", response_model=TopicClusterResponse)
@limiter.limit("100/minute")
async def api_lookup_topic_cluster(request: Request, body: TopicQueryRequest):
    """API-prefixed alias for /topics/lookup endpoint."""
    return await lookup_topic_cluster(request, body)


@app.get("/api/clusters/overview", response_model=List[ClusterSummary])
@limiter.limit("60/minute")
async def api_clusters_overview(request: Request):
    """API-prefixed alias for /clusters/overview endpoint."""
    return await clusters_overview(request)


@app.get("/api/clusters/{cluster_id}", response_model=ClusterSummary)
@limiter.limit("100/minute")
async def api_cluster_detail(request: Request, cluster_id: int):
    """API-prefixed alias for /clusters/{cluster_id} endpoint."""
    return await cluster_detail(request, cluster_id)


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("100/minute")
async def api_search(request: Request, body: SearchRequest):
    """
    Hybrid search endpoint combining semantic (vector) and keyword (BM25) search.
    
    Args:
        request: FastAPI request object (for rate limiting)
        body: Search request with query and optional top_k
        
    Returns:
        Search results with titles, scores, and ranks
    """
    global _search_engine
    if _search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine is not available. Run the data pipeline first.",
        )

    try:
        query = body.query.strip()
        top_k = max(1, min(body.top_k, 50))  # Clamp between 1 and 50

        if not query:
            return SearchResponse(
                query=body.query,
                results=[],
                total_results=0,
            )

        search_results = _search_engine.search(query, top_k=top_k)
        
        results = [
            SearchResultResponse(
                title=result.title,
                score=result.score,
                rank=result.rank,
            )
            for result in search_results
        ]

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error performing search for query '%s'", body.query)
        raise HTTPException(status_code=500, detail="Internal server error during search")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

