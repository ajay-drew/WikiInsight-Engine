"""
FastAPI main application for the WikiInsight Engine.

Exposes endpoints for:
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

setup_logging()
logger = logging.getLogger(__name__)

_topic_index: Optional[TopicIndex] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global _topic_index
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
    yield
    # Shutdown (if needed)
    _topic_index = None


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

