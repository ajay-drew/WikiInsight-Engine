"""
FastAPI main application for the WikiInsight Engine.

Exposes endpoints for:
- Hybrid search (semantic + keyword search with RRF).
- Health and metadata.
- Topic cluster lookup for a given article title.
- Cluster-level explanations/summaries.
"""

import logging
import urllib.parse
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Dict, List, Optional

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
from src.graph.graph_service import GraphService
from src.research.wikidata_linker import WikidataLinker

setup_logging()
logger = logging.getLogger(__name__)

_topic_index: Optional[TopicIndex] = None
_search_engine: Optional[HybridSearchEngine] = None
_graph_service: Optional[GraphService] = None
_wikidata_linker: Optional[WikidataLinker] = None
_cleaned_articles_df = None


def _get_wikipedia_url(title: str) -> str:
    """Generate Wikipedia URL for an article title."""
    # Replace spaces with underscores and URL encode
    encoded = title.replace(" ", "_")
    encoded = urllib.parse.quote(encoded, safe="_")
    return f"https://en.wikipedia.org/wiki/{encoded}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global _topic_index, _search_engine, _graph_service, _wikidata_linker, _cleaned_articles_df
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
            _cleaned_articles_df = cleaned_df  # Store for metadata lookup
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

            # Load embedding model and search configuration
            config = yaml.safe_load(open(CONFIG_PATH, "r"))
            model_name = config.get("preprocessing", {}).get("embeddings", {}).get(
                "model",
                "all-MiniLM-L6-v2",
            )
            bm25_cfg = config.get("search", {}).get("bm25", {}) if isinstance(config, dict) else {}
            title_weight = float(bm25_cfg.get("title_weight", 2.0))
            body_weight = float(bm25_cfg.get("body_weight", 1.0))
            use_nltk_normalization = bool(bm25_cfg.get("use_nltk_normalization", True))
            embedding_model = EmbeddingGenerator(model_name=model_name)

            _search_engine = HybridSearchEngine(
                articles=articles,
                embeddings=embeddings,
                model=embedding_model.model,
                use_nltk_normalization=use_nltk_normalization,
                title_weight=title_weight,
                body_weight=body_weight,
            )
            logger.info("HybridSearchEngine loaded successfully with %d articles", len(articles))
        else:
            logger.warning(
                "Search engine artifacts not found. Search endpoint will be unavailable.\n"
                "To generate the required artifacts, run:\n"
                "  dvc repro"
            )
            _search_engine = None
            _cleaned_articles_df = None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load HybridSearchEngine: %s", exc)
        _search_engine = None
        _cleaned_articles_df = None

    # Load graph service
    try:
        _graph_service = GraphService()
        logger.info("Graph service loaded successfully")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph service not available: %s", exc)
        _graph_service = None

    # Load Wikidata linker
    try:
        import yaml
        config = yaml.safe_load(open("config.yaml", "r"))
        wikidata_cfg = config.get("wikidata", {})
        enabled = wikidata_cfg.get("enabled", True)
        cache_path = wikidata_cfg.get("cache_path", "data/entities/wikidata_mappings.parquet")
        request_delay = wikidata_cfg.get("request_delay", 0.3)
        _wikidata_linker = WikidataLinker(
            cache_path=cache_path,
            enabled=enabled,
            request_delay=request_delay,
        )
        logger.info("Wikidata linker initialized")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikidata linker not available: %s", exc)
        _wikidata_linker = None

    yield
    # Shutdown (if needed)
    if _wikidata_linker:
        _wikidata_linker.finalize()
    _topic_index = None
    _search_engine = None
    _graph_service = None
    _wikidata_linker = None
    _cleaned_articles_df = None


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
    wikipedia_url: str
    wikidata_qid: Optional[str] = None
    wikidata_url: Optional[str] = None
    cluster_id: Optional[int] = None
    categories: List[str] = []
    link_count: int = 0


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    query: str
    results: List[SearchResultResponse]
    total_results: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses with latency, and track metrics."""
    from src.monitoring.api_metrics import record_api_request
    
    start = perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        
        # Record metrics
        record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=duration_ms,
        )
        
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
        
        # Record error
        record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            latency_ms=duration_ms,
        )
        
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


# Graph API endpoints

class GraphNeighborsResponse(BaseModel):
    """Response model for graph neighbors."""

    article_title: str
    neighbors: List[Dict]


class GraphPathResponse(BaseModel):
    """Response model for graph path."""

    from_title: str
    to_title: str
    path: Optional[List[str]]
    found: bool


class GraphVisualizationResponse(BaseModel):
    """Response model for graph visualization."""

    nodes: List[Dict]
    edges: List[Dict]


@app.get("/api/graph/neighbors/{article_title}")
@limiter.limit("100/minute")
async def get_graph_neighbors(request: Request, article_title: str):
    """Get graph neighbors for an article."""
    if _graph_service is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is not available. Run graph construction pipeline first.",
        )

    try:
        neighbors = _graph_service.get_neighbors(article_title)
        return {"article_title": article_title, "neighbors": neighbors}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error getting graph neighbors for '%s'", article_title)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/graph/path/{from_title}/{to_title}")
@limiter.limit("100/minute")
async def get_graph_path(request: Request, from_title: str, to_title: str):
    """Find shortest path between two articles."""
    if _graph_service is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is not available. Run graph construction pipeline first.",
        )

    try:
        path = _graph_service.find_path(from_title, to_title)
        return {
            "from_title": from_title,
            "to_title": to_title,
            "path": path,
            "found": path is not None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error finding path from '%s' to '%s'", from_title, to_title)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/graph/visualization/{cluster_id}")
@limiter.limit("60/minute")
async def get_graph_visualization(request: Request, cluster_id: int):
    """Get graph data for cluster visualization."""
    if _graph_service is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is not available. Run graph construction pipeline first.",
        )

    if _topic_index is None:
        raise HTTPException(
            status_code=503,
            detail="Topic index is not available. Run clustering pipeline first.",
        )

    try:
        # Get cluster assignments
        assignments_df = _topic_index.assignments_df
        cluster_assignments = {
            str(row["title"]).lower(): int(row["cluster_id"])
            for _, row in assignments_df.iterrows()
        }

        # Get subgraph for cluster
        nodes, edges = _graph_service.get_cluster_subgraph(
            cluster_id, cluster_assignments
        )

        return {"nodes": nodes, "edges": edges}
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {cluster_id} not found.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error getting graph visualization for cluster %d", cluster_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/graph/article/{article_title}")
@limiter.limit("100/minute")
async def get_article_graph(request: Request, article_title: str):
    """Get graph centered on a specific article."""
    if _graph_service is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is not available. Run graph construction pipeline first.",
        )

    try:
        nodes, edges = _graph_service.get_article_graph(article_title)
        return {"nodes": nodes, "edges": edges}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error getting article graph for '%s'", article_title)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("100/minute")
async def api_search(request: Request, body: SearchRequest):
    """
    Hybrid search endpoint combining semantic (vector) and keyword (BM25) search.
    
    Args:
        request: FastAPI request object (for rate limiting)
        body: Search request with query and optional top_k
        
    Returns:
        Search results with titles, scores, ranks, and metadata
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
        
        # Build results with metadata
        results = []
        for result in search_results:
            # Get metadata from cleaned articles
            wikipedia_url = _get_wikipedia_url(result.title)
            wikidata_qid = None
            wikidata_url = None
            cluster_id = None
            categories = []
            link_count = 0

            # Look up article metadata
            if _cleaned_articles_df is not None:
                article_row = _cleaned_articles_df[_cleaned_articles_df["title"].str.lower() == result.title.lower()]
                if not article_row.empty:
                    row = article_row.iloc[0]
                    categories = row.get("categories", [])
                    if isinstance(categories, list):
                        categories = [str(c) for c in categories[:5]]  # Limit to 5
                    links = row.get("links", [])
                    if isinstance(links, list):
                        link_count = len(links)

            # Get cluster ID from topic index
            if _topic_index is not None:
                try:
                    lookup_result = _topic_index.lookup(result.title)
                    cluster_id = lookup_result.cluster_id
                except (KeyError, Exception):  # noqa: BLE001
                    pass

            # Get Wikidata QID (with error handling - don't break search if Wikidata fails)
            if _wikidata_linker is not None:
                try:
                    wikidata_qid = _wikidata_linker.link_entity(result.title)
                    if wikidata_qid:
                        wikidata_url = _wikidata_linker.get_wikidata_url(wikidata_qid)
                except Exception as exc:  # noqa: BLE001
                    # Log but don't fail - Wikidata linking is optional
                    logger.debug("Wikidata linking failed for '%s': %s", result.title, exc)
                    wikidata_qid = None
                    wikidata_url = None

            results.append(
                SearchResultResponse(
                    title=result.title,
                    score=result.score,
                    rank=result.rank,
                    wikipedia_url=wikipedia_url,
                    wikidata_qid=wikidata_qid,
                    wikidata_url=wikidata_url,
                    cluster_id=cluster_id,
                    categories=categories,
                    link_count=link_count,
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error performing search for query '%s'", body.query)
        raise HTTPException(status_code=500, detail="Internal server error during search")


# Monitoring endpoints
@app.get("/api/monitoring/pipeline-status")
@limiter.limit("60/minute")
async def get_pipeline_status(request: Request):
    """Get pipeline status (last run times, success/failure)."""
    import os
    from datetime import datetime
    
    status = {
        "ingestion": {
            "has_artifacts": os.path.exists("data/raw/articles.json"),
            "artifact_size": os.path.getsize("data/raw/articles.json") if os.path.exists("data/raw/articles.json") else 0,
        },
        "preprocessing": {
            "has_artifacts": os.path.exists("data/processed/cleaned_articles.parquet") and os.path.exists("data/features/embeddings.parquet"),
        },
        "clustering": {
            "has_artifacts": os.path.exists("models/clustering/cluster_assignments.parquet"),
        },
    }
    
    # Try to get last modified times
    for stage, info in status.items():
        if stage == "ingestion" and info["has_artifacts"]:
            mtime = os.path.getmtime("data/raw/articles.json")
            info["last_modified"] = datetime.fromtimestamp(mtime).isoformat()
        elif stage == "preprocessing" and info["has_artifacts"]:
            mtime = max(
                os.path.getmtime("data/processed/cleaned_articles.parquet"),
                os.path.getmtime("data/features/embeddings.parquet"),
            )
            info["last_modified"] = datetime.fromtimestamp(mtime).isoformat()
        elif stage == "clustering" and info["has_artifacts"]:
            mtime = os.path.getmtime("models/clustering/cluster_assignments.parquet")
            info["last_modified"] = datetime.fromtimestamp(mtime).isoformat()
    
    return status


@app.get("/api/monitoring/metrics")
@limiter.limit("60/minute")
async def get_metrics(request: Request):
    """Get API performance metrics."""
    from src.monitoring.api_metrics import get_api_metrics_summary
    
    window_seconds = request.query_params.get("window_seconds")
    window = float(window_seconds) if window_seconds else None
    
    summary = get_api_metrics_summary(window_seconds=window)
    return summary


@app.get("/api/monitoring/drift")
@limiter.limit("60/minute")
async def get_drift_scores(request: Request):
    """Get latest drift detection scores."""
    import json
    import os
    
    report_path = "reports/drift_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Drift report not found. Run clustering first.")


@app.get("/api/monitoring/stability")
@limiter.limit("60/minute")
async def get_cluster_stability(request: Request):
    """Get cluster stability metrics."""
    from src.monitoring.cluster_stability import calculate_cluster_stability
    
    stability = calculate_cluster_stability()
    return stability


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

