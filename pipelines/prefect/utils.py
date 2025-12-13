"""
Utility functions for Prefect flows.

Helper functions to check artifact existence, validate data quality,
and extract metrics from stage outputs.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Artifact paths
RAW_DATA_PATH = "data/raw/articles.json"
CLEANED_ARTICLES_PATH = "data/processed/cleaned_articles.parquet"
EMBEDDINGS_PATH = "data/features/embeddings.parquet"
CLUSTER_ASSIGNMENTS_PATH = "models/clustering/cluster_assignments.parquet"
CLUSTERS_SUMMARY_PATH = "models/clustering/clusters_summary.parquet"
METRICS_PATH = "models/clustering/metrics.json"


def artifact_exists(path: str) -> bool:
    """Check if an artifact file exists and is non-empty."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    return True


def check_ingestion_artifacts() -> bool:
    """Check if ingestion artifacts exist."""
    return artifact_exists(RAW_DATA_PATH)


def check_preprocessing_artifacts() -> bool:
    """Check if preprocessing artifacts exist."""
    return artifact_exists(CLEANED_ARTICLES_PATH) and artifact_exists(EMBEDDINGS_PATH)


def check_clustering_artifacts() -> bool:
    """Check if clustering artifacts exist."""
    return (
        artifact_exists(CLUSTER_ASSIGNMENTS_PATH)
        and artifact_exists(CLUSTERS_SUMMARY_PATH)
    )


def validate_raw_articles() -> Dict[str, any]:
    """
    Validate raw articles data quality.
    
    Returns:
        Dict with validation results and metrics
    """
    if not check_ingestion_artifacts():
        return {"valid": False, "error": "Raw articles file not found"}

    try:
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            return {"valid": False, "error": "Articles must be a list"}

        if len(articles) == 0:
            return {"valid": False, "error": "No articles found"}

        # Check article structure
        required_keys = {"title", "text"}
        invalid_count = 0
        for article in articles:
            if not isinstance(article, dict):
                invalid_count += 1
                continue
            if not required_keys.issubset(article.keys()):
                invalid_count += 1

        return {
            "valid": invalid_count == 0,
            "article_count": len(articles),
            "invalid_count": invalid_count,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def validate_embeddings() -> Dict[str, any]:
    """
    Validate embeddings data quality.
    
    Returns:
        Dict with validation results and metrics
    """
    if not check_preprocessing_artifacts():
        return {"valid": False, "error": "Embeddings file not found"}

    try:
        emb_df = pd.read_parquet(EMBEDDINGS_PATH)
        cleaned_df = pd.read_parquet(CLEANED_ARTICLES_PATH)

        if "embedding" not in emb_df.columns:
            return {"valid": False, "error": "Embeddings column missing"}

        if len(emb_df) == 0:
            return {"valid": False, "error": "No embeddings found"}

        if len(emb_df) != len(cleaned_df):
            return {
                "valid": False,
                "error": f"Embeddings count ({len(emb_df)}) != cleaned articles count ({len(cleaned_df)})",
            }

        # Check embedding dimensions
        sample_emb = emb_df["embedding"].iloc[0]
        if not isinstance(sample_emb, list) or len(sample_emb) == 0:
            return {"valid": False, "error": "Invalid embedding format"}

        return {
            "valid": True,
            "article_count": len(emb_df),
            "embedding_dim": len(sample_emb),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def extract_ingestion_metrics() -> Dict[str, any]:
    """Extract metrics from ingestion stage output."""
    if not check_ingestion_artifacts():
        return {}

    try:
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            return {}

        # Calculate metrics
        total_articles = len(articles)
        total_words = sum(len(article.get("text", "").split()) for article in articles)
        avg_words = total_words / total_articles if total_articles > 0 else 0
        total_links = sum(len(article.get("links", [])) for article in articles)
        avg_links = total_links / total_articles if total_articles > 0 else 0

        return {
            "articles_fetched": total_articles,
            "total_words": total_words,
            "avg_article_length": avg_words,
            "total_links": total_links,
            "avg_links_per_article": avg_links,
        }
    except Exception as exc:
        logger.warning("Failed to extract ingestion metrics: %s", exc)
        return {}


def extract_clustering_metrics() -> Dict[str, any]:
    """Extract metrics from clustering stage output."""
    if not artifact_exists(METRICS_PATH):
        return {}

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as exc:
        logger.warning("Failed to extract clustering metrics: %s", exc)
        return {}

