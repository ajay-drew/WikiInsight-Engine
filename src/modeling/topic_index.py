"""
Utilities for loading and querying the topic clustering artifacts.

This module provides a `TopicIndex` class that:
- Loads embeddings, clustering assignments, cluster summaries, and k-NN index
  produced by `src.modeling.cluster_topics`.
- Exposes simple lookup methods for:
  - Getting the cluster and similar articles for a given title.
  - Retrieving cluster-level summaries (keywords, representative articles, size).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .cluster_topics import (
    CLUSTER_ASSIGNMENTS_PATH,
    CLUSTERS_SUMMARY_PATH,
    EMBEDDINGS_PATH,
    KMEANS_MODEL_PATH,
    NN_INDEX_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class TopicLookupResult:
    article_title: str
    cluster_id: Optional[int]
    similar_articles: List[str]
    keywords: Optional[List[str]]
    explanation: Dict


class TopicIndex:
    """In-memory index over clustered Wikipedia articles."""

    def __init__(
        self,
        embeddings_df: pd.DataFrame,
        embeddings_matrix: np.ndarray,
        assignments_df: pd.DataFrame,
        summaries_df: pd.DataFrame,
        cluster_model,
        nn_index,
    ) -> None:
        self.embeddings_df = embeddings_df
        self.embeddings_matrix = embeddings_matrix
        self.assignments_df = assignments_df
        self.summaries_df = summaries_df
        self.cluster_model = cluster_model
        self.nn_index = nn_index

        # Fast title lookups (case-insensitive)
        self._title_to_idx: Dict[str, int] = {
            str(title).lower(): idx for idx, title in enumerate(self.embeddings_df["title"])
        }

        # Cluster summaries by id
        if "cluster_id" in self.summaries_df.columns:
            self._summary_by_id = self.summaries_df.set_index("cluster_id")
        else:
            self._summary_by_id = pd.DataFrame().set_index(pd.Index([], name="cluster_id"))

    @classmethod
    def load_default(cls) -> "TopicIndex":
        """Load all artifacts from their default locations."""
        logger.info("Loading topic index artifacts")

        # Check if required files exist
        import os
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_PATH}\n"
                "Please run the data pipeline first:\n"
                "  dvc repro\n"
                "  OR\n"
                "  python -m src.ingestion.fetch_wikipedia_data\n"
                "  python -m src.preprocessing.process_data\n"
                "  python -m src.modeling.cluster_topics"
            )

        emb_df = pd.read_parquet(EMBEDDINGS_PATH)
        if "embedding" not in emb_df.columns:
            raise ValueError("Embeddings parquet must contain an 'embedding' column")
        embeddings = np.vstack(emb_df["embedding"].to_list())

        assignments_df = pd.read_parquet(CLUSTER_ASSIGNMENTS_PATH)
        summaries_df = pd.read_parquet(CLUSTERS_SUMMARY_PATH)

        cluster_model = joblib.load(KMEANS_MODEL_PATH)
        nn_index = joblib.load(NN_INDEX_PATH)

        logger.info(
            "Loaded %d embeddings, %d assignments, %d cluster summaries",
            len(emb_df),
            len(assignments_df),
            len(summaries_df),
        )

        return cls(
            embeddings_df=emb_df,
            embeddings_matrix=embeddings,
            assignments_df=assignments_df,
            summaries_df=summaries_df,
            cluster_model=cluster_model,
            nn_index=nn_index,
        )

    def _get_index_for_title(self, article_title: str) -> int:
        key = str(article_title).strip().lower()
        if key not in self._title_to_idx:
            raise KeyError(f"Article title '{article_title}' not found in index")
        return self._title_to_idx[key]

    def lookup(
        self,
        article_title: str,
        n_neighbors: Optional[int] = None,
    ) -> TopicLookupResult:
        """
        Look up the cluster and similar articles for a given title.

        Note: This currently only supports articles that are part of the
        precomputed corpus (no on-the-fly Wikipedia fetch).
        """
        idx = self._get_index_for_title(article_title)

        # Cluster id
        cluster_id = int(self.assignments_df.iloc[idx]["cluster_id"])

        # Similar articles via k-NN on embeddings
        emb_vec = self.embeddings_matrix[idx : idx + 1]
        n_neighbors = n_neighbors or self.nn_index.n_neighbors
        _, indices = self.nn_index.kneighbors(emb_vec, n_neighbors=n_neighbors)
        neighbor_indices = indices[0]

        titles = self.embeddings_df["title"].tolist()
        similar_articles: List[str] = []
        for ni in neighbor_indices:
            if int(ni) == int(idx):
                continue
            similar_articles.append(str(titles[int(ni)]))

        # Cluster summary
        keywords: Optional[List[str]] = None
        explanation: Dict = {}
        if cluster_id in self._summary_by_id.index:
            row = self._summary_by_id.loc[cluster_id]
            keywords = list(row.get("keywords", []))
            top_articles = list(row.get("top_articles", []))
            size = int(row.get("size", 0))
            explanation = {
                "cluster_size": size,
                "top_articles": top_articles,
            }

        return TopicLookupResult(
            article_title=str(self.embeddings_df.iloc[idx]["title"]),
            cluster_id=cluster_id,
            similar_articles=similar_articles,
            keywords=keywords,
            explanation=explanation,
        )

    def get_cluster_summary(self, cluster_id: int) -> Dict:
        """Return a simple dict summary for a cluster id."""
        if cluster_id not in self._summary_by_id.index:
            raise KeyError(f"Cluster id {cluster_id} not found")
        row = self._summary_by_id.loc[cluster_id]
        return {
            "cluster_id": int(cluster_id),
            "size": int(row.get("size", 0)),
            "keywords": list(row.get("keywords", [])),
            "top_articles": list(row.get("top_articles", [])),
        }

    def get_clusters_overview(self) -> pd.DataFrame:
        """Return a DataFrame with cluster_id, size, and keywords columns."""
        cols = [c for c in ["cluster_id", "size", "keywords"] if c in self.summaries_df.columns]
        return self.summaries_df[cols].copy()



