"""
Unit tests for TopicIndex helper.

These tests construct a small in-memory TopicIndex instance without touching disk.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.modeling.topic_index import TopicIndex


def _make_small_topic_index() -> TopicIndex:
    titles = ["A", "B", "C"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    emb_df = pd.DataFrame(
        {
            "title": titles,
            "embedding": [vec.tolist() for vec in embeddings],
        }
    )

    assignments_df = pd.DataFrame(
        {
            "title": titles,
            "cluster_id": [0, 0, 1],
        }
    )

    summaries_df = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "size": [2, 1],
            "keywords": [["kw0", "topic"], ["kw1"]],
            "top_articles": [["A", "B"], ["C"]],
        }
    )

    nn_index = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_index.fit(embeddings)

    # cluster_model is not used in TopicIndex logic for now; provide a simple stub
    cluster_model = SimpleNamespace()

    return TopicIndex(
        embeddings_df=emb_df,
        embeddings_matrix=embeddings,
        assignments_df=assignments_df,
        summaries_df=summaries_df,
        cluster_model=cluster_model,
        nn_index=nn_index,
    )


def test_topic_index_lookup_and_similar_articles():
    """TopicIndex.lookup should return cluster id and non-empty similars for known titles."""
    index = _make_small_topic_index()

    result = index.lookup("A", n_neighbors=2)
    assert result.article_title == "A"
    assert result.cluster_id == 0
    # Should at least see B as similar to A
    assert "B" in result.similar_articles


def test_topic_index_cluster_summary_and_overview():
    """TopicIndex should return consistent summaries and overview."""
    index = _make_small_topic_index()

    summary = index.get_cluster_summary(0)
    assert summary["cluster_id"] == 0
    assert summary["size"] == 2
    assert "kw0" in summary["keywords"]

    overview = index.get_clusters_overview()
    assert not overview.empty
    assert set(overview["cluster_id"]) == {0, 1}


