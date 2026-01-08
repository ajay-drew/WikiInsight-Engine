"""
Tests for clustering utilities.
"""

import numpy as np
import pytest

from src.modeling.cluster_topics import compute_cluster_summaries, make_clusterer


@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_make_clusterer_and_summaries(method: str) -> None:
    """Smoke test for clustering and summary generation for multiple methods."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 4))
    cfg = {"method": method, "n_clusters": 3, "random_state": 0, "adaptive": False}

    model, labels, _ = make_clusterer(embeddings, cfg)
    assert labels.shape[0] == embeddings.shape[0]

    # Both KMeans and our Agglomerative wrapper expose cluster_centers_
    assert hasattr(model, "cluster_centers_")
    assert model.cluster_centers_.shape[0] == cfg["n_clusters"]

    titles = [f"Article {i}" for i in range(len(embeddings))]
    texts = [f"sample text for article {i}" for i in range(len(embeddings))]
    centers = model.cluster_centers_

    summaries = compute_cluster_summaries(
        titles=titles,
        cleaned_texts=texts,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
    )

    assert not summaries.empty
    assert {"cluster_id", "size", "keywords", "top_articles"}.issubset(summaries.columns)

    # Basic checks on c-TF-IDF based keywords: non-empty, short list, no trivial stopwords.
    for _, row in summaries.iterrows():
        keywords = row["keywords"]
        assert isinstance(keywords, list)
        # We cap at 20 keywords per cluster in compute_cluster_summaries
        assert len(keywords) <= 20
        for kw in keywords:
            assert isinstance(kw, str)
            assert kw == kw.lower()
            # Very weak stopword check – mainly to ensure the tokenizer + filtering runs.
            assert kw not in {"the", "and", "is", "of", "to"}


def test_make_clusterer_adaptive_mode():
    """Test make_clusterer with adaptive mode enabled."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(100, 10))
    cfg = {
        "adaptive": True,
        "auto_n_clusters": False,
        "n_clusters": 5,
        "random_state": 0,
        "preprocessing": {"normalize_embeddings": True},
    }

    model, labels, _ = make_clusterer(embeddings, cfg)
    assert labels.shape[0] == embeddings.shape[0]
    assert hasattr(model, "cluster_centers_")
    assert model.cluster_centers_.shape[0] == 5

