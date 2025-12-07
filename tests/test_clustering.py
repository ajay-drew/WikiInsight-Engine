"""
Tests for clustering utilities.
"""

import numpy as np

from src.modeling.cluster_topics import compute_cluster_summaries, make_clusterer


def test_make_clusterer_and_summaries():
    """Smoke test for clustering and summary generation."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 4))
    cfg = {"method": "kmeans", "n_clusters": 3, "random_state": 0}

    model, labels = make_clusterer(embeddings, cfg)
    assert labels.shape[0] == embeddings.shape[0]

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


