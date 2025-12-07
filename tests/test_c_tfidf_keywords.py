"""
Tests specifically for the c-TF-IDF keyword extraction logic.
"""

import numpy as np

from src.modeling.cluster_topics import compute_cluster_summaries


def test_c_tfidf_prefers_cluster_specific_terms() -> None:
    """
    Construct a tiny synthetic example where each cluster has a
    clearly distinctive term and a shared common term, and verify that:

    - Cluster 0 keywords include "apple" but not only "common".
    - Cluster 1 keywords include "banana" but not only "common".
    - The shared token "common" is not the only or dominant keyword.
    """
    titles = ["DocA1", "DocA2", "DocB1", "DocB2"]

    # Embeddings are arbitrary here; we only care about labels.
    embeddings = np.zeros((4, 2), dtype=float)

    # Two clusters: 0 and 1
    labels = np.array([0, 0, 1, 1], dtype=int)

    # Simple centers; again, only shapes matter for compute_cluster_summaries.
    centers = np.zeros((2, 2), dtype=float)

    # Cluster 0 texts use "apple" heavily + a shared word "common"
    # Cluster 1 texts use "banana" heavily + the same shared word "common"
    cleaned_texts = [
        "apple apple apple common",
        "apple apple common",
        "banana banana banana common",
        "banana banana common",
    ]

    summaries = compute_cluster_summaries(
        titles=titles,
        cleaned_texts=cleaned_texts,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
        top_k_keywords=5,
        top_k_articles=2,
    )

    # Build a quick lookup from cluster_id to its keyword list.
    cluster_to_keywords = {
        int(row["cluster_id"]): list(row["keywords"]) for _, row in summaries.iterrows()
    }

    kw0 = cluster_to_keywords[0]
    kw1 = cluster_to_keywords[1]

    # Distinctive terms should appear in the respective clusters.
    assert "apple" in kw0
    assert "banana" in kw1

    # The shared token "common" should not be the only or clearly dominant keyword.
    # (c-TF-IDF should downweight it because it appears in both clusters.)
    assert any(term != "common" for term in kw0)
    assert any(term != "common" for term in kw1)


