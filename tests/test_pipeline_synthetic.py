"""
Small end-to-end style test of the clustering + TopicIndex pipeline using synthetic data.

This test does NOT hit Wikipedia or the filesystem; it constructs a minimal set of
embeddings, runs the clustering utilities, and wires a TopicIndex on top to verify
that the basic flow works together.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.modeling.cluster_topics import compute_cluster_summaries, make_clusterer
from src.modeling.topic_index import TopicIndex


def test_synthetic_pipeline_end_to_end():
    """Run a tiny synthetic pipeline: embeddings -> clustering -> summaries -> TopicIndex."""
    titles = ["Doc1", "Doc2", "Doc3", "Doc4"]
    # Two clusters in 2D space
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],  # near Doc1
            [0.0, 1.0],
            [0.1, 0.9],  # near Doc3
        ],
        dtype=float,
    )

    cfg = {"method": "kmeans", "n_clusters": 2, "random_state": 0}
    model, labels = make_clusterer(embeddings, cfg)

    texts = [f"sample text for {t}" for t in titles]
    centers = model.cluster_centers_

    summaries_df = compute_cluster_summaries(
        titles=titles,
        cleaned_texts=texts,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
    )

    # Build assignments + embeddings df expected by TopicIndex
    emb_df = pd.DataFrame(
        {
            "title": titles,
            "embedding": [vec.tolist() for vec in embeddings],
        }
    )
    assignments_df = pd.DataFrame(
        {
            "title": titles,
            "cluster_id": labels.astype(int),
        }
    )

    nn_index = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_index.fit(embeddings)
    cluster_model = SimpleNamespace()

    index = TopicIndex(
        embeddings_df=emb_df,
        embeddings_matrix=embeddings,
        assignments_df=assignments_df,
        summaries_df=summaries_df,
        cluster_model=cluster_model,
        nn_index=nn_index,
    )

    # Lookup should succeed and return non-empty similar articles and summary
    result = index.lookup("Doc1", n_neighbors=2)
    assert result.cluster_id in (0, 1)
    assert len(result.similar_articles) >= 1

    summary = index.get_cluster_summary(result.cluster_id)  # type: ignore[arg-type]
    assert summary["cluster_id"] == result.cluster_id
    assert summary["size"] >= 1


