"""
Integration-style tests for the FastAPI app when a TopicIndex is available.

These tests inject a small in-memory TopicIndex so that /topics/lookup and
/explain/{article_title} return successful responses.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.neighbors import NearestNeighbors

from src.api import main as api_main
from src.modeling.topic_index import TopicIndex, TopicLookupResult


def _make_small_topic_index() -> TopicIndex:
    titles = ["A", "B"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=float)

    emb_df = pd.DataFrame(
        {"title": titles, "embedding": [embeddings[0].tolist(), embeddings[1].tolist()]}
    )
    assignments_df = pd.DataFrame({"title": titles, "cluster_id": [0, 0]})
    summaries_df = pd.DataFrame(
        {
            "cluster_id": [0],
            "size": [2],
            "keywords": [["kw0", "topic"]],
            "top_articles": [["A", "B"]],
        }
    )

    nn_index = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_index.fit(embeddings)
    cluster_model = SimpleNamespace()

    return TopicIndex(
        embeddings_df=emb_df,
        embeddings_matrix=embeddings,
        assignments_df=assignments_df,
        summaries_df=summaries_df,
        cluster_model=cluster_model,
        nn_index=nn_index,
    )


def test_topics_lookup_with_injected_index():
    """POST /topics/lookup should return 200 and cluster info when index is present."""
    # Inject a small TopicIndex into the module-level variable used by the app
    api_main._topic_index = _make_small_topic_index()
    client = TestClient(api_main.app)

    resp = client.post("/topics/lookup", json={"article_title": "A"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["article_title"] == "A"
    assert data["cluster_id"] == 0
    assert "B" in data["similar_articles"]


def test_explain_with_injected_index():
    """GET /explain/{article_title} should return 200 and explanation when index is present."""
    api_main._topic_index = _make_small_topic_index()
    client = TestClient(api_main.app)

    resp = client.get("/explain/A")
    assert resp.status_code == 200
    data = resp.json()
    assert data["article_title"] == "A"
    explanation = data.get("explanation", {})
    assert explanation.get("cluster_id") == 0
    assert explanation.get("cluster_size") == 2
    assert "kw0" in explanation.get("keywords", [])


