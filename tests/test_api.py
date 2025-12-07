"""
Tests for the FastAPI application.
"""

from fastapi.testclient import TestClient

from src.api.main import app


def test_root_endpoint():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("message") == "WikiInsight Engine API"


def test_topics_lookup_without_index_returns_503_or_404():
    """
    When clustering artifacts are missing, the topics endpoint should not crash.
    """
    client = TestClient(app)
    resp = client.post("/topics/lookup", json={"article_title": "Nonexistent"})
    assert resp.status_code in (404, 503)


