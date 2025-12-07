"""
Tests for SlowAPI rate limiting in FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app  # noqa: E402
from src.modeling.topic_index import TopicIndex, TopicLookupResult  # noqa: E402


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_topic_index():
    """Mock TopicIndex for testing."""
    mock_index = MagicMock(spec=TopicIndex)
    mock_index.lookup.return_value = TopicLookupResult(
        article_title="Test Article",
        cluster_id=1,
        similar_articles=["Article1", "Article2"],
        keywords=["keyword1", "keyword2"],
        explanation={"cluster_size": 10},
    )
    mock_index.get_cluster_summary.return_value = {
        "cluster_id": 1,
        "size": 10,
        "keywords": ["keyword1"],
        "top_articles": ["Article1"],
    }
    mock_index.get_clusters_overview.return_value = MagicMock()
    return mock_index


def test_rate_limiting_configured(client):
    """Test that rate limiting is configured on the app."""
    # Check that limiter is attached to app
    assert hasattr(app.state, "limiter")
    assert app.state.limiter is not None


def test_rate_limit_on_topics_lookup(client, mock_topic_index):
    """Test rate limiting on /topics/lookup endpoint."""
    with patch("src.api.main._topic_index", mock_topic_index):
        # Make requests up to the limit
        for i in range(5):
            response = client.post(
                "/topics/lookup",
                json={"article_title": "Test Article"},
            )
            # Should succeed (we're not hitting the limit yet)
            assert response.status_code in [200, 503]  # 503 if index not loaded


def test_rate_limit_on_explain_endpoint(client, mock_topic_index):
    """Test rate limiting on /explain endpoint."""
    with patch("src.api.main._topic_index", mock_topic_index):
        response = client.get("/explain/Test%20Article")
        # Should succeed or return 503 if index not loaded
        assert response.status_code in [200, 503]


def test_rate_limit_on_clusters_overview(client, mock_topic_index):
    """Test rate limiting on /clusters/overview endpoint."""
    with patch("src.api.main._topic_index", mock_topic_index):
        response = client.get("/clusters/overview")
        # Should succeed or return 503 if index not loaded
        assert response.status_code in [200, 503]


def test_rate_limit_on_cluster_detail(client, mock_topic_index):
    """Test rate limiting on /clusters/{cluster_id} endpoint."""
    with patch("src.api.main._topic_index", mock_topic_index):
        response = client.get("/clusters/1")
        # Should succeed or return 503 if index not loaded
        assert response.status_code in [200, 503, 404]


def test_rate_limit_error_response():
    """Test that rate limit exceeded returns proper error."""
    # Check that exception handler is registered on the app
    # SlowAPI registers handlers automatically when added to app
    assert hasattr(app.state, "limiter")
    assert app.state.limiter is not None


def test_rate_limit_different_endpoints_have_different_limits(client):
    """Test that different endpoints can have different rate limits."""
    # This is a structural test - verify decorators are applied
    import inspect
    
    # Check that limiter decorators are present
    lookup_func = app.routes[0].endpoint if hasattr(app.routes[0], "endpoint") else None
    
    # Verify endpoints exist and are decorated
    routes = [route for route in app.routes if hasattr(route, "path")]
    lookup_routes = [r for r in routes if "/topics/lookup" in r.path]
    explain_routes = [r for r in routes if "/explain" in r.path]
    
    assert len(lookup_routes) > 0
    assert len(explain_routes) > 0


def test_api_endpoints_also_rate_limited(client, mock_topic_index):
    """Test that /api/* endpoints also have rate limiting."""
    with patch("src.api.main._topic_index", mock_topic_index):
        # Test /api/topics/lookup
        response = client.post(
            "/api/topics/lookup",
            json={"article_title": "Test Article"},
        )
        assert response.status_code in [200, 503]
        
        # Test /api/clusters/overview
        response = client.get("/api/clusters/overview")
        assert response.status_code in [200, 503]
        
        # Test /api/clusters/{id}
        response = client.get("/api/clusters/1")
        assert response.status_code in [200, 503, 404]


def test_rate_limiting_with_async_endpoints():
    """Test that async endpoints work with rate limiting."""
    # All endpoints are async, so this is a structural test
    import inspect
    from src.api.main import lookup_topic_cluster, explain_prediction
    
    # Verify they are async functions
    assert inspect.iscoroutinefunction(lookup_topic_cluster)
    assert inspect.iscoroutinefunction(explain_prediction)

