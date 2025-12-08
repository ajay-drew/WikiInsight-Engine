"""
Tests for the HybridSearchEngine in src.serving.search_engine.
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock

# Ensure project root is on sys.path so `import src...` works reliably in all environments.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.search_engine import HybridSearchEngine, SearchResult


@pytest.fixture
def mock_data() -> List[Dict[str, str]]:
    """
    Small synthetic corpus of 5 articles spanning distinct topics.
    """
    return [
        {"title": "AI Basics", "text": "Artificial intelligence and machine learning."},
        {"title": "Deep Learning", "text": "Neural networks and deep learning techniques."},
        {"title": "Cooking Pasta", "text": "Cooking pasta with tomato sauce and basil."},
        {"title": "Space Exploration", "text": "Rockets, planets, and space missions."},
        {"title": "Baking Bread", "text": "Baking sourdough bread and other recipes."},
    ]


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """
    Random embeddings for 5 articles with dimension 384.
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(5, 384)).astype(float)


@pytest.fixture
def mock_model() -> MagicMock:
    """
    Mock sentence-transformer-like model; .encode() returns a random embedding.
    """
    rng = np.random.default_rng(1)
    model = MagicMock()
    model.encode.side_effect = lambda text: rng.normal(size=(384,))
    return model


@pytest.fixture
def search_engine(mock_data, mock_embeddings, mock_model) -> HybridSearchEngine:
    """
    Construct a HybridSearchEngine instance on the synthetic corpus.
    """
    return HybridSearchEngine(
        articles=mock_data,
        embeddings=mock_embeddings,
        model=mock_model,
    )


def test_initialization_builds_indexes(mock_data, mock_embeddings, mock_model) -> None:
    """
    Ensure HybridSearchEngine initializes NearestNeighbors and BM25 properly.
    """
    engine = HybridSearchEngine(
        articles=mock_data,
        embeddings=mock_embeddings,
        model=mock_model,
    )

    # Basic shape and internal fields
    assert len(engine.articles) == 5
    assert engine.embeddings.shape == (5, 384)
    assert len(engine._titles) == 5


def test_semantic_search_uses_model_and_returns_results(search_engine, mock_model) -> None:
    """
    Verify that semantic search calls the model's encode() and returns ranked titles.
    """
    query = "neural networks"
    results = search_engine._search_semantic(query, top_k=3)

    # Under the hood, model.encode() must have been called
    mock_model.encode.assert_called()

    # Results should be a list of (title, rank) tuples
    assert isinstance(results, list)
    assert all(isinstance(t, tuple) and isinstance(t[0], str) and isinstance(t[1], int) for t in results)


def test_keyword_search_finds_exact_match(search_engine) -> None:
    """
    Verify BM25 keyword search can find a document with exact term match.
    """
    query = "Cooking"
    results = search_engine._search_keyword(query, top_k=5)

    titles = [title for title, _ in results]
    assert "Cooking Pasta" in titles


def test_bm25_prefers_title_match_over_body_only(mock_model) -> None:
    """
    Multi-field BM25 should prefer an article where the query appears in the title
    over one where it only appears in the body text.
    """
    articles = [
        {
            "title": "Python (programming language)",
            "text": "Python is a popular programming language used for many applications.",
        },
        {
            "title": "Monty Python",
            "text": "This article discusses the comedy group Monty Python and its history.",
        },
    ]

    # Two simple embeddings – values do not matter for pure keyword search.
    embeddings = np.eye(2, 16)

    engine = HybridSearchEngine(
        articles=articles,
        embeddings=embeddings,
        model=mock_model,
    )

    results = engine._search_keyword("Python", top_k=2)
    assert results, "Expected at least one BM25 result"
    top_title, _ = results[0]
    assert (
        top_title == "Python (programming language)"
    ), "Title match should be ranked ahead of body-only mentions"


def test_rrf_merge_prefers_consistently_high_ranked_items() -> None:
    """
    Manually test RRF merging logic.

    If an item appears at rank 0 in both lists, it should have the highest fused score.
    """
    from src.serving.search_engine import HybridSearchEngine as HSE

    semantic = [("A", 0), ("B", 1), ("C", 2)]
    keyword = [("A", 0), ("C", 1), ("D", 2)]

    fused = HSE._rrf_merge(semantic, keyword, k=60)
    assert fused[0].title == "A"

    # Ensure all unique items from both lists are present
    titles = {r.title for r in fused}
    assert titles == {"A", "B", "C", "D"}


@pytest.mark.parametrize("empty_query", ["", "   "])
def test_empty_query_returns_empty_list(search_engine, empty_query: str) -> None:
    """
    Ensure the public search() method handles empty query strings gracefully.
    """
    results = search_engine.search(empty_query, top_k=5)
    assert results == []


def test_search_returns_searchresult_objects(search_engine) -> None:
    """
    The public search() method should return a list of SearchResult objects.
    """
    results = search_engine.search("space rockets", top_k=3)
    assert isinstance(results, list)
    if results:  # It is allowed to return an empty list if nothing matches
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].title, str)
        assert isinstance(results[0].score, float)
        assert isinstance(results[0].rank, int)



