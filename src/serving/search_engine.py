"""
Hybrid search engine that combines semantic (vector) search and keyword (BM25) search
using Reciprocal Rank Fusion (RRF).

This is intended for use inside the WikiInsight Engine, on top of pre-computed
embeddings and article metadata.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.preprocessing.nltk_utils import normalize_text as nltk_normalize_text

logger = logging.getLogger(__name__)


# Module-level function for parallel tokenization (required for multiprocessing on Windows)
def _tokenize_worker(text: str, use_nltk: bool) -> List[str]:
    """Worker function for parallel tokenization."""
    text = text or ""
    if not text:
        return []

    if use_nltk:
        normalized = nltk_normalize_text(text)
        if not normalized:
            return []
        return normalized.split()

    # Fallback path: simple whitespace-based tokenization.
    return [t for t in text.lower().split() if t]


@dataclass
class SearchResult:
    """Container for a single search result."""

    title: str
    score: float
    rank: int


class HybridSearchEngine:
    """
    Hybrid search engine that combines:

    - Semantic search via k-NN on embeddings (cosine similarity)
    - Keyword search via BM25Okapi
    - Reciprocal Rank Fusion (RRF) to merge both result lists
    """

    def __init__(
        self,
        articles: List[Dict[str, str]],
        embeddings: np.ndarray,
        model,
        *,
        use_nltk_normalization: bool = True,
        title_weight: float = 2.0,
        body_weight: float = 1.0,
        max_workers: int = None,
    ) -> None:
        """
        Initialize the hybrid search engine.

        Args:
            articles: List of article dicts, each containing at least 'title' and 'text'.
            embeddings: Pre-computed embedding matrix of shape (n_articles, dim).
            model: Sentence-transformer-like model with an `.encode(text: str) -> np.ndarray` method.
            use_nltk_normalization: Whether to use NLTK-based text normalization.
            title_weight: Weight for title matches in BM25 scoring.
            body_weight: Weight for body matches in BM25 scoring.
            max_workers: Maximum number of parallel workers for tokenization (None = auto-detect).
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings array must be non-empty.")
        if len(articles) != embeddings.shape[0]:
            raise ValueError("Number of articles must match number of embedding rows.")

        self.articles = articles
        self.embeddings = embeddings
        self.model = model

        # Configuration flags
        self._use_nltk_normalization = use_nltk_normalization
        # We keep weights non-negative; if someone passes a negative weight,
        # clamp it to zero to avoid surprising behavior.
        self._title_weight = max(0.0, float(title_weight))
        self._body_weight = max(0.0, float(body_weight))

        # Determine number of workers for parallel processing
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
        self._max_workers = max_workers

        # Titles and texts extracted up-front for convenience
        logger.info("Extracting titles and texts from %d articles...", len(articles))
        self._titles: List[str] = [a.get("title", "") or "" for a in articles]
        texts: List[str] = [a.get("text", "") or "" for a in articles]

        # ---------------- Semantic index (k-NN over embeddings) ----------------
        # Use cosine distance; we will convert to similarity for ranking.
        logger.info("Building semantic search index (k-NN)...")
        self._nn = NearestNeighbors(metric="cosine")
        self._nn.fit(self.embeddings)
        logger.info("Semantic index built successfully")

        # ---------------- Keyword index (BM25 over tokenized title/body) -------
        # We build two BM25 indexes:
        #   - one over normalized titles (to strongly reward title matches)
        #   - one over normalized article bodies
        # Final scores are a weighted sum of the two.
        logger.info("Tokenizing articles for BM25 index (using %d workers)...", self._max_workers)
        title_tokens_corpus = self._tokenize_parallel(self._titles, desc="Tokenizing titles")
        body_tokens_corpus = self._tokenize_parallel(texts, desc="Tokenizing article bodies")

        logger.info("Building BM25 indexes...")
        self._bm25_title = BM25Okapi(title_tokens_corpus)
        self._bm25_body = BM25Okapi(body_tokens_corpus)
        logger.info("BM25 indexes built successfully")

        logger.info(
            "Initialized HybridSearchEngine with %d articles (dim=%d)",
            len(self.articles),
            self.embeddings.shape[1],
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _tokenize_basic(text: str) -> List[str]:
        """
        Basic whitespace tokenizer with lowercasing.

        This is used as a fallback when NLTK-based normalization is disabled
        or unavailable.
        """
        return [t for t in text.lower().split() if t]

    def _prepare_tokens(self, text: str) -> List[str]:
        """
        Normalize and tokenize text for BM25.

        When NLTK is enabled, this uses `normalize_text` from
        `src.preprocessing.nltk_utils` which performs:
          - lowercasing
          - regex cleanup
          - tokenization
          - stopword removal
          - lemmatization (and optional stemming, if enabled there)

        If NLTK or its data packages are missing, normalize_text()
        gracefully falls back to a simple regex-based tokenizer, so we
        never fail hard here.
        """
        text = text or ""
        if not text:
            return []

        if self._use_nltk_normalization:
            normalized = nltk_normalize_text(text)
            if not normalized:
                return []
            return normalized.split()

        # Fallback path: simple whitespace-based tokenization.
        return self._tokenize_basic(text)

    def _tokenize_parallel(self, texts: List[str], desc: str = "Tokenizing") -> List[List[str]]:
        """
        Tokenize a list of texts in parallel using multiprocessing.

        Args:
            texts: List of text strings to tokenize.
            desc: Description for progress bar.

        Returns:
            List of token lists, one per input text.
        """
        if len(texts) == 0:
            return []

        # For small datasets, use sequential processing (overhead not worth it)
        if len(texts) < 100 or self._max_workers == 1:
            return [self._prepare_tokens(text) for text in tqdm(texts, desc=desc, unit="text")]

        # Use parallel processing for larger datasets
        use_nltk = self._use_nltk_normalization

        # Process in parallel with progress bar
        results: List[List[str]] = []
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all tasks with use_nltk parameter
            future_to_idx = {
                executor.submit(_tokenize_worker, text, use_nltk): idx 
                for idx, text in enumerate(texts)
            }

            # Collect results in order
            results = [None] * len(texts)  # type: ignore[list-item]
            with tqdm(total=len(texts), desc=desc, unit="text") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Error tokenizing text at index %d: %s", idx, exc)
                        results[idx] = []  # Fallback to empty tokens
                    pbar.update(1)

        return results

    # --------------------------------------------------------------------- #
    # Semantic search
    # --------------------------------------------------------------------- #

    def _search_semantic(self, query: str, top_k: int) -> List[Tuple[str, int]]:
        """
        Perform semantic (vector) search using the embedding model + k-NN index.

        Args:
            query: Input query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (title, rank) pairs, where rank is 0-based (0 is best).
        """
        if not query.strip():
            return []

        # Encode query using the provided model
        try:
            q_emb = self.model.encode(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error encoding query for semantic search: %s", exc)
            return []

        q_emb = np.asarray(q_emb).reshape(1, -1)

        n_results = min(top_k, len(self.articles))
        distances, indices = self._nn.kneighbors(q_emb, n_neighbors=n_results)

        # Convert distances (cosine) to similarity for ranking clarity,
        # but we only need ranks here, so we ignore similarity values.
        idx_list = indices[0].tolist()
        results: List[Tuple[str, int]] = []
        for rank, idx in enumerate(idx_list):
            title = self._titles[int(idx)]
            results.append((title, rank))

        return results

    # --------------------------------------------------------------------- #
    # Keyword (BM25) search
    # --------------------------------------------------------------------- #

    def _search_keyword(self, query: str, top_k: int) -> List[Tuple[str, int]]:
        """
        Perform keyword search using BM25Okapi.

        Args:
            query: Input query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (title, rank) pairs, where rank is 0-based (0 is best).
        """
        if not query.strip():
            return []

        tokens = self._prepare_tokens(query)
        if not tokens:
            return []
        # Compute scores separately for titles and bodies, then combine
        # them with a configurable weighting scheme.
        scores_title = self._bm25_title.get_scores(tokens)
        scores_body = self._bm25_body.get_scores(tokens)

        scores_combined = (self._title_weight * scores_title) + (self._body_weight * scores_body)

        # argsort in descending order of scores
        ranked_indices = np.argsort(scores_combined)[::-1]

        results: List[Tuple[str, int]] = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            title = self._titles[int(idx)]
            results.append((title, rank))
        return results

    # --------------------------------------------------------------------- #
    # Reciprocal Rank Fusion
    # --------------------------------------------------------------------- #

    @staticmethod
    def _rrf_merge(
        semantic_results: Iterable[Tuple[str, int]],
        keyword_results: Iterable[Tuple[str, int]],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

        score = sum(1 / (k + rank_i))  over all rankings in which the item appears

        Args:
            semantic_results: Iterable of (title, rank) from semantic search.
            keyword_results: Iterable of (title, rank) from keyword search.
            k: RRF constant (default 60 as in common literature).

        Returns:
            List of SearchResult sorted by score descending.
        """
        scores: Dict[str, float] = {}
        ranks: Dict[str, int] = {}

        def _accumulate(results: Iterable[Tuple[str, int]]) -> None:
            for title, rank in results:
                # RRF formula
                increment = 1.0 / (k + rank)
                scores[title] = scores.get(title, 0.0) + increment
                # Track best (lowest) rank seen for reporting
                ranks[title] = min(ranks.get(title, rank), rank)

        _accumulate(semantic_results)
        _accumulate(keyword_results)

        fused: List[SearchResult] = [
            SearchResult(title=title, score=score, rank=ranks.get(title, 0))
            for title, score in scores.items()
        ]

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Run a hybrid search query.

        Args:
            query: Input query string.
            top_k: Maximum number of results to return after fusion.

        Returns:
            List of SearchResult objects sorted by combined RRF score.
        """
        query = query.strip()
        if not query:
            logger.warning("HybridSearchEngine.search called with empty query.")
            return []

        if top_k <= 0:
            logger.warning("HybridSearchEngine.search called with non-positive top_k=%d.", top_k)
            return []

        semantic_results = self._search_semantic(query, top_k=top_k)
        keyword_results = self._search_keyword(query, top_k=top_k)

        if not semantic_results and not keyword_results:
            logger.info("No results found for query '%s'.", query)
            return []

        fused = self._rrf_merge(semantic_results, keyword_results)
        return fused[:top_k]

