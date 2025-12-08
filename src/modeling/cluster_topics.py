"""
Clustering pipeline for WikiInsight Engine topic explorer.

This module:
- Loads embeddings (and optionally cleaned articles) produced by
  `src.preprocessing.process_data`.
- Fits a clustering model (e.g., KMeans) according to `config.yaml`.
- Builds a k-NN index for similar-article lookup.
- Generates cluster summaries:
    * representative articles (closest to cluster center)
    * **topic words** using c-TF-IDF (class-based TF-IDF) with tokenization + stopword filtering
      so that the keywords are both frequent in the cluster (normalized TF) and
      distinctive compared to other clusters (high IDF based on class frequency).
- Saves artifacts under `models/clustering/` and a metrics JSON for DVC/CI.
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Set

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.common.logging_utils import setup_logging

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
EMBEDDINGS_PATH = os.path.join("data", "features", "embeddings.parquet")
CLEANED_ARTICLES_PATH = os.path.join("data", "processed", "cleaned_articles.parquet")

MODEL_DIR = os.path.join("models", "clustering")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
NN_INDEX_PATH = os.path.join(MODEL_DIR, "nn_index.pkl")
CLUSTER_ASSIGNMENTS_PATH = os.path.join(MODEL_DIR, "cluster_assignments.parquet")
CLUSTERS_SUMMARY_PATH = os.path.join(MODEL_DIR, "clusters_summary.parquet")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Tokenization / stopwords for keyword extraction
# ---------------------------------------------------------------------------

# A lightweight, hard-coded English stop word list. We keep this even when
# spaCy is available so behaviour is stable and easy to reason about.
STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "of",
    "to",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "you",
    "we",
    "i",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

try:  # spaCy is optional; we fall back to regex tokenization if unavailable
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:  # noqa: BLE001
        logger.warning(
            "spaCy model 'en_core_web_sm' not available; "
            "falling back to simple regex tokenization for keywords.",
        )
        _NLP = None
except Exception:  # noqa: BLE001
    _NLP = None


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text into normalized word tokens with stopword and noise filtering.

    For consistency with the wider preprocessing stack, this function mirrors
    the behaviour of the NLTK-based normalizer used before embeddings. We keep
    the spaCy path for environments where it is already configured, but fall
    back to a simple regex-based tokenizer when neither spaCy nor NLTK helpers
    are available.
    """
    text = text or ""

    # Prefer spaCy if available: better tokenization and stopword detection.
    if _NLP is not None:
        doc = _NLP(text)
        tokens: List[str] = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < 3:
                continue
            if lemma in STOP_WORDS or tok.is_stop:
                continue
            tokens.append(lemma)
        return tokens

    # Fallback: simple regex-based tokenization aligned with our NLTK pipeline.
    tokens = []
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0)
        if len(token) < 3:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    if not os.path.exists(EMBEDDINGS_PATH):
        logger.error("Embeddings file not found at %s", EMBEDDINGS_PATH)
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    df = pd.read_parquet(EMBEDDINGS_PATH)
    if "embedding" not in df.columns:
        raise ValueError("Embeddings parquet must contain an 'embedding' column")
    embeddings = np.vstack(df["embedding"].to_list())
    return df, embeddings


def load_cleaned_articles() -> pd.DataFrame:
    if not os.path.exists(CLEANED_ARTICLES_PATH):
        logger.error("Cleaned articles file not found at %s", CLEANED_ARTICLES_PATH)
        raise FileNotFoundError(f"Cleaned articles file not found at {CLEANED_ARTICLES_PATH}")
    return pd.read_parquet(CLEANED_ARTICLES_PATH)


class AgglomerativeWrapper:
    """
    Wrapper for AgglomerativeClustering that exposes cluster_centers_ attribute.
    
    This is needed because AgglomerativeClustering doesn't expose explicit cluster
    centers, but downstream code expects a `.cluster_centers_` attribute.
    
    NOTE: Must be defined at module level (not inside a function) so it can be
    pickled by joblib when saving the model.
    """

    def __init__(self, centers: np.ndarray, labels: np.ndarray, n_clusters: int):
        """
        Initialize wrapper with computed cluster centers.
        
        Args:
            centers: Cluster centers array (n_clusters, n_features)
            labels: Cluster labels for each sample
            n_clusters: Number of clusters
        """
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters = n_clusters
        self.method = "agglomerative"


def make_clusterer(embeddings: np.ndarray, cfg: Dict):
    method = (cfg.get("method") or "kmeans").lower()
    n_clusters = int(cfg.get("n_clusters", 100))
    random_state = int(cfg.get("random_state", 42))

    # NOTE:
    #   - We keep KMeans/MiniBatchKMeans support for tests and backward
    #     compatibility.
    #   - For hierarchical/topic-exploration use-cases, the recommended
    #     setting is `method: "agglomerative"` so we can build on top of a
    #     hierarchical clustering structure.
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        logger.info("Fitting MiniBatchKMeans with n_clusters=%d", n_clusters)
        labels = model.fit_predict(embeddings)
        return model, labels

    if method == "agglomerative":
        logger.info("Fitting AgglomerativeClustering with n_clusters=%d", n_clusters)
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(embeddings)

        # AgglomerativeClustering does not expose explicit cluster centers,
        # but downstream code (and tests) expect a `.cluster_centers_`
        # attribute. We compute simple centroids per cluster label.
        centers: List[np.ndarray] = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                # If a cluster ended up empty (can happen in edge cases),
                # fall back to a zero vector with the correct dimensionality.
                centers.append(np.zeros(embeddings.shape[1], dtype=embeddings.dtype))
            else:
                centers.append(embeddings[mask].mean(axis=0))
        centers_arr = np.vstack(centers)

        # Wrap in a lightweight object exposing `cluster_centers_` so the
        # rest of the pipeline (and tests) can stay unchanged.
        # NOTE: AgglomerativeWrapper is defined at module level so it can be pickled.
        model = AgglomerativeWrapper(centers_arr, labels, n_clusters)
        return model, labels

    # Default: standard KMeans
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    logger.info("Fitting KMeans with n_clusters=%d", n_clusters)
    labels = model.fit_predict(embeddings)
    return model, labels


def build_nn_index(embeddings: np.ndarray, cfg: Dict) -> NearestNeighbors:
    n_neighbors = int(cfg.get("n_neighbors", 10))
    algorithm = cfg.get("algorithm", "auto")
    logger.info("Building NearestNeighbors index (n_neighbors=%d, algorithm=%s)", n_neighbors, algorithm)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
    nn.fit(embeddings)
    return nn


def compute_cluster_summaries(
    titles: List[str],
    cleaned_texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    top_k_keywords: int = 20,
    top_k_articles: int = 10,
) -> pd.DataFrame:
    """
    Compute cluster summaries using c-TF-IDF (class-based TF-IDF):
      - size: number of articles in cluster
      - top keywords: terms ranked by c-TF-IDF score (normalized TF × IDF across clusters)
      - representative articles: articles closest to cluster centroid
    
    c-TF-IDF treats each cluster as a "class" and computes:
      - TF(term, cluster) = count(term) / total_tokens_in_cluster (normalized)
      - IDF(term) = log(N_clusters / CF(term)) where CF is class frequency
      - c-TF-IDF(term, cluster) = TF(term, cluster) × IDF(term)
    
    This emphasizes terms that are both frequent within a cluster AND
    distinctive compared to other clusters.
    """
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_to_indices[int(lbl)].append(idx)

    # ------------------------------------------------------------------
    # c-TF-IDF (class-based TF-IDF) computation:
    # Treats each cluster as a "class" and computes:
    #   - TF(term, cluster)  = normalized frequency: count(term) / total_tokens_in_cluster
    #   - DF(term)           = number of clusters (classes) where term appears
    #   - IDF(term)          = log(N_clusters / DF(term))
    #   - c-TF-IDF(term, cluster) = TF(term, cluster) × IDF(term)
    #
    # This emphasizes terms that are frequent within a cluster AND
    # distinctive compared to other clusters.
    # ------------------------------------------------------------------
    cluster_token_counters: Dict[int, Counter] = {}
    cluster_token_sets: Dict[int, Set[str]] = {}
    cluster_total_tokens: Dict[int, int] = {}

    for cluster_id, indices in cluster_to_indices.items():
        token_counter: Counter = Counter()
        token_set: Set[str] = set()
        total_tokens = 0
        for i in indices:
            text = cleaned_texts[i] or ""
            tokens = list(_tokenize(text))
            for tok in tokens:
                token_counter[tok] += 1
                token_set.add(tok)
                total_tokens += 1
        cluster_token_counters[cluster_id] = token_counter
        cluster_token_sets[cluster_id] = token_set
        cluster_total_tokens[cluster_id] = total_tokens

    # Compute class frequency (CF) across clusters: how many clusters contain each term
    cf_counter: Counter = Counter()
    for token_set in cluster_token_sets.values():
        cf_counter.update(token_set)

    n_clusters = max(len(cluster_token_sets), 1)
    idf: Dict[str, float] = {}
    for term, cf in cf_counter.items():
        # Standard c-TF-IDF IDF formula: log(N_classes / CF(term))
        # Add small epsilon to avoid division by zero
        if cf > 0:
            idf[term] = math.log(n_clusters / cf)
        else:
            idf[term] = 0.0

    # ------------------------------------------------------------------
    # Second pass: build output rows using c-TF-IDF ranking for "keywords"
    # and Euclidean distance to cluster center for representative articles.
    # ------------------------------------------------------------------
    rows: List[Dict] = []
    for cluster_id, indices in tqdm(
        cluster_to_indices.items(),
        total=len(cluster_to_indices),
        desc="Summarizing clusters",
    ):
        # Representative/top articles by proximity to centroid
        cluster_embeddings = embeddings[indices]
        center = centers[cluster_id].reshape(1, -1)
        dists = pairwise_distances(cluster_embeddings, center, metric="euclidean").ravel()
        order = np.argsort(dists)
        rep_indices = [indices[i] for i in order[:top_k_articles]]
        top_articles = [titles[i] for i in rep_indices]

        # c-TF-IDF based topic words
        token_counts = cluster_token_counters.get(cluster_id, Counter())
        total_tokens_in_cluster = cluster_total_tokens.get(cluster_id, 1)
        
        if token_counts and total_tokens_in_cluster > 0:
            # Normalize TF by cluster size: TF = count / total_tokens_in_cluster
            c_tfidf_scores = {
                term: (count / total_tokens_in_cluster) * idf.get(term, 0.0)
                for term, count in token_counts.items()
            }
            # Sort by c-TF-IDF descending and keep the top_k_keywords terms
            sorted_terms = sorted(c_tfidf_scores.items(), key=lambda kv: kv[1], reverse=True)
            keywords = [term for term, _ in sorted_terms[:top_k_keywords]]
        else:
            keywords = []

        rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "keywords": keywords,
                "top_articles": top_articles,
            }
        )

    return pd.DataFrame(rows)


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    setup_logging()
    logger.info("Starting topic clustering pipeline")

    try:
        config = load_config(CONFIG_PATH)
        model_cfg = config.get("models", {}).get("clustering", {})
        nn_cfg = config.get("models", {}).get("neighbors", {})

        emb_df, embeddings = load_embeddings()
        cleaned_df = load_cleaned_articles()

        if len(emb_df) != len(cleaned_df):
            logger.warning(
                "Embeddings and cleaned articles have different lengths: %d vs %d",
                len(emb_df),
                len(cleaned_df),
            )

        titles = emb_df["title"].astype(str).tolist()
        cleaned_texts = cleaned_df["cleaned_text"].astype(str).tolist()

        logger.info(
            "Clustering %d articles into ~%s clusters (method=%s)...",
            len(titles),
            model_cfg.get("n_clusters", "unknown"),
            model_cfg.get("method", "kmeans"),
        )
        model, labels = make_clusterer(embeddings, model_cfg)

        logger.info("Building nearest-neighbor index for similar-article lookup...")
        nn_index = build_nn_index(embeddings, nn_cfg)

        # Prepare outputs
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, KMEANS_MODEL_PATH)
        joblib.dump(nn_index, NN_INDEX_PATH)

        assignments_df = pd.DataFrame({"title": titles, "cluster_id": labels.astype(int)})
        assignments_df.to_parquet(CLUSTER_ASSIGNMENTS_PATH, index=False)

        centers = model.cluster_centers_
        summaries_df = compute_cluster_summaries(
            titles=titles,
            cleaned_texts=cleaned_texts,
            embeddings=embeddings,
            labels=labels,
            centers=centers,
        )
        summaries_df.to_parquet(CLUSTERS_SUMMARY_PATH, index=False)

        # Basic metrics
        metrics = {
            "n_articles": int(len(titles)),
            "n_clusters": int(len(summaries_df)),
        }
        save_json(metrics, METRICS_PATH)

        # Optional MLflow logging
        try:
            import mlflow

            ml_cfg = config.get("mlops", {}).get("mlflow", {})
            tracking_uri = ml_cfg.get("tracking_uri")
            experiment_name = ml_cfg.get("experiment_name", "wikiinsight")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="cluster_topics"):
                for key, value in model_cfg.items():
                    mlflow.log_param(f"clustering_{key}", value)
                for key, value in nn_cfg.items():
                    mlflow.log_param(f"neighbors_{key}", value)
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging for clustering skipped or failed: %s", exc)

        logger.info("Topic clustering complete. Wrote artifacts to %s", MODEL_DIR)
    except FileNotFoundError as exc:
        logger.error(
            "Clustering failed because required inputs are missing: %s. "
            "Make sure the preprocessing stage has completed successfully "
            "and produced both embeddings and cleaned_articles parquet files.",
            exc,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Clustering pipeline failed: %s. Consider reducing 'n_clusters' in config.yaml "
            "or re-running preprocessing to regenerate embeddings.",
            exc,
        )
        raise


if __name__ == "__main__":
    main()


