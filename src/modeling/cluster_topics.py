"""
Clustering pipeline for WikiInsight Engine topic explorer.

This module:
- Loads embeddings (and optionally cleaned articles) produced by `src.preprocessing.process_data`.
- Fits a clustering model (e.g., KMeans) according to `config.yaml`.
- Builds a k-NN index for similar-article lookup.
- Generates simple cluster summaries (keywords + representative articles).
- Saves artifacts under `models/clustering/` and a metrics JSON for DVC/CI.
"""

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

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


def make_clusterer(embeddings: np.ndarray, cfg: Dict):
    method = (cfg.get("method") or "kmeans").lower()
    n_clusters = int(cfg.get("n_clusters", 100))
    random_state = int(cfg.get("random_state", 42))

    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )

    logger.info("Fitting %s with n_clusters=%d", method, n_clusters)
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
    top_k_keywords: int = 10,
    top_k_articles: int = 10,
) -> pd.DataFrame:
    """
    Compute simple summaries for each cluster:
      - size
      - top keywords (by frequency in cleaned text)
      - representative articles (closest to cluster center)
    """
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_to_indices[int(lbl)].append(idx)

    rows: List[Dict] = []
    for cluster_id, indices in cluster_to_indices.items():
        cluster_embeddings = embeddings[indices]
        center = centers[cluster_id].reshape(1, -1)
        dists = pairwise_distances(cluster_embeddings, center, metric="euclidean").ravel()
        order = np.argsort(dists)

        # Representative/top articles
        rep_indices = [indices[i] for i in order[:top_k_articles]]
        top_articles = [titles[i] for i in rep_indices]

        # Keyword extraction from cleaned text (very simple)
        tokens: List[str] = []
        for i in indices:
            text = cleaned_texts[i] or ""
            tokens.extend(str(text).lower().split())

        counter = Counter(tokens)
        keywords = [w for w, _ in counter.most_common(top_k_keywords)]

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

    model, labels = make_clusterer(embeddings, model_cfg)
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


if __name__ == "__main__":
    main()


