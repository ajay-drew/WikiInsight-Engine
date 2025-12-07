"""
Preprocessing entrypoint to:
- Load raw Wikipedia articles from `data/raw/articles.json`.
- Clean the text using `TextProcessor`.
- Generate sentence-transformer embeddings using `EmbeddingGenerator`.
- Save cleaned articles and embeddings to `data/processed/` and `data/features/`.

This script is designed to be called via:
    python -m src.preprocessing.process_data
and is referenced from `dvc.yaml` as the `preprocess` stage.
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.common.logging_utils import setup_logging
from .text_processor import TextProcessor
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw", "articles.json")
CLEANED_ARTICLES_PATH = os.path.join("data", "processed", "cleaned_articles.parquet")
EMBEDDINGS_PATH = os.path.join("data", "features", "embeddings.parquet")
CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_articles(path: str = RAW_DATA_PATH) -> List[Dict]:
    articles: List[Dict] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw articles file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d raw articles from %s", len(articles), path)
    return articles


def clean_articles(articles: List[Dict]) -> pd.DataFrame:
    processor = TextProcessor()
    records: List[Dict] = []

    for art in articles:
        title = art.get("title")
        text = art.get("text", "") or ""
        cleaned_text = processor.clean_text(text)

        records.append(
            {
                "title": title,
                "raw_text": text,
                "cleaned_text": cleaned_text,
                "categories": art.get("categories", []),
                "links": art.get("links", []),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Created cleaned articles DataFrame with %d rows", len(df))
    return df


def generate_embeddings(
    cleaned_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> pd.DataFrame:
    texts = cleaned_df["cleaned_text"].fillna("").astype(str).tolist()
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.encode_batch(texts, batch_size=batch_size)

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    # Store embeddings as lists so they serialize cleanly via Parquet
    embedding_rows = [emb.tolist() for emb in embeddings]
    emb_df = pd.DataFrame(
        {
            "title": cleaned_df["title"].tolist(),
            "embedding": embedding_rows,
        }
    )
    logger.info("Generated embeddings for %d articles", len(emb_df))
    return emb_df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved DataFrame with %d rows to %s", len(df), path)


def main() -> None:
    setup_logging()
    logger.info("Starting preprocessing pipeline")

    config = load_config(CONFIG_PATH)
    emb_cfg = config.get("preprocessing", {}).get("embeddings", {})
    model_name = emb_cfg.get("model", "all-MiniLM-L6-v2")
    batch_size = int(emb_cfg.get("batch_size", 32))

    articles = load_raw_articles(RAW_DATA_PATH)
    cleaned_df = clean_articles(articles)
    emb_df = generate_embeddings(cleaned_df, model_name=model_name, batch_size=batch_size)
    save_parquet(cleaned_df, CLEANED_ARTICLES_PATH)
    save_parquet(emb_df, EMBEDDINGS_PATH)

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

        with mlflow.start_run(run_name="preprocess_articles"):
            mlflow.log_param("embedding_model", model_name)
            mlflow.log_param("embedding_batch_size", batch_size)
            mlflow.log_metric("n_articles", len(cleaned_df))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging skipped or failed: %s", exc)

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()


