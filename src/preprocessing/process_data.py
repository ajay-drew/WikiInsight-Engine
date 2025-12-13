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
from tqdm import tqdm

from src.common.logging_utils import setup_logging
from .embeddings import EmbeddingGenerator
from .nltk_utils import normalize_text
from .text_processor import TextProcessor

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

    for art in tqdm(articles, desc="Cleaning articles", unit="article"):
        title = art.get("title")
        text = art.get("text", "") or ""

        # First apply lightweight structural cleaning (remove wiki markup).
        cleaned_text = processor.clean_text(text)

        # Then apply NLTK-based normalization (lowercase, stopwords, lemma/stem)
        # on the cleaned text. If NLTK or its data is unavailable, this will
        # gracefully fall back to a regex-based normalizer.
        nltk_cleaned = normalize_text(cleaned_text)

        records.append(
            {
                "title": title,
                "raw_text": text,
                "cleaned_text": cleaned_text,
                "nltk_cleaned_text": nltk_cleaned,
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
    # Prefer the NLTK-normalized text when available, otherwise fall back to
    # the older cleaned_text column for backward compatibility.
    source_column = (
        "nltk_cleaned_text" if "nltk_cleaned_text" in cleaned_df.columns else "cleaned_text"
    )
    texts = cleaned_df[source_column].fillna("").astype(str).tolist()
    logger.info(
        "Generating embeddings for %d articles using model '%s' (batch_size=%d). "
        "This step can take several minutes for large corpora.",
        len(texts),
        model_name,
        batch_size,
    )
    generator = EmbeddingGenerator(model_name=model_name)
    # encode_batch has show_progress_bar=True, so it will show progress
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

    try:
        config = load_config(CONFIG_PATH)
        emb_cfg = config.get("preprocessing", {}).get("embeddings", {})
        model_name = emb_cfg.get("model", "all-MiniLM-L6-v2")
        batch_size = int(emb_cfg.get("batch_size", 32))

        articles = load_raw_articles(RAW_DATA_PATH)
        cleaned_df = clean_articles(articles)
        emb_df = generate_embeddings(cleaned_df, model_name=model_name, batch_size=batch_size)
        save_parquet(cleaned_df, CLEANED_ARTICLES_PATH)
        save_parquet(emb_df, EMBEDDINGS_PATH)

        # Enhanced MLflow logging
        from time import perf_counter
        from src.common.mlflow_utils import (
            log_metrics_safely,
            log_params_safely,
            start_mlflow_run,
        )
        
        emb_start_time = perf_counter()
        
        try:
            # Calculate additional metrics
            avg_text_length_before = sum(len(art.get("text", "")) for art in articles) / len(articles) if articles else 0
            avg_text_length_after = cleaned_df["cleaned_text"].str.len().mean() if "cleaned_text" in cleaned_df.columns else 0
            
            # Estimate vocabulary size (unique words)
            all_words = set()
            for text in cleaned_df.get("cleaned_text", []):
                if isinstance(text, str):
                    all_words.update(text.lower().split())
            vocab_size = len(all_words)
            
            embedding_duration = perf_counter() - emb_start_time
            
            with start_mlflow_run("preprocess_articles"):
                # Log parameters
                log_params_safely({
                    "embedding_model": model_name,
                    "embedding_batch_size": batch_size,
                })
                
                # Log metrics
                log_metrics_safely({
                    "n_articles": len(cleaned_df),
                    "avg_text_length_before": avg_text_length_before,
                    "avg_text_length_after": avg_text_length_after,
                    "vocabulary_size": vocab_size,
                    "embedding_duration_seconds": embedding_duration,
                    "articles_per_second": len(cleaned_df) / embedding_duration if embedding_duration > 0 else 0,
                })
                
                # Log embedding dimension if available
                if "embedding" in emb_df.columns and len(emb_df) > 0:
                    sample_emb = emb_df["embedding"].iloc[0]
                    if isinstance(sample_emb, list):
                        log_metrics_safely({"embedding_dimension": len(sample_emb)})
                
                logger.info("Logged preprocessing metrics to MLflow")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging skipped or failed: %s", exc)

        logger.info("Preprocessing complete")
    except FileNotFoundError as exc:
        logger.error(
            "Preprocessing failed because the raw articles file is missing: %s. "
            "Run the ingestion stage first (e.g. `python -m src.ingestion.fetch_wikipedia_data` "
            "or `dvc repro`).",
            exc,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Preprocessing pipeline failed: %s. Check the raw data file for corruption or "
            "try re-running ingestion with a smaller sample using the `--sample` flag.",
            exc,
        )
        raise


if __name__ == "__main__":
    main()


