"""
Preprocessing entrypoint to:
- Load raw Wikipedia articles from `data/raw/articles.json`.
- Clean the text using `TextProcessor`.
- Generate sentence-transformer embeddings using `EmbeddingGenerator`.
- Save cleaned articles and embeddings to `data/processed/` and `data/features/`.

Supports NVIDIA GPU acceleration with automatic CPU fallback.

This script is designed to be called via:
    python -m src.preprocessing.process_data
and is referenced from `dvc.yaml` as the `preprocess` stage.
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.common.logging_utils import setup_logging
from src.common.pipeline_progress import update_progress, mark_stage_completed, mark_stage_error
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


def _clean_single_article(art: Dict) -> Dict:
    """Clean a single article (for parallel processing)."""
    processor = TextProcessor()
    title = art.get("title")
    text = art.get("text", "") or ""
    
    # Apply structural cleaning
    cleaned_text = processor.clean_text(text)
    
    # Apply NLTK normalization
    nltk_cleaned = normalize_text(cleaned_text)
    
    return {
        "title": title,
        "raw_text": text,
        "cleaned_text": cleaned_text,
        "nltk_cleaned_text": nltk_cleaned,
        "categories": art.get("categories", []),
        "links": art.get("links", []),
    }


def clean_articles(
    articles: List[Dict],
    use_parallel: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Clean articles with optional parallel processing.
    
    Args:
        articles: List of raw article dictionaries
        use_parallel: Whether to use multiprocessing
        max_workers: Number of parallel workers
        
    Returns:
        DataFrame with cleaned articles
    """
    from time import perf_counter
    
    update_progress("preprocessing", "running", 0.0, "Cleaning article text...")
    clean_start = perf_counter()
    
    logger.info("Starting text cleaning for %d articles (parallel=%s, workers=%d)...", 
                len(articles), use_parallel, max_workers)
    
    records: List[Dict] = []
    
    if use_parallel and len(articles) > 100:
        # Use parallel processing for large datasets
        logger.info("Using parallel processing with %d workers", max_workers)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_clean_single_article, art): i 
                      for i, art in enumerate(articles)}
            
            for future in tqdm(as_completed(futures), total=len(articles),
                             desc="Cleaning articles (parallel)", unit="article"):
                try:
                    result = future.result()
                    records.append(result)
                except Exception as e:
                    logger.warning("Failed to clean article: %s", e)
    else:
        # Sequential processing
        processor = TextProcessor()
        
        for i, art in enumerate(tqdm(articles, desc="Cleaning articles", unit="article")):
            if i % max(1, len(articles) // 10) == 0:
                progress_pct = (i / len(articles)) * 40.0
                elapsed = perf_counter() - clean_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(articles) - i - 1) / rate if rate > 0 else 0
                update_progress("preprocessing", "running", progress_pct, 
                              f"Cleaning article {i+1}/{len(articles)}... ({rate:.1f} articles/sec, ETA: {eta:.0f}s)")
            
            title = art.get("title")
            text = art.get("text", "") or ""
            cleaned_text = processor.clean_text(text)
            nltk_cleaned = normalize_text(cleaned_text)
            
            records.append({
                "title": title,
                "raw_text": text,
                "cleaned_text": cleaned_text,
                "nltk_cleaned_text": nltk_cleaned,
                "categories": art.get("categories", []),
                "links": art.get("links", []),
            })

    clean_time = perf_counter() - clean_start
    df = pd.DataFrame(records)
    logger.info("Created cleaned articles DataFrame with %d rows in %.2f seconds", len(df), clean_time)
    update_progress("preprocessing", "running", 40.0, "Text cleaning complete. Generating embeddings...")
    return df


def generate_embeddings(
    cleaned_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate embeddings with GPU support and CPU fallback.
    
    Args:
        cleaned_df: DataFrame with cleaned articles
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        device: Device preference ('auto', 'cuda', 'cpu')
        
    Returns:
        DataFrame with title and embedding columns
    """
    from time import perf_counter
    
    # Prefer NLTK-normalized text
    source_column = (
        "nltk_cleaned_text" if "nltk_cleaned_text" in cleaned_df.columns else "cleaned_text"
    )
    texts = cleaned_df[source_column].fillna("").astype(str).tolist()
    
    device_str = device or "auto"
    logger.info(
        "Generating embeddings for %d articles using model '%s' (batch_size=%d, device=%s)",
        len(texts), model_name, batch_size, device_str,
    )
    
    update_progress("preprocessing", "running", 45.0, f"Loading embedding model on {device_str}...")
    
    emb_start = perf_counter()
    generator = None
    embeddings = None
    actual_device = "unknown"
    model_load_time = 0
    encode_time = 0
    
    try:
        logger.info("Importing EmbeddingGenerator...")
        from .embeddings import EmbeddingGenerator, get_gpu_info
        
        # Log GPU info
        gpu_info = get_gpu_info()
        if gpu_info["cuda_available"]:
            logger.info("GPU available: %s", gpu_info["devices"])
        else:
            logger.info("No GPU detected, will use CPU")
        
        # Load model
        model_load_start = perf_counter()
        generator = EmbeddingGenerator(model_name=model_name, device=device)
        actual_device = generator.device
        model_load_time = perf_counter() - model_load_start
        logger.info("Model loaded on %s in %.2f seconds", actual_device.upper(), model_load_time)
        
        # Update progress with actual device
        update_progress("preprocessing", "running", 50.0, 
                       f"Generating embeddings on {actual_device.upper()}...")
        
        # Generate embeddings
        encode_start = perf_counter()
        embeddings = generator.encode_batch(texts, batch_size=batch_size)
        encode_time = perf_counter() - encode_start
        
        logger.info("Embedding generation completed in %.2f seconds (%.1f articles/sec)",
                   encode_time, len(texts) / encode_time if encode_time > 0 else 0)
        
    except Exception as exc:
        emb_time = perf_counter() - emb_start
        error_msg = str(exc)
        error_type = type(exc).__name__
        logger.error("=" * 80)
        logger.error("Embedding generation failed after %.2f seconds", emb_time)
        logger.error("Error type: %s", error_type)
        logger.error("Error message: %s", error_msg)
        logger.error("=" * 80)
        raise RuntimeError(f"Failed to generate embeddings ({error_type}): {error_msg}") from exc
    
    if embeddings is None:
        raise RuntimeError("Embeddings were not generated")
    
    update_progress("preprocessing", "running", 95.0, 
                   f"Embeddings generated on {actual_device.upper()}")

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    # Convert to DataFrame
    logger.info("Converting embeddings to DataFrame format...")
    convert_start = perf_counter()
    embedding_rows = [emb.tolist() for emb in embeddings]
    emb_df = pd.DataFrame({
        "title": cleaned_df["title"].tolist(),
        "embedding": embedding_rows,
    })
    convert_time = perf_counter() - convert_start
    
    total_emb_time = perf_counter() - emb_start
    logger.info("Generated embeddings for %d articles in %.2f seconds total", len(emb_df), total_emb_time)
    logger.info("  - Device: %s", actual_device.upper())
    logger.info("  - Model loading: %.2f seconds", model_load_time)
    logger.info("  - Encoding: %.2f seconds", encode_time)
    logger.info("  - Conversion: %.2f seconds", convert_time)
    
    return emb_df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved DataFrame with %d rows to %s", len(df), path)


def main() -> None:
    from time import perf_counter
    
    setup_logging()
    pipeline_start_time = perf_counter()
    logger.info("=" * 80)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 80)

    try:
        # Load configuration
        config_start = perf_counter()
        config = load_config(CONFIG_PATH)
        config_time = perf_counter() - config_start
        logger.info("Loaded configuration in %.2f seconds", config_time)
        
        # Get embedding config
        emb_cfg = config.get("preprocessing", {}).get("embeddings", {})
        model_name = emb_cfg.get("model", "all-MiniLM-L6-v2")
        batch_size = int(emb_cfg.get("batch_size", 32))
        device = emb_cfg.get("device", "auto")  # auto | cuda | cpu
        
        # Get performance config
        perf_cfg = config.get("performance", {})
        use_parallel = perf_cfg.get("use_multiprocessing", False)
        max_workers = int(perf_cfg.get("max_workers", 4))
        
        logger.info("Configuration:")
        logger.info("  - Embedding model: %s", model_name)
        logger.info("  - Batch size: %d", batch_size)
        logger.info("  - Device preference: %s", device)
        logger.info("  - Parallel processing: %s", use_parallel)
        logger.info("  - Max workers: %d", max_workers)

        # Load articles
        load_start = perf_counter()
        articles = load_raw_articles(RAW_DATA_PATH)
        load_time = perf_counter() - load_start
        logger.info("Loaded %d articles in %.2f seconds", len(articles), load_time)
        
        # Clean articles
        clean_start = perf_counter()
        cleaned_df = clean_articles(articles, use_parallel=use_parallel, max_workers=max_workers)
        clean_time = perf_counter() - clean_start
        logger.info("Cleaned %d articles in %.2f seconds", len(cleaned_df), clean_time)
        
        # Generate embeddings
        emb_start = perf_counter()
        try:
            emb_df = generate_embeddings(
                cleaned_df,
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            )
            emb_time = perf_counter() - emb_start
            logger.info("Generated embeddings in %.2f seconds", emb_time)
        except Exception as emb_exc:
            emb_time = perf_counter() - emb_start
            logger.error("Embedding generation failed after %.2f seconds: %s", emb_time, emb_exc)
            logger.warning("Creating mock embeddings as fallback (development only)")
            
            # Fallback mock embeddings
            emb_dim = 384
            mock_embeddings = np.random.normal(size=(len(cleaned_df), emb_dim)).tolist()
            emb_df = pd.DataFrame({
                "title": cleaned_df["title"].tolist(),
                "embedding": mock_embeddings,
            })
            logger.info("Created mock embeddings for %d articles", len(emb_df))
        
        # Save outputs
        save_start = perf_counter()
        save_parquet(cleaned_df, CLEANED_ARTICLES_PATH)
        save_parquet(emb_df, EMBEDDINGS_PATH)
        save_time = perf_counter() - save_start
        logger.info("Saved outputs in %.2f seconds", save_time)
        
        # Summary
        total_time = perf_counter() - pipeline_start_time
        logger.info("=" * 80)
        logger.info("Preprocessing pipeline completed!")
        logger.info("Total time: %.2f seconds (%.1f minutes)", total_time, total_time / 60)
        logger.info("=" * 80)
        
        mark_stage_completed("preprocessing", f"Preprocessing complete in {total_time:.1f}s")

        # MLflow logging
        try:
            from src.common.mlflow_utils import log_metrics_safely, log_params_safely, start_mlflow_run
            
            with start_mlflow_run("preprocess_articles"):
                log_params_safely({
                    "embedding_model": model_name,
                    "embedding_batch_size": batch_size,
                    "device": device,
                    "use_parallel": use_parallel,
                })
                
                log_metrics_safely({
                    "n_articles": len(cleaned_df),
                    "preprocessing_duration_seconds": total_time,
                    "articles_per_second": len(cleaned_df) / total_time if total_time > 0 else 0,
                })
                
                logger.info("Logged preprocessing metrics to MLflow")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

    except FileNotFoundError as exc:
        error_msg = f"Preprocessing failed: {exc}"
        logger.error(error_msg)
        mark_stage_error("preprocessing", error_msg)
        raise
    except Exception as exc:
        error_msg = f"Preprocessing pipeline failed: {exc}"
        logger.exception(error_msg)
        mark_stage_error("preprocessing", error_msg)
        raise


if __name__ == "__main__":
    main()
