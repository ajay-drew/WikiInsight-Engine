"""
Async script to fetch a corpus of Wikipedia articles using concurrent API calls.

This is the main ingestion entrypoint that uses AsyncWikipediaClient to fetch
multiple articles concurrently, significantly speeding up data ingestion.
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from src.common.logging_utils import setup_logging
from src.common.pipeline_progress import update_progress, reset_progress, mark_stage_completed, mark_stage_error
from .wikipedia_client_async import AsyncWikipediaClient

# Constants (previously from .constants)
RAW_DATA_PATH = os.path.join("data", "raw", "articles.json")

logger = logging.getLogger(__name__)


def load_seed_queries(config_path: str = "config.yaml") -> List[str]:
    """
    Load seed queries from config.yaml and validate.

    Args:
        config_path: Path to config.yaml

    Returns:
        List of seed query strings (3-6 queries)

    Raises:
        ValueError: If queries are invalid (not 3-6 queries)
    """
    import yaml

    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    queries = config.get("ingestion", {}).get("seed_queries", [])

    if not isinstance(queries, list):
        raise ValueError("seed_queries must be a list")

    # Filter out empty strings and strip whitespace
    queries = [q.strip() for q in queries if q and q.strip()]

    if len(queries) < 3:
        raise ValueError(f"Must have at least 3 seed queries, got {len(queries)}")
    if len(queries) > 6:
        raise ValueError(f"Must have at most 6 seed queries, got {len(queries)}")

    return queries[:6]  # Enforce max 6


def validate_ingestion_config(
    queries: List[str], per_query_limit: int, max_articles: int
) -> None:
    """
    Validate ingestion configuration.

    Args:
        queries: List of seed queries
        per_query_limit: Maximum articles per query (1-70)
        max_articles: Maximum total articles (hard cap: 1000, minimum: 50)

    Raises:
        ValueError: If configuration is invalid
    """
    if len(queries) < 3 or len(queries) > 6:
        raise ValueError(f"Must have 3-6 seed queries, got {len(queries)}")

    if per_query_limit < 1 or per_query_limit > 70:
        raise ValueError(f"per_query_limit must be between 1 and 70, got {per_query_limit}")

    if max_articles < 50:
        raise ValueError(
            f"max_articles must be at least 50 for meaningful clustering, got {max_articles}. "
            "Please increase max_articles to at least 50 or adjust your seed queries/per_query_limit."
        )

    if max_articles > 1000:
        raise ValueError(f"max_articles cannot exceed 1000, got {max_articles}")

    # Check total potential doesn't exceed max_articles
    total_potential = len(queries) * per_query_limit
    if total_potential > max_articles:
        logger.warning(
            "Total potential articles (%d) exceeds max_articles (%d). "
            "System will cap at %d articles.",
            total_potential,
            max_articles,
            max_articles,
        )


def _normalize_article(raw: Dict) -> Dict:
    """
    Normalize raw article data into a minimal schema.

    Expected keys in `raw` (from WikipediaClient.get_article):
      - title: str
      - text: str
      - categories: iterable
      - links: iterable
    """
    title = raw.get("title") or raw.get("name")
    text = raw.get("text") or ""

    # Convert generators/iterables to simple lists of strings where possible
    def _to_list(values: Iterable) -> List[str]:
        items: List[str] = []
        for v in values or []:
            try:
                # mwclient category/link objects often have a .name attribute
                name = getattr(v, "name", None)
                if name is not None:
                    items.append(str(name))
                else:
                    items.append(str(v))
            except Exception:
                continue
        return items

    categories = _to_list(raw.get("categories"))
    links = _to_list(raw.get("links"))

    return {
        "title": title,
        "text": text,
        "categories": categories,
        "links": links,
    }


async def fetch_corpus_async(
    max_articles: int = 1000,
    per_query_limit: int = 50,
    batch_size: int = 20,
    max_workers: int = 10,
    *,
    seed_queries: Optional[List[str]] = None,
    stub_min_words: int = 200,
    existing_articles: Optional[List[Dict]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1000,
    rate_limit: float = 50.0,
) -> List[Dict]:
    """
    Fetch a corpus of Wikipedia articles concurrently with rate limiting.

    Args:
        max_articles: Maximum number of unique articles to fetch.
        per_query_limit: Maximum search results per seed query.
        batch_size: Number of articles to fetch concurrently in each batch.
        max_workers: Maximum number of concurrent workers.
        seed_queries: List of seed queries to search (3-6 queries). If None, loads from config.
        rate_limit: Maximum requests per second (default: 50.0, conservative for unregistered bots).

    Returns:
        List of normalized article dictionaries.
    """
    # Load queries if not provided
    if seed_queries is None:
        seed_queries = load_seed_queries()

    # Validate configuration
    validate_ingestion_config(seed_queries, per_query_limit, max_articles)
    # Optimize batch size and workers based on rate limit
    # We want to maximize throughput while staying under the limit
    # Each article fetch makes multiple API calls (info, text, revisions, links, categories)
    # So we need to be more conservative with concurrency
    # Use ~30% of rate limit for batch size and ~40% for workers to account for multiple API calls per article
    optimal_batch_size = min(batch_size, max(1, int(rate_limit * 0.3)))  # Process batches at ~30% of rate limit for safety
    optimal_workers = min(max_workers, max(1, int(rate_limit * 0.4)))  # Use up to 40% of rate limit for concurrency
    
    logger.info(
        "Initializing async corpus fetch (max_articles=%d, per_query_limit=%d, batch_size=%d->%d, max_workers=%d->%d, rate_limit=%.1f req/sec)",
        max_articles,
        per_query_limit,
        batch_size,
        optimal_batch_size,
        max_workers,
        optimal_workers,
        rate_limit,
    )
    client = AsyncWikipediaClient(max_workers=optimal_workers, rate_limit=rate_limit)
    seen_titles: set[str] = set()
    articles: List[Dict] = []

    # If we are resuming from an existing raw file, keep those articles and
    # avoid re-fetching their titles.
    if existing_articles:
        for art in existing_articles:
            title = art.get("title")
            if title:
                seen_titles.add(str(title))
                articles.append(art)

        logger.info(
            "Loaded %d existing articles; will fetch up to %d total.",
            len(articles),
            max_articles,
        )

    try:
        # Update progress: starting ingestion
        update_progress("ingestion", "running", 0.0, "Searching Wikipedia for articles...")

        # Process all queries concurrently
        search_tasks = [
            client.search_articles(query, limit=per_query_limit) for query in seed_queries
        ]
        all_search_results = await asyncio.gather(*search_tasks)

        # Collect all unique titles
        titles_to_fetch: List[str] = []
        current_count = len(articles)
        total_search_results = 0
        for query_idx, search_results in enumerate(all_search_results):
            query = seed_queries[query_idx]
            total_search_results += len(search_results)
            logger.info("Found %d search results for '%s'", len(search_results), query)
            
            for result in search_results:
                title = result.get("title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    titles_to_fetch.append(title)
                    
                    if current_count + len(titles_to_fetch) >= max_articles:
                        break
            
            if current_count + len(titles_to_fetch) >= max_articles:
                break

        if not titles_to_fetch:
            logger.warning("No articles to fetch")
            return []

        # Limit to max_articles (taking into account any existing/resumed articles)
        remaining_budget = max(0, max_articles - current_count)
        titles_to_fetch = titles_to_fetch[:remaining_budget]
        logger.info("Fetching %d new articles concurrently...", len(titles_to_fetch))

        if not titles_to_fetch:
            logger.info("No new titles to fetch (budget already satisfied by existing articles).")
            mark_stage_completed("ingestion", "No new articles to fetch")
            return articles

        checkpoint_path = checkpoint_path or RAW_DATA_PATH

        # Update progress: found articles, starting fetch
        update_progress(
            "ingestion",
            "running",
            5.0,
            f"Found {len(titles_to_fetch)} unique articles. Fetching content...",
        )

        # Fetch articles in batches so we can report progress and checkpoint regularly.
        # Use optimized batch size for better rate limit utilization
        progress = tqdm(total=len(titles_to_fetch), desc="Fetching articles", unit="article")
        try:
            for i in range(0, len(titles_to_fetch), optimal_batch_size):
                batch_titles = titles_to_fetch[i : i + optimal_batch_size]
                # Process batch with rate limiting built into the client
                batch_articles = await client.get_articles_batch(
                    batch_titles,
                    fetch_links=False,
                    fetch_categories=False,
                    max_links=50,
                )

                # Process and normalize articles in this batch
                for article in batch_articles:
                    progress.update(1)
                    if not article:
                        continue

                    normalized = _normalize_article(article)
                    text = normalized.get("text") or ""
                    # Filter out stub articles with very few words to avoid
                    # extremely short pages that add little value.
                    if len(text.split()) < stub_min_words:
                        continue

                    if not normalized.get("title") or not text:
                        continue

                    articles.append(normalized)

                    # Update progress percentage
                    progress_pct = min(95.0, 5.0 + (len(articles) / len(titles_to_fetch)) * 90.0)
                    update_progress(
                        "ingestion",
                        "running",
                        progress_pct,
                        f"Fetched {len(articles)}/{len(titles_to_fetch)} articles...",
                    )

                    # Periodically checkpoint progress so long runs can resume.
                    if checkpoint_path and len(articles) % checkpoint_interval == 0:
                        logger.info(
                            "Checkpoint reached at %d articles; flushing to %s",
                            len(articles),
                            checkpoint_path,
                        )
                        save_articles(articles, checkpoint_path)

                logger.info("Processed %d/%d titles so far...", min(i + batch_size, len(titles_to_fetch)), len(titles_to_fetch))
        finally:
            progress.close()

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during async corpus fetch")
        mark_stage_error("ingestion", f"Ingestion failed: {str(exc)}")
        raise
    finally:
        # Cleanup executor
        client.executor.shutdown(wait=True)

    logger.info("Fetched %d articles in total", len(articles))
    mark_stage_completed("ingestion", f"Successfully fetched {len(articles)} articles")
    return articles


def save_articles(articles: List[Dict], path: str = RAW_DATA_PATH) -> None:
    """Save articles as newline-delimited JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    logger.info("Saved %d articles to %s", len(articles), path)


async def main_async(
    max_articles: int,
    per_query_limit: int,
    batch_size: int,
    max_workers: int,
    sample: Optional[int],
    resume: bool,
) -> None:
    """Async main function with MLflow logging."""
    from time import perf_counter
    
    setup_logging()
    logger.info("Starting async Wikipedia ingestion")
    
    # Reset progress tracking
    reset_progress()
    
    start_time = perf_counter()

    if sample is not None:
        max_articles = min(max_articles, sample)
        logger.info("Sample mode enabled; limiting to %d articles", max_articles)

    existing_articles: Optional[List[Dict]] = None
    if resume and os.path.exists(RAW_DATA_PATH):
        logger.info("Resume mode enabled; loading existing articles from %s", RAW_DATA_PATH)
        existing_articles = []
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        logger.info("Loaded %d existing articles from previous run", len(existing_articles))

    # Load rate limit, seed queries, max_articles, per_query_limit, and performance settings from config
    import yaml
    config_path = "config.yaml"
    rate_limit = 50.0  # Default - conservative for unregistered bots
    seed_queries = None
    config_max_workers = max_workers  # Use CLI arg as default
    config_max_articles = max_articles  # Use CLI arg as default
    config_per_query_limit = per_query_limit  # Use CLI arg as default
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            rate_limit = float(config.get("data", {}).get("wikipedia", {}).get("api_rate_limit", 50.0))
            logger.info("Loaded rate limit from config: %.1f req/sec", rate_limit)
            
            # Load seed queries
            seed_queries = load_seed_queries(config_path)
            logger.info("Loaded %d seed queries from config: %s", len(seed_queries), seed_queries)
            
            # Load max_articles from config if CLI arg is default (1000) or when called from pipeline
            ingestion_cfg = config.get("ingestion", {})
            if max_articles == 1000:  # Default CLI value - prefer config
                config_max_articles = int(ingestion_cfg.get("max_articles", 1000))
                logger.info("Using max_articles from config: %d (CLI default was %d)", config_max_articles, max_articles)
            elif "max_articles" in ingestion_cfg:
                # Even if CLI arg is provided, log what's in config for visibility
                logger.info("CLI max_articles=%d, config has max_articles=%d (using CLI value)", 
                           max_articles, ingestion_cfg.get("max_articles"))
            
            # Load per_query_limit from config if CLI arg is default (50) or when called from pipeline
            if per_query_limit == 50:  # Default CLI value - prefer config
                config_per_query_limit = int(ingestion_cfg.get("per_query_limit", 50))
                logger.info("Using per_query_limit from config: %d (CLI default was %d)", config_per_query_limit, per_query_limit)
            elif "per_query_limit" in ingestion_cfg:
                logger.info("CLI per_query_limit=%d, config has per_query_limit=%d (using CLI value)",
                           per_query_limit, ingestion_cfg.get("per_query_limit"))
            
            # Load max_workers from performance config (if CLI arg is default)
            perf_cfg = config.get("performance", {})
            if max_workers == 10:  # Default CLI value
                config_max_workers = int(perf_cfg.get("max_workers", 8))
                logger.info("Using max_workers from config: %d", config_max_workers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load config, using defaults: %s", exc)
    
    # Use configured values (config takes precedence when CLI uses defaults)
    max_articles = config_max_articles
    per_query_limit = config_per_query_limit
    max_workers = config_max_workers
    
    logger.info("Final ingestion parameters: max_articles=%d, per_query_limit=%d, max_workers=%d",
               max_articles, per_query_limit, max_workers)

    articles = await fetch_corpus_async(
        max_articles=max_articles,
        per_query_limit=per_query_limit,
        batch_size=batch_size,
        max_workers=max_workers,
        seed_queries=seed_queries,
        existing_articles=existing_articles,
        checkpoint_path=RAW_DATA_PATH,
        rate_limit=rate_limit,
    )
    save_articles(articles, RAW_DATA_PATH)
    
    duration = perf_counter() - start_time
    logger.info("Ingestion complete in %.2f seconds", duration)
    
    # Calculate metrics for MLflow
    total_articles = len(articles)
    total_words = sum(len(article.get("text", "").split()) for article in articles)
    avg_words = total_words / total_articles if total_articles > 0 else 0
    total_links = sum(len(article.get("links", [])) for article in articles)
    avg_links = total_links / total_articles if total_articles > 0 else 0
    
    # Count stubs (articles with < 200 words)
    stub_count = sum(1 for article in articles if len(article.get("text", "").split()) < 200)
    
    # Optional MLflow logging (wrapped in try-except to prevent crashes)
    try:
        from src.common.mlflow_utils import start_mlflow_run, log_params_safely, log_metrics_safely
        
        with start_mlflow_run("ingestion"):
            # Log parameters
            log_params_safely({
                "max_articles": max_articles,
                "per_query_limit": per_query_limit,
                "batch_size": batch_size,
                "max_workers": max_workers,
                "sample": sample is not None,
                "resume": resume,
            })
            
            # Log metrics
            log_metrics_safely({
                "articles_fetched": total_articles,
                "articles_filtered_stubs": stub_count,
                "avg_article_length_words": avg_words,
                "total_words": total_words,
                "total_links": total_links,
                "avg_links_per_article": avg_links,
                "fetch_duration_seconds": duration,
                "articles_per_second": total_articles / duration if duration > 0 else 0,
            })
            
            logger.info("Logged ingestion metrics to MLflow")
    except ImportError:
        logger.debug("MLflow not installed, skipping logging")
    except Exception as exc:
        # Catch-all to ensure MLflow errors never crash the pipeline
        logger.warning("MLflow logging skipped or failed (non-critical): %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a corpus of Wikipedia articles.")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=1000,
        help="Maximum number of articles to fetch (default: 1000).",
    )
    parser.add_argument(
        "--per-query-limit",
        type=int,
        default=50,
        help="Maximum search results per seed query (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of articles to fetch concurrently in each batch (default: 20).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum concurrent workers (default: 10, or from config.yaml performance.max_workers).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional quick-sample mode; if set, limits the run to this many articles.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing raw articles file if present, instead of starting from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point that parses CLI arguments and runs the async function."""
    args = parse_args()
    asyncio.run(
        main_async(
            max_articles=args.max_articles,
            per_query_limit=args.per_query_limit,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            sample=args.sample,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
