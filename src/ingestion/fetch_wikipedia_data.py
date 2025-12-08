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
from .constants import RAW_DATA_PATH, SEED_QUERIES
from .wikipedia_client_async import AsyncWikipediaClient

logger = logging.getLogger(__name__)


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
    stub_min_words: int = 200,
    existing_articles: Optional[List[Dict]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1000,
) -> List[Dict]:
    """
    Fetch a corpus of Wikipedia articles concurrently (no sequential requests).

    Args:
        max_articles: Maximum number of unique articles to fetch.
        per_query_limit: Maximum search results per seed query.
        batch_size: Number of articles to fetch concurrently in each batch.
        max_workers: Maximum number of concurrent workers.

    Returns:
        List of normalized article dictionaries.
    """
    logger.info(
        "Initializing async corpus fetch (max_articles=%d, per_query_limit=%d, batch_size=%d, max_workers=%d)",
        max_articles,
        per_query_limit,
        batch_size,
        max_workers,
    )
    client = AsyncWikipediaClient(max_workers=max_workers)
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
        # Process all queries concurrently
        search_tasks = [
            client.search_articles(query, limit=per_query_limit) for query in SEED_QUERIES
        ]
        all_search_results = await asyncio.gather(*search_tasks)

        # Collect all unique titles
        titles_to_fetch: List[str] = []
        current_count = len(articles)
        for query_idx, search_results in enumerate(all_search_results):
            query = SEED_QUERIES[query_idx]
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
            return articles

        checkpoint_path = checkpoint_path or RAW_DATA_PATH

        # Fetch articles in batches so we can report progress and checkpoint regularly.
        progress = tqdm(total=len(titles_to_fetch), desc="Fetching articles", unit="article")
        try:
            for i in range(0, len(titles_to_fetch), batch_size):
                batch_titles = titles_to_fetch[i : i + batch_size]
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

    except Exception:  # noqa: BLE001
        logger.exception("Error during async corpus fetch")
    finally:
        # Cleanup executor
        client.executor.shutdown(wait=True)

    logger.info("Fetched %d articles in total", len(articles))
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
    """Async main function."""
    setup_logging()
    logger.info("Starting async Wikipedia ingestion")

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

    articles = await fetch_corpus_async(
        max_articles=max_articles,
        per_query_limit=per_query_limit,
        batch_size=batch_size,
        max_workers=max_workers,
        existing_articles=existing_articles,
        checkpoint_path=RAW_DATA_PATH,
    )
    save_articles(articles, RAW_DATA_PATH)
    logger.info("Ingestion complete")


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
        help="Maximum number of concurrent worker threads for mwclient calls (default: 10).",
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
