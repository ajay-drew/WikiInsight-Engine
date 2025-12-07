"""
Async script to fetch a corpus of Wikipedia articles using concurrent API calls.

This is the main ingestion entrypoint that uses AsyncWikipediaClient to fetch
multiple articles concurrently, significantly speeding up data ingestion.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Iterable, List

from src.common.logging_utils import setup_logging
from .wikipedia_client_async import AsyncWikipediaClient
from .constants import RAW_DATA_PATH, SEED_QUERIES

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
    max_articles: int = 100,
    per_query_limit: int = 50,
    batch_size: int = 10,
    max_workers: int = 10,
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

    try:
        # Process all queries concurrently
        search_tasks = [
            client.search_articles(query, limit=per_query_limit) for query in SEED_QUERIES
        ]
        all_search_results = await asyncio.gather(*search_tasks)

        # Collect all unique titles
        titles_to_fetch: List[str] = []
        for query_idx, search_results in enumerate(all_search_results):
            query = SEED_QUERIES[query_idx]
            logger.info("Found %d search results for '%s'", len(search_results), query)
            
            for result in search_results:
                title = result.get("title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    titles_to_fetch.append(title)
                    
                    if len(titles_to_fetch) >= max_articles:
                        break
            
            if len(titles_to_fetch) >= max_articles:
                break

        if not titles_to_fetch:
            logger.warning("No articles to fetch")
            return []

        # Limit to max_articles
        titles_to_fetch = titles_to_fetch[:max_articles]
        logger.info("Fetching %d articles concurrently...", len(titles_to_fetch))

        # Fetch ALL articles concurrently (not in batches)
        batch_articles = await client.get_articles_batch(
            titles_to_fetch, fetch_links=False, fetch_categories=False
        )

        # Process and normalize articles
        for article in batch_articles:
            if not article:
                continue

            normalized = _normalize_article(article)
            if not normalized.get("title") or not normalized.get("text"):
                continue

            articles.append(normalized)

            if len(articles) % 10 == 0:
                logger.info("Processed %d articles so far...", len(articles))

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


async def main_async() -> None:
    """Async main function."""
    setup_logging()
    logger.info("Starting async Wikipedia ingestion")
    articles = await fetch_corpus_async()
    save_articles(articles, RAW_DATA_PATH)
    logger.info("Ingestion complete")


def main() -> None:
    """Main entry point that runs async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
