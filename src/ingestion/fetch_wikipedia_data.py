"""
Script to fetch a corpus of Wikipedia articles and store them in `data/raw/articles.json`.

This is a lightweight ingestion entrypoint that:
- Uses `WikipediaClient` to fetch articles for a set of seed queries.
- Normalizes them into a minimal article schema.
- Writes newline-delimited JSON (one article per line) for downstream preprocessing.
"""

import json
import logging
import os
from typing import Dict, Iterable, List, Set

from tqdm import tqdm

from src.common.logging_utils import setup_logging
from .wikipedia_client import WikipediaClient

logger = logging.getLogger(__name__)


RAW_DATA_PATH = os.path.join("data", "raw", "articles.json")

# Simple, hard-coded seeds for now; can be extended or made configurable later.
SEED_QUERIES: List[str] = [
    "Machine learning",
    "Artificial intelligence",
    "Data science",
    "Physics",
    "Biology",
    "History",
]


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


def fetch_corpus(
    max_articles: int = 100,  # Reduced default for faster testing
    per_query_limit: int = 50,  # Reduced default for faster testing
) -> List[Dict]:
    """
    Fetch a corpus of Wikipedia articles based on seed queries.

    Args:
        max_articles: Maximum number of unique articles to fetch.
        per_query_limit: Maximum search results per seed query.
    """
    logger.info(
        "Initializing corpus fetch (max_articles=%d, per_query_limit=%d)",
        max_articles,
        per_query_limit,
    )
    client = WikipediaClient()
    seen_titles: Set[str] = set()
    articles: List[Dict] = []

    for query_idx, query in enumerate(SEED_QUERIES, 1):
        logger.info("Processing query %d/%d: '%s'", query_idx, len(SEED_QUERIES), query)
        try:
            search_results = client.search_articles(query, limit=per_query_limit)
            logger.info("Found %d search results for '%s'", len(search_results), query)
            
            for result in tqdm(search_results, desc=f"Fetching articles for '{query}'", leave=False):
                title = result.get("title")
                if not title or title in seen_titles:
                    continue

                seen_titles.add(title)
                article = client.get_article(title)
                if not article:
                    continue

                normalized = _normalize_article(article)
                if not normalized.get("title") or not normalized.get("text"):
                    continue

                articles.append(normalized)
                if len(articles) % 10 == 0:
                    logger.info("Fetched %d articles so far...", len(articles))
                
                if len(articles) >= max_articles:
                    logger.info("Reached max_articles=%d", max_articles)
                    return articles
        except Exception:  # noqa: BLE001
            logger.exception("Error while processing query '%s'", query)

    logger.info("Fetched %d articles in total", len(articles))
    return articles


def save_articles(articles: List[Dict], path: str = RAW_DATA_PATH) -> None:
    """Save articles as newline-delimited JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    logger.info("Saved %d articles to %s", len(articles), path)


def main() -> None:
    setup_logging()
    logger.info("Starting Wikipedia ingestion")
    articles = fetch_corpus()
    save_articles(articles, RAW_DATA_PATH)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()


