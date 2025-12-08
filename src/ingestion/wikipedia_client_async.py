"""
Async Wikipedia API client for concurrent article fetching.

This module provides an async version of WikipediaClient that can fetch
multiple articles concurrently, significantly speeding up data ingestion.
"""

import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import mwclient

logger = logging.getLogger(__name__)


async def _async_sleep_with_backoff(attempt: int, base_delay: float = 0.5, max_delay: float = 8.0) -> None:
    """Async sleep helper for exponential backoff with jitter."""
    delay = min(max_delay, base_delay * (2**attempt))
    delay *= 1 + random.uniform(-0.2, 0.2)
    if delay > 0:
        await asyncio.sleep(delay)


class AsyncWikipediaClient:
    """Async client for interacting with Wikipedia API using mwclient."""

    def __init__(self, site: str = "en.wikipedia.org", max_workers: int = 10, *, max_retries: int = 3):
        """
        Initialize async Wikipedia client.

        Args:
            site: Wikipedia site (default: en.wikipedia.org)
            max_workers: Maximum number of concurrent workers for fetching articles
            max_retries: Maximum number of retries for transient failures.
        """
        self.site = mwclient.Site(site)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max(1, int(max_retries))
        logger.info(
            "Initialized async Wikipedia client for %s (max_workers=%d, max_retries=%d)",
            site,
            max_workers,
            self.max_retries,
        )

    async def get_article(
        self,
        title: str,
        fetch_links: bool = False,
        fetch_categories: bool = False,
        max_links: int = 50,
    ) -> Optional[Dict]:
        """
        Fetch article by title asynchronously.

        Args:
            title: Article title
            fetch_links: Whether to fetch article links (can be slow for large articles)
            fetch_categories: Whether to fetch categories (can be slow)

        Returns:
            Article data dictionary or None
        """
        loop = asyncio.get_event_loop()
        for attempt in range(self.max_retries):
            try:
                # Run blocking mwclient calls in thread pool
                page = await loop.run_in_executor(self.executor, lambda: self.site.pages[title])

                if not page.exists:
                    logger.debug("Article '%s' does not exist", title)
                    return None

                # Fetch text first (most important)
                text = await loop.run_in_executor(self.executor, lambda: page.text())

                # Fetch other data conditionally (these can be slow)
                categories: List[str] = []
                links: List[str] = []
                revisions: List[Dict] = []

                if fetch_categories:
                    try:
                        categories = await loop.run_in_executor(
                            self.executor,
                            lambda: list(page.categories()),
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not fetch categories for '%s'", title)

                if fetch_links:
                    try:
                        # Limit links to avoid fetching thousands
                        all_links = await loop.run_in_executor(
                            self.executor,
                            lambda: list(page.links()),
                        )
                        links = all_links[:max_links]
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not fetch links for '%s'", title)

                try:
                    revisions = await loop.run_in_executor(
                        self.executor,
                        lambda: list(page.revisions(max_items=1)),
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Could not fetch revisions for '%s'", title)

                return {
                    "title": page.name,
                    "text": text,
                    "revisions": revisions,
                    "categories": categories,
                    "links": links,
                }
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= self.max_retries:
                    logger.error(
                        "Error fetching article '%s' after %d attempts: %s",
                        title,
                        self.max_retries,
                        exc,
                    )
                    return None

                logger.warning(
                    "Transient error fetching article '%s' (attempt %d/%d): %s; retrying...",
                    title,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                await _async_sleep_with_backoff(attempt)

    async def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for articles asynchronously.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of article metadata
        """
        loop = asyncio.get_event_loop()

        async def _search_with_retries() -> List[Dict]:
            for attempt in range(self.max_retries):
                try:
                    def _search() -> List[Dict]:
                        search_results: List[Dict] = []
                        for page in self.site.search(query, max_items=limit):
                            search_results.append(
                                {
                                    "title": page.get("title"),
                                    "snippet": page.get("snippet"),
                                }
                            )
                        return search_results

                    return await loop.run_in_executor(self.executor, _search)
                except Exception as exc:  # noqa: BLE001
                    if attempt + 1 >= self.max_retries:
                        logger.error(
                            "Error searching articles for query '%s' after %d attempts: %s",
                            query,
                            self.max_retries,
                            exc,
                        )
                        return []

                    logger.warning(
                        "Transient error during search for query '%s' (attempt %d/%d): %s; retrying...",
                        query,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    await _async_sleep_with_backoff(attempt)

            return []

        return await _search_with_retries()

    async def get_articles_batch(
        self,
        titles: List[str],
        fetch_links: bool = False,
        fetch_categories: bool = False,
        max_links: int = 50,
    ) -> List[Optional[Dict]]:
        """
        Fetch multiple articles concurrently.

        Args:
            titles: List of article titles
            fetch_links: Whether to fetch article links
            fetch_categories: Whether to fetch categories

        Returns:
            List of article dictionaries (None for failed fetches)
        """
        tasks = [
            self.get_article(
                title,
                fetch_links=fetch_links,
                fetch_categories=fetch_categories,
                max_links=max_links,
            )
            for title in titles
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Exception in batch fetch: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

