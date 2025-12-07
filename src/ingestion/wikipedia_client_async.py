"""
Async Wikipedia API client for concurrent article fetching.

This module provides an async version of WikipediaClient that can fetch
multiple articles concurrently, significantly speeding up data ingestion.
"""

import asyncio
import logging
from typing import Optional, Dict, List

import mwclient
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncWikipediaClient:
    """Async client for interacting with Wikipedia API using mwclient."""

    def __init__(self, site: str = "en.wikipedia.org", max_workers: int = 10):
        """
        Initialize async Wikipedia client.

        Args:
            site: Wikipedia site (default: en.wikipedia.org)
            max_workers: Maximum number of concurrent workers for fetching articles
        """
        self.site = mwclient.Site(site)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized async Wikipedia client for {site} (max_workers={max_workers})")

    async def get_article(
        self, title: str, fetch_links: bool = False, fetch_categories: bool = False
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
        try:
            # Run blocking mwclient calls in thread pool
            page = await loop.run_in_executor(self.executor, lambda: self.site.pages[title])

            if not page.exists:
                logger.debug(f"Article '{title}' does not exist")
                return None

            # Fetch text first (most important)
            text = await loop.run_in_executor(self.executor, lambda: page.text())

            # Fetch other data conditionally (these can be slow)
            categories = []
            links = []
            revisions = []

            if fetch_categories:
                try:
                    categories = await loop.run_in_executor(
                        self.executor, lambda: list(page.categories())
                    )
                except Exception:
                    logger.debug(f"Could not fetch categories for '{title}'")

            if fetch_links:
                try:
                    # Limit links to avoid fetching thousands
                    all_links = await loop.run_in_executor(
                        self.executor, lambda: list(page.links())
                    )
                    links = all_links[:100]  # Limit to first 100 links
                except Exception:
                    logger.debug(f"Could not fetch links for '{title}'")

            try:
                revisions = await loop.run_in_executor(
                    self.executor, lambda: list(page.revisions(max_items=1))
                )
            except Exception:
                logger.debug(f"Could not fetch revisions for '{title}'")

            return {
                "title": page.name,
                "text": text,
                "revisions": revisions,
                "categories": categories,
                "links": links,
            }
        except Exception as e:
            logger.error(f"Error fetching article '{title}': {e}")
            return None

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
        results = []

        def _search():
            search_results = []
            for page in self.site.search(query, max_items=limit):
                search_results.append({
                    "title": page.get("title"),
                    "snippet": page.get("snippet"),
                })
            return search_results

        results = await loop.run_in_executor(self.executor, _search)
        return results

    async def get_articles_batch(
        self, titles: List[str], fetch_links: bool = False, fetch_categories: bool = False
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
            self.get_article(title, fetch_links=fetch_links, fetch_categories=fetch_categories)
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

