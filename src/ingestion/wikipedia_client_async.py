"""
Async Wikipedia API client for concurrent article fetching.

This module provides an async version of WikipediaClient that can fetch
multiple articles concurrently, significantly speeding up data ingestion.
Includes rate limiting to ensure we stay within Wikipedia's 200 req/sec limit.
"""

import asyncio
import logging
import random
import time
from collections import deque
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


class AsyncRateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    
    Ensures we don't exceed a specified requests per second limit
    while allowing bursts up to the limit.
    """
    
    def __init__(self, rate: float = 200.0, burst: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum requests per second
            burst: Maximum burst size (defaults to rate if not specified)
        """
        self.rate = float(rate)
        self.burst = int(burst) if burst is not None else int(rate)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        
    async def acquire(self, n: int = 1) -> None:
        """
        Acquire n tokens, waiting if necessary.
        
        Args:
            n: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Wait if we don't have enough tokens
            if self.tokens < n:
                wait_time = (n - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
                self.last_update = time.monotonic()
            else:
                self.tokens -= n


class AsyncWikipediaClient:
    """Async client for interacting with Wikipedia API using mwclient."""

    def __init__(
        self,
        site: str = "en.wikipedia.org",
        max_workers: int = 10,
        *,
        max_retries: int = 3,
        rate_limit: float = 200.0,
    ):
        """
        Initialize async Wikipedia client.

        Args:
            site: Wikipedia site (default: en.wikipedia.org)
            max_workers: Maximum number of concurrent workers for fetching articles
            max_retries: Maximum number of retries for transient failures.
            rate_limit: Maximum requests per second (default: 200.0)
        """
        self.site = mwclient.Site(site)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_retries = max(1, int(max_retries))
        self.rate_limiter = AsyncRateLimiter(rate=rate_limit, burst=int(rate_limit))
        self.semaphore = asyncio.Semaphore(max_workers)
        logger.info(
            "Initialized async Wikipedia client for %s (max_workers=%d, max_retries=%d, rate_limit=%.1f req/sec)",
            site,
            max_workers,
            self.max_retries,
            rate_limit,
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
        # Acquire semaphore to limit concurrent requests
        async with self.semaphore:
            # Acquire rate limiter token (each API call counts as 1 request)
            await self.rate_limiter.acquire(1)
            
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
        # Acquire semaphore and rate limiter token
        async with self.semaphore:
            await self.rate_limiter.acquire(1)
            
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

