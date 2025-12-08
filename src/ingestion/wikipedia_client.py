"""
Wikipedia API client for fetching articles and metadata.
"""

import logging
import random
import time
from typing import Dict, List, Optional

import mwclient

logger = logging.getLogger(__name__)


def _sleep_with_backoff(attempt: int, base_delay: float = 0.5, max_delay: float = 8.0) -> None:
    """Sleep for an exponentially increasing backoff interval with jitter."""
    delay = min(max_delay, base_delay * (2**attempt))
    # Add a little jitter so callers don't all retry in lock-step.
    delay *= 1 + random.uniform(-0.2, 0.2)
    if delay > 0:
        time.sleep(delay)


class WikipediaClient:
    """Client for interacting with Wikipedia API using mwclient."""

    def __init__(self, site: str = "en.wikipedia.org", *, max_retries: int = 3) -> None:
        """
        Initialize Wikipedia client.

        Args:
            site: Wikipedia site (default: en.wikipedia.org)
            max_retries: Maximum number of retries for transient failures.
        """
        self.site = mwclient.Site(site)
        self.max_retries = max(1, int(max_retries))
        logger.info("Initialized Wikipedia client for %s (max_retries=%d)", site, self.max_retries)

    def get_article(
        self,
        title: str,
        fetch_links: bool = True,
        fetch_categories: bool = True,
        max_links: int = 50,
    ) -> Optional[Dict]:
        """
        Fetch article by title.

        Args:
            title: Article title
            fetch_links: Whether to fetch article links (can be slow for large articles)
            fetch_categories: Whether to fetch categories (can be slow)
            max_links: Maximum number of links to retrieve per article (to avoid timeouts)

        Returns:
            Article data dictionary or None
        """
        for attempt in range(self.max_retries):
            try:
                page = self.site.pages[title]
                if not page.exists:
                    logger.warning("Article '%s' does not exist", title)
                    return None

                # Fetch text first (most important)
                text = page.text()

                # Fetch other data conditionally (these can be slow)
                categories: List[str] = []
                links: List[str] = []
                revisions: List[Dict] = []

                if fetch_categories:
                    try:
                        categories = list(page.categories())
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not fetch categories for '%s'", title)

                if fetch_links:
                    try:
                        # Limit links to avoid fetching thousands
                        links = list(page.links())[:max_links]
                    except Exception:  # noqa: BLE001
                        logger.debug("Could not fetch links for '%s'", title)

                try:
                    revisions = list(page.revisions(max_items=1))
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
                    logger.error("Error fetching article '%s' after %d attempts: %s", title, self.max_retries, exc)
                    return None

                logger.warning(
                    "Transient error fetching article '%s' (attempt %d/%d): %s; retrying...",
                    title,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                _sleep_with_backoff(attempt)

        return None

    def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for articles.

        Args:
            query: Search query
            limit: Maximum number of results (uses max_items internally)

        Returns:
            List of article metadata
        """
        for attempt in range(self.max_retries):
            try:
                results: List[Dict] = []
                # Use max_items instead of deprecated limit parameter
                for page in self.site.search(query, max_items=limit):
                    results.append(
                        {
                            "title": page.get("title"),
                            "snippet": page.get("snippet"),
                        }
                    )
                return results
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
                _sleep_with_backoff(attempt)

        return []

