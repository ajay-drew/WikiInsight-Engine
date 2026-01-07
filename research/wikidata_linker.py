"""
Wikidata entity linking for Wikipedia articles.

Links Wikipedia article titles to Wikidata entities (QIDs) using the
Wikidata Search API and caches results to avoid repeated API calls.

Note: Wikidata API requires a User-Agent header and has rate limits.
Requests without proper User-Agent may be blocked with 403 Forbidden.
"""

import logging
import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
CACHE_PATH = "data/entities/wikidata_mappings.parquet"

# Wikidata API rate limit: ~200 requests per minute per IP
# Add delay between requests to avoid hitting rate limits
WIKIDATA_REQUEST_DELAY = 0.3  # seconds between requests


class WikidataLinker:
    """Links Wikipedia articles to Wikidata entities."""

    def __init__(
        self,
        cache_path: str = CACHE_PATH,
        enabled: bool = True,
        request_delay: float = WIKIDATA_REQUEST_DELAY,
    ):
        """
        Initialize Wikidata linker.

        Args:
            cache_path: Path to cache file for entity mappings
            enabled: Whether Wikidata linking is enabled
            request_delay: Delay in seconds between API requests to respect rate limits
        """
        self.cache_path = cache_path
        self.enabled = enabled
        self.request_delay = request_delay
        self.cache: Dict[str, Optional[str]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached entity mappings from disk."""
        if not self.enabled:
            return

        if os.path.exists(self.cache_path):
            try:
                df = pd.read_parquet(self.cache_path)
                if "title" in df.columns and "qid" in df.columns:
                    for _, row in df.iterrows():
                        title = str(row["title"]).lower()
                        qid = row["qid"]
                        self.cache[title] = qid if pd.notna(qid) else None
                    logger.info("Loaded %d Wikidata mappings from cache", len(self.cache))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load Wikidata cache: %s", exc)
                self.cache = {}

    def _save_cache(self) -> None:
        """Save entity mappings to disk."""
        if not self.enabled or not self.cache:
            return

        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            records = [
                {"title": title, "qid": qid} for title, qid in self.cache.items()
            ]
            df = pd.DataFrame(records)
            df.to_parquet(self.cache_path, index=False)
            logger.info("Saved %d Wikidata mappings to cache", len(self.cache))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save Wikidata cache: %s", exc)

    def link_entity(self, article_title: str) -> Optional[str]:
        """
        Link a Wikipedia article to its Wikidata QID.

        Args:
            article_title: Wikipedia article title

        Returns:
            Wikidata QID (e.g., "Q1234") or None if not found
        """
        if not self.enabled:
            return None

        title_lower = article_title.lower()

        # Check cache first
        if title_lower in self.cache:
            return self.cache[title_lower]

        # Add delay to respect rate limits (only if not cached)
        time.sleep(self.request_delay)

        # Query Wikidata API
        qid = self._search_wikidata(article_title)

        # Cache result (even if None)
        self.cache[title_lower] = qid

        # Save cache periodically (every 100 new entries)
        if len(self.cache) % 100 == 0:
            self._save_cache()

        return qid

    def _search_wikidata(self, article_title: str) -> Optional[str]:
        """
        Search Wikidata for an entity matching the article title.

        Args:
            article_title: Wikipedia article title

        Returns:
            Wikidata QID or None
        """
        try:
            params = {
                "action": "wbsearchentities",
                "search": article_title,
                "language": "en",
                "format": "json",
                "limit": 1,
            }

            # Wikidata API requires User-Agent header to identify the application
            # Without it, requests may be blocked with 403 Forbidden
            headers = {
                "User-Agent": "WikiInsight-Engine/0.1.0 (https://github.com/wikiinsight/wikiinsight-engine; contact@example.com) Python/requests"
            }

            response = requests.get(WIKIDATA_API_URL, params=params, headers=headers, timeout=5)
            response.raise_for_status()

            data = response.json()
            if "search" in data and len(data["search"]) > 0:
                entity = data["search"][0]
                qid = entity.get("id")
                logger.debug("Found Wikidata QID %s for article '%s'", qid, article_title)
                return qid

            logger.debug("No Wikidata entity found for article '%s'", article_title)
            return None
        except requests.exceptions.HTTPError as exc:
            # Handle 403 Forbidden specifically (rate limiting or missing User-Agent)
            if exc.response.status_code == 403:
                logger.warning(
                    "Wikidata API returned 403 Forbidden for '%s'. "
                    "This may be due to rate limiting or missing User-Agent header. "
                    "Consider reducing request frequency or adding delays.",
                    article_title,
                )
            else:
                logger.warning("Wikidata API request failed for '%s': %s", article_title, exc)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Wikidata API request failed for '%s': %s", article_title, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error searching Wikidata for '%s': %s", article_title, exc)
            return None

    def link_batch(self, article_titles: list[str], delay: Optional[float] = None) -> Dict[str, Optional[str]]:
        """
        Link multiple articles to Wikidata entities.

        Args:
            article_titles: List of article titles
            delay: Delay between API requests (seconds) to respect rate limits.
                   If None, uses instance request_delay.

        Returns:
            Dictionary mapping article titles to QIDs
        """
        if delay is None:
            delay = self.request_delay

        results = {}
        for title in article_titles:
            # link_entity already has delay built-in, but we can add extra delay for batch processing
            qid = self.link_entity(title)
            results[title] = qid
            # Additional delay only if we made an API call (not cached)
            title_lower = title.lower()
            if title_lower not in self.cache:
                time.sleep(delay)

        # Save cache after batch
        self._save_cache()

        return results

    def get_wikidata_url(self, qid: str) -> str:
        """
        Generate Wikidata URL for a QID.

        Args:
            qid: Wikidata entity ID (e.g., "Q1234")

        Returns:
            Wikidata URL
        """
        return f"https://www.wikidata.org/wiki/{qid}"

    def finalize(self) -> None:
        """Save cache to disk (call when done linking)."""
        self._save_cache()

