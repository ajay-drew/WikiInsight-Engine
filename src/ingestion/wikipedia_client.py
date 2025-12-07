"""
Wikipedia API client for fetching articles and metadata.
"""

import mwclient
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class WikipediaClient:
    """Client for interacting with Wikipedia API using mwclient."""
    
    def __init__(self, site: str = "en.wikipedia.org"):
        """
        Initialize Wikipedia client.
        
        Args:
            site: Wikipedia site (default: en.wikipedia.org)
        """
        self.site = mwclient.Site(site)
        logger.info(f"Initialized Wikipedia client for {site}")
    
    def get_article(self, title: str, fetch_links: bool = True, fetch_categories: bool = True) -> Optional[Dict]:
        """
        Fetch article by title.
        
        Args:
            title: Article title
            fetch_links: Whether to fetch article links (can be slow for large articles)
            fetch_categories: Whether to fetch categories (can be slow)
            
        Returns:
            Article data dictionary or None
        """
        try:
            page = self.site.pages[title]
            if not page.exists:
                logger.warning(f"Article '{title}' does not exist")
                return None
            
            # Fetch text first (most important)
            text = page.text()
            
            # Fetch other data conditionally (these can be slow)
            categories = []
            links = []
            revisions = []
            
            if fetch_categories:
                try:
                    categories = list(page.categories())
                except Exception:
                    logger.debug(f"Could not fetch categories for '{title}'")
            
            if fetch_links:
                try:
                    # Limit links to avoid fetching thousands
                    links = list(page.links())[:100]  # Limit to first 100 links
                except Exception:
                    logger.debug(f"Could not fetch links for '{title}'")
            
            try:
                revisions = list(page.revisions(max_items=1))
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
    
    def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for articles.
        
        Args:
            query: Search query
            limit: Maximum number of results (uses max_items internally)
            
        Returns:
            List of article metadata
        """
        results = []
        # Use max_items instead of deprecated limit parameter
        for page in self.site.search(query, max_items=limit):
            results.append({
                "title": page.get("title"),
                "snippet": page.get("snippet"),
            })
        return results

