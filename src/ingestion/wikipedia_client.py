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
    
    def get_article(self, title: str) -> Optional[Dict]:
        """
        Fetch article by title.
        
        Args:
            title: Article title
            
        Returns:
            Article data dictionary or None
        """
        try:
            page = self.site.pages[title]
            if not page.exists:
                logger.warning(f"Article '{title}' does not exist")
                return None
            
            return {
                "title": page.name,
                "text": page.text(),
                "revisions": list(page.revisions(max_items=1)),
                "categories": list(page.categories()),
                "links": list(page.links()),
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

