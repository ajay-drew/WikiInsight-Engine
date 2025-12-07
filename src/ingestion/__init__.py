"""
Data ingestion module for Wikipedia data.
"""

from .wikipedia_client import WikipediaClient
from .wikipedia_client_async import AsyncWikipediaClient

__all__ = ["WikipediaClient", "AsyncWikipediaClient"]

