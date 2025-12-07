"""
Data ingestion module for Wikipedia data.
"""

from .wikipedia_client import WikipediaClient
from .eventstreams import EventStreamsClient

__all__ = ["WikipediaClient", "EventStreamsClient"]

