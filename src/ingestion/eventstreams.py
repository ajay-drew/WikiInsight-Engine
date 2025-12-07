"""
Wikimedia EventStreams client for real-time data.
"""

import requests
from typing import Iterator, Dict
import logging

logger = logging.getLogger(__name__)


class EventStreamsClient:
    """Client for Wikimedia EventStreams API."""
    
    BASE_URL = "https://stream.wikimedia.org/v2/stream"
    
    def __init__(self):
        """Initialize EventStreams client."""
        logger.info("Initialized EventStreams client")
    
    def stream_events(self, stream: str = "recentchange") -> Iterator[Dict]:
        """
        Stream events from Wikimedia EventStreams.
        
        Args:
            stream: Stream name (e.g., 'recentchange', 'page-links-change')
            
        Yields:
            Event dictionaries
        """
        url = f"{self.BASE_URL}/{stream}"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        event = json.loads(line)
                        yield event
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error streaming events: {e}")
            raise

