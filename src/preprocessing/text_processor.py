"""
Text preprocessing utilities.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text preprocessing and cleaning."""

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove Wikipedia markup
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # Remove [[links]]
        text = re.sub(r'{{[^}]+}}', '', text)  # Remove templates
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

