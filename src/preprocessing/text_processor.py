"""
Text preprocessing utilities.
"""

import re
import logging
from typing import List, Optional, Dict

try:
    import spacy
except ImportError:  # spaCy is optional; entity extraction will be disabled if missing
    spacy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text preprocessing and cleaning."""

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize text processor.

        Args:
            model: spaCy model name
        """
        if spacy is None:
            logger.warning(
                "spaCy is not installed. Entity extraction will be disabled. "
                "Install 'spacy' and the model to enable it."
            )
            self.nlp = None
            return

        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(
                f"spaCy model '{model}' not found. "
                f"Run: python -m spacy download {model}"
            )
            self.nlp = None
    
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
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of entities with labels
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

