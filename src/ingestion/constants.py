"""
Shared constants for Wikipedia ingestion.
"""

import os
from typing import List

RAW_DATA_PATH = os.path.join("data", "raw", "articles.json")

# Simple, hard-coded seeds for now; can be extended or made configurable later.
SEED_QUERIES: List[str] = [
    "Machine learning",
    "Artificial intelligence",
    "Data science",
    "Physics",
    "Biology",
    "History",
]

