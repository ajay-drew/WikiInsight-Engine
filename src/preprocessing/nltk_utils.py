"""
NLTK-based text normalization utilities.

This module provides a lightweight, defensive wrapper around NLTK so that:
- We can use proper tokenization, stopword removal, lemmatization, and optional
  stemming when NLTK and its data packages are available.
- The rest of the pipeline degrades gracefully (falls back to simple regex +
  lowercasing) when NLTK or its corpora are missing, instead of crashing.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

logger = logging.getLogger(__name__)

try:  # NLTK is an optional dependency at runtime
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except Exception:  # noqa: BLE001
    nltk = None  # type: ignore[assignment]
    stopwords = None  # type: ignore[assignment]
    WordNetLemmatizer = None  # type: ignore[assignment]
    PorterStemmer = None  # type: ignore[assignment]
    word_tokenize = None  # type: ignore[assignment]
    _NLTK_AVAILABLE = False


_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']+")


def _simple_normalize(text: str) -> List[str]:
    """
    Fallback normalization that does not rely on NLTK data.

    - Lowercases
    - Strips non-alphanumeric characters
    - Splits on whitespace
    """
    text = text or ""
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def normalize_text(
    text: str,
    *,
    use_stem: bool = False,
) -> str:
    """
    Normalize text using NLTK when possible, with a safe fallback.

    Steps:
    - Lowercasing
    - Regex cleanup of obvious noise
    - Tokenization (NLTK word_tokenize when available)
    - Stopword removal (NLTK stopword list when available)
    - Lemmatization (WordNet)
    - Optional stemming (Porter)

    Returns:
        A single string with normalized tokens joined by spaces.
    """
    text = text or ""

    if not _NLTK_AVAILABLE:
        # NLTK (or its modules) not importable at all.
        logger.warning(
            "NLTK is not installed; falling back to simple regex-based normalization.",
        )
        tokens = _simple_normalize(text)
        return " ".join(tokens)

    # Ensure required NLTK data packages are available. If they are not,
    # we log a warning and fall back to the simple normalizer.
    try:
        # These will raise LookupError if resources are missing.
        _ = stopwords.words("english")  # type: ignore[call-arg]
        _ = nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.warning(
            "Required NLTK data packages (e.g. 'punkt', 'stopwords') are missing. "
            "Run: python -m nltk.downloader punkt stopwords wordnet",
        )
        tokens = _simple_normalize(text)
        return " ".join(tokens)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK setup failed, falling back to simple normalization: %s", exc)
        tokens = _simple_normalize(text)
        return " ".join(tokens)

    # At this point we know the core NLTK resources are present.
    try:
        raw_tokens = word_tokenize(text.lower())  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK word_tokenize failed, using simple tokenizer instead: %s", exc)
        tokens = _simple_normalize(text)
        return " ".join(tokens)

    try:
        stop_words = set(stopwords.words("english"))  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK stopwords failed, continuing without stopword removal: %s", exc)
        stop_words = set()

    lemmatizer = WordNetLemmatizer() if WordNetLemmatizer is not None else None  # type: ignore[call-arg]
    stemmer = PorterStemmer() if (use_stem and PorterStemmer is not None) else None  # type: ignore[call-arg]

    normalized_tokens: List[str] = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok or not _WORD_RE.match(tok):
            continue
        if tok in stop_words:
            continue

        lemma = tok
        if lemmatizer is not None:
            try:
                lemma = lemmatizer.lemmatize(tok)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                lemma = tok

        if stemmer is not None:
            try:
                lemma = stemmer.stem(lemma)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                # If stemming fails, keep the lemma.
                pass

        if lemma:
            normalized_tokens.append(lemma)

    return " ".join(normalized_tokens)


def normalize_corpus(
    texts: Iterable[str],
    *,
    use_stem: bool = False,
) -> List[str]:
    """
    Convenience helper to normalize a corpus of documents.

    Args:
        texts: Iterable of raw text documents.
        use_stem: Whether to apply stemming in addition to lemmatization.

    Returns:
        List of normalized documents (space-joined tokens).
    """
    return [normalize_text(t, use_stem=use_stem) for t in texts]


