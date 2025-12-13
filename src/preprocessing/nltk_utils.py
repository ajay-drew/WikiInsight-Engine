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
import time
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
    # try to download them automatically, then fall back if that fails.
    try:
        # Check stopwords
        try:
            _ = stopwords.words("english")  # type: ignore[call-arg]
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)
            _ = stopwords.words("english")  # type: ignore[call-arg]

        # Check punkt tokenizer (try both old and new versions)
        punkt_available = False
        for punkt_name in ["tokenizers/punkt_tab", "tokenizers/punkt"]:
            try:
                _ = nltk.data.find(punkt_name)
                punkt_available = True
                break
            except LookupError:
                continue

        if not punkt_available:
            logger.info("Downloading NLTK punkt tokenizer...")
            # Try downloading punkt_tab first (newer versions), then fall back to punkt
            downloaded = False
            for resource_name in ["punkt_tab", "punkt"]:
                try:
                    nltk.download(resource_name, quiet=True)
                    # Verify download worked by checking if we can find it now
                    try:
                        if resource_name == "punkt_tab":
                            _ = nltk.data.find("tokenizers/punkt_tab")
                        else:
                            _ = nltk.data.find("tokenizers/punkt")
                        punkt_available = True
                        downloaded = True
                        logger.info("Successfully downloaded NLTK %s tokenizer", resource_name)
                        break
                    except LookupError:
                        # Download didn't work, try next resource
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to download %s: %s", resource_name, exc)
                    continue
            
            if not downloaded:
                logger.warning(
                    "Failed to download punkt tokenizer. "
                    "NLTK tokenization will not work, falling back to simple tokenizer."
                )

        # Check wordnet for lemmatization
        try:
            from nltk.corpus import wordnet  # noqa: F401
        except LookupError:
            logger.info("Downloading NLTK wordnet...")
            nltk.download("wordnet", quiet=True)

    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK setup failed, falling back to simple normalization: %s", exc)
        tokens = _simple_normalize(text)
        return " ".join(tokens)

    # At this point we've attempted to ensure NLTK resources are present.
    # Try to use word_tokenize, but if it fails (e.g., punkt still not available),
    # try downloading again with verbose output, then fall back gracefully.
    raw_tokens = None
    try:
        raw_tokens = word_tokenize(text.lower())  # type: ignore[operator]
    except LookupError as exc:
        # Resource still missing even after download attempt - try one more download with verbose output
        error_msg = str(exc)
        if "punkt" in error_msg.lower():
            logger.info("punkt tokenizer still missing after download attempt. Retrying download with verbose output...")
            download_succeeded = False
            for resource_name in ["punkt_tab", "punkt"]:
                try:
                    logger.info("Attempting to download %s...", resource_name)
                    nltk.download(resource_name, quiet=False)  # Show output this time
                    # Small delay to ensure download is complete
                    time.sleep(0.5)
                    # Try word_tokenize again
                    raw_tokens = word_tokenize(text.lower())  # type: ignore[operator]
                    logger.info("Successfully downloaded and loaded %s after retry", resource_name)
                    download_succeeded = True
                    break
                except LookupError:
                    # Still not found, try next resource
                    continue
                except Exception as download_exc:  # noqa: BLE001
                    logger.debug("Download of %s failed: %s", resource_name, download_exc)
                    continue
            
            if not download_succeeded:
                logger.warning(
                    "NLTK word_tokenize failed after download retry attempts. "
                    "Using simple tokenizer instead. Error: %s",
                    exc,
                )
                tokens = _simple_normalize(text)
                return " ".join(tokens)
        else:
            # Some other LookupError
            logger.warning("NLTK word_tokenize failed, using simple tokenizer instead: %s", exc)
            tokens = _simple_normalize(text)
            return " ".join(tokens)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK word_tokenize failed, using simple tokenizer instead: %s", exc)
        tokens = _simple_normalize(text)
        return " ".join(tokens)
    
    # If we get here, raw_tokens should be set (either from first try or retry)
    if raw_tokens is None:
        # This shouldn't happen, but just in case
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


