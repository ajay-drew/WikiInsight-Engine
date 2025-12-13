"""
Setup script to download required NLTK and spaCy resources.

This script ensures all required NLP resources are downloaded:
- NLTK: punkt_tab (or punkt), stopwords, wordnet
- spaCy: en_core_web_sm model

Run this script once after installing dependencies to ensure all resources are available.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_nltk():
    """Download required NLTK data packages."""
    try:
        import nltk

        logger.info("Setting up NLTK resources...")

        # Download stopwords
        try:
            from nltk.corpus import stopwords

            _ = stopwords.words("english")
            logger.info("✓ NLTK stopwords already available")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=False)
            logger.info("✓ NLTK stopwords downloaded")

        # Download punkt tokenizer (try punkt_tab first, then punkt)
        punkt_available = False
        for punkt_name in ["tokenizers/punkt_tab", "tokenizers/punkt"]:
            try:
                _ = nltk.data.find(punkt_name)
                logger.info(f"✓ NLTK {punkt_name} already available")
                punkt_available = True
                break
            except LookupError:
                continue

        if not punkt_available:
            logger.info("Downloading NLTK punkt tokenizer...")
            try:
                nltk.download("punkt_tab", quiet=False)
                logger.info("✓ NLTK punkt_tab downloaded")
            except Exception:  # noqa: BLE001
                try:
                    nltk.download("punkt", quiet=False)
                    logger.info("✓ NLTK punkt downloaded")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to download punkt tokenizer: {exc}")

        # Download wordnet
        try:
            from nltk.corpus import wordnet  # noqa: F401

            logger.info("✓ NLTK wordnet already available")
        except LookupError:
            logger.info("Downloading NLTK wordnet...")
            nltk.download("wordnet", quiet=False)
            logger.info("✓ NLTK wordnet downloaded")

        logger.info("NLTK setup complete!")
        return True

    except ImportError:
        logger.error("NLTK is not installed. Install it with: pip install nltk")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"NLTK setup failed: {exc}")
        return False


def setup_spacy():
    """Download required spaCy model."""
    try:
        import spacy

        logger.info("Setting up spaCy resources...")

        try:
            _ = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model 'en_core_web_sm' already available")
            return True
        except OSError:
            logger.info("spaCy model 'en_core_web_sm' not found. Downloading...")
            
            # Check if pip is available first
            pip_available = False
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                pip_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("pip is not available in this environment")
            
            # Try method 1: spacy download command
            try:
                logger.info("Attempting to download via 'spacy download' command...")
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                # Verify it worked
                _ = spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model 'en_core_web_sm' downloaded successfully")
                return True
            except (subprocess.CalledProcessError, OSError) as exc:
                if not pip_available:
                    logger.error(
                        f"spacy download command failed and pip is not available.\n"
                        f"Error: {exc}\n\n"
                        f"To fix this:\n"
                        f"1. Ensure pip is installed: python -m ensurepip --upgrade\n"
                        f"2. Or install the model manually: pip install en_core_web_sm\n"
                        f"3. Or use: python -m spacy download en_core_web_sm (after fixing pip)"
                    )
                    return False
                
                logger.warning(
                    f"spacy download command failed: {exc}. "
                    "Trying alternative method with pip..."
                )
                
                # Try method 2: pip install directly
                try:
                    logger.info("Attempting to install via pip...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "en_core_web_sm"],
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                    )
                    # Verify it worked
                    _ = spacy.load("en_core_web_sm")
                    logger.info("✓ spaCy model 'en_core_web_sm' installed successfully via pip")
                    return True
                except subprocess.CalledProcessError as pip_exc:
                    logger.error(
                        f"Both download methods failed.\n"
                        f"spacy download error: {exc}\n"
                        f"pip install error: {pip_exc}\n\n"
                        f"Please try manually:\n"
                        f"  pip install en_core_web_sm\n"
                        f"Or if pip is not available:\n"
                        f"  python -m ensurepip --upgrade\n"
                        f"  pip install en_core_web_sm"
                    )
                    return False
        except Exception as exc:  # noqa: BLE001
            logger.error(f"spaCy setup failed: {exc}")
            return False

    except ImportError:
        logger.error("spaCy is not installed. Install it with: pip install spacy")
        return False


def main():
    """Run setup for all NLP resources."""
    logger.info("=" * 60)
    logger.info("NLP Resources Setup")
    logger.info("=" * 60)

    nltk_ok = setup_nltk()
    spacy_ok = setup_spacy()

    logger.info("=" * 60)
    if nltk_ok and spacy_ok:
        logger.info("✓ All NLP resources are ready!")
        return 0
    else:
        logger.warning("⚠ Some resources failed to download. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

