"""
Pytest configuration and shared fixtures for WikiInsight Engine tests.

This module provides:
- Centralized mwclient availability checking
- Shared fixtures for Wikipedia clients
- Common test utilities
"""

import pytest

# Check for mwclient availability - skip all mwclient-dependent tests if not available
try:
    import mwclient
    MWCLIENT_AVAILABLE = True
except ImportError:
    MWCLIENT_AVAILABLE = False
    pytest.skip("mwclient is not installed. Install it with: pip install mwclient", allow_module_level=True)


@pytest.fixture(scope="session")
def mwclient_available():
    """Fixture to check if mwclient is available."""
    return MWCLIENT_AVAILABLE




@pytest.fixture
def async_wikipedia_client():
    """
    Fixture providing an AsyncWikipediaClient instance for tests.
    
    This fixture is available to all tests that need an async Wikipedia client.
    Tests will be skipped if mwclient is not installed.
    """
    if not MWCLIENT_AVAILABLE:
        pytest.skip("mwclient is not installed")
    
    from src.ingestion.wikipedia_client_async import AsyncWikipediaClient
    return AsyncWikipediaClient(site="en.wikipedia.org", max_workers=5)

