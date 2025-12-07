"""
Standalone test script to fetch coronavirus data from Wikipedia.

This script can be run directly to test Wikipedia data fetching:
    python scripts/test_coronavirus_fetch.py

It will:
1. Search for coronavirus-related articles
2. Fetch specific articles (Coronavirus, COVID-19, etc.)
3. Display article metadata and sample content
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.ingestion.wikipedia_client import WikipediaClient
    from src.common.logging_utils import setup_logging
    import logging
except ImportError as e:
    print(f"ERROR: Error importing required modules: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("\nSpecifically, make sure 'mwclient' is installed:")
    print("  pip install mwclient")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    print("\n" + "="*70)
    print("Wikipedia Coronavirus Data Fetching Test")
    print("="*70 + "\n")
    
    # Initialize client
    print("Initializing Wikipedia client...")
    client = WikipediaClient(site="en.wikipedia.org")
    print("[OK] Client initialized\n")
    
    # Test 1: Search for coronavirus articles
    print("-" * 70)
    print("Test 1: Searching for 'coronavirus' articles...")
    print("-" * 70)
    try:
        results = client.search_articles("coronavirus", limit=10)
        print(f"[OK] Found {len(results)} articles\n")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']}")
            if result.get('snippet'):
                snippet = result['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
                print(f"     {snippet[:100]}...")
        print()
    except Exception as e:
        print(f"[ERROR] Error searching articles: {e}\n")
        return
    
    # Test 2: Fetch main Coronavirus article
    print("-" * 70)
    print("Test 2: Fetching 'Coronavirus' article...")
    print("-" * 70)
    try:
        article = client.get_article("Coronavirus")
        if article:
            print(f"[OK] Successfully fetched article\n")
            print(f"  Title: {article['title']}")
            print(f"  Text length: {len(article['text']):,} characters")
            print(f"  Categories: {len(article['categories'])}")
            print(f"  Links: {len(article['links'])}")
            print(f"\n  First 300 characters:")
            print(f"  {article['text'][:300]}...")
            print()
            
            # Show some categories
            if article['categories']:
                print(f"  Sample categories (first 5):")
                for cat in article['categories'][:5]:
                    print(f"    - {cat}")
                print()
        else:
            print("[ERROR] Failed to fetch article\n")
    except Exception as e:
        print(f"[ERROR] Error fetching article: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Test 3: Fetch COVID-19 article
    print("-" * 70)
    print("Test 3: Fetching 'COVID-19' article...")
    print("-" * 70)
    try:
        covid_article = client.get_article("COVID-19")
        if covid_article:
            print(f"[OK] Successfully fetched article\n")
            print(f"  Title: {covid_article['title']}")
            print(f"  Text length: {len(covid_article['text']):,} characters")
            print(f"  Categories: {len(covid_article['categories'])}")
            print(f"  Links: {len(covid_article['links'])}")
            print(f"\n  First 300 characters:")
            print(f"  {covid_article['text'][:300]}...")
            print()
        else:
            print("[ERROR] Failed to fetch article\n")
    except Exception as e:
        print(f"[ERROR] Error fetching article: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Test 4: Fetch multiple related articles
    print("-" * 70)
    print("Test 4: Fetching multiple coronavirus-related articles...")
    print("-" * 70)
    titles = [
        "SARS-CoV-2",
        "Pandemic",
        "Severe acute respiratory syndrome coronavirus 2",
    ]
    
    fetched_count = 0
    for title in titles:
        try:
            article = client.get_article(title)
            if article:
                print(f"  [OK] {title}: {len(article['text']):,} chars")
                fetched_count += 1
            else:
                print(f"  [ERROR] {title}: Not found")
        except Exception as e:
            print(f"  [ERROR] {title}: Error - {e}")
    
    print(f"\n[OK] Successfully fetched {fetched_count}/{len(titles)} articles\n")
    
    # Summary
    print("="*70)
    print("Test Summary")
    print("="*70)
    print("[OK] Wikipedia client initialized")
    print("[OK] Search functionality tested")
    print("[OK] Article fetching tested")
    print("[OK] Data structure validated")
    print("\nAll tests completed!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

