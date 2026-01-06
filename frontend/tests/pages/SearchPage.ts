import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Search page
 */
export class SearchPage {
  readonly page: Page;
  
  // Form elements
  readonly searchInput: Locator;
  readonly searchButton: Locator;
  readonly errorMessage: Locator;
  readonly loadingIndicator: Locator;
  readonly resultsContainer: Locator;
  readonly resultsList: Locator;
  readonly noResultsMessage: Locator;

  constructor(page: Page) {
    this.page = page;
    this.searchInput = page.locator('input[type="text"]').first();
    // Target the submit button in the form, not the navigation button
    this.searchButton = page.locator('form').getByRole('button', { name: /Search/i });
    this.errorMessage = page.locator('p.text-red-400, span.text-red-400').first();
    this.loadingIndicator = page.locator('[class*="animate-spin"]').first();
    this.resultsContainer = page.locator('section:has-text("Search Results")');
    this.resultsList = page.locator('div.space-y-3');
    this.noResultsMessage = page.locator('text=/No results found/i');
  }

  /**
   * Navigate to search page
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
    // Wait for search input to be visible
    await this.searchInput.waitFor({ state: 'visible', timeout: 5000 });
  }

  /**
   * Enter search query
   */
  async enterQuery(query: string): Promise<void> {
    await this.searchInput.fill(query);
  }

  /**
   * Submit search form
   */
  async search(query: string): Promise<void> {
    await this.enterQuery(query);
    await this.searchButton.click();
  }

  /**
   * Wait for search results to appear
   */
  async waitForResults(timeout: number = 10000): Promise<void> {
    await this.page.waitForSelector('section:has-text("Search Results")', { timeout });
  }

  /**
   * Get search results
   */
  async getResults(): Promise<Array<{ title: string; score: number; rank: number }>> {
    const results: Array<{ title: string; score: number; rank: number }> = [];
    const resultCards = await this.page.locator('[class*="SearchResultCard"], div.border').all();
    
    for (const card of resultCards) {
      const title = await card.locator('h3, h2, [class*="font-semibold"]').first().textContent();
      if (title) {
        results.push({
          title: title.trim(),
          score: 0, // Would need to extract from DOM
          rank: results.length + 1,
        });
      }
    }
    
    return results;
  }

  /**
   * Get error message if present
   */
  async getError(): Promise<string | null> {
    try {
      const error = await this.errorMessage.textContent();
      return error;
    } catch {
      return null;
    }
  }

  /**
   * Check if loading indicator is visible
   */
  async isLoading(): Promise<boolean> {
    try {
      return await this.loadingIndicator.isVisible();
    } catch {
      return false;
    }
  }

  /**
   * Check if results are displayed
   */
  async hasResults(): Promise<boolean> {
    try {
      const hasResults = await this.resultsContainer.isVisible();
      const noResults = await this.noResultsMessage.isVisible();
      return hasResults && !noResults;
    } catch {
      return false;
    }
  }

  /**
   * Get result count from UI
   */
  async getResultCount(): Promise<number> {
    try {
      const countText = await this.page.locator('text=/\\d+ (result|results)/i').first().textContent();
      if (countText) {
        const match = countText.match(/(\d+)/);
        return match ? parseInt(match[1], 10) : 0;
      }
      return 0;
    } catch {
      return 0;
    }
  }

  /**
   * Click on a search result
   */
  async clickResult(index: number): Promise<void> {
    const results = await this.page.locator('div.space-y-3 > div').all();
    if (results[index]) {
      await results[index].click();
    }
  }
}

