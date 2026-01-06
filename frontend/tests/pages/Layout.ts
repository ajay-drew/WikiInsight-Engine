import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Layout/Navigation
 */
export class Layout {
  readonly page: Page;
  
  // Navigation buttons
  readonly searchButton: Locator;
  readonly topicLookupButton: Locator;
  readonly clustersButton: Locator;
  readonly monitoringButton: Locator;
  readonly ingestionButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.searchButton = page.getByRole('button', { name: 'Search' });
    this.topicLookupButton = page.getByRole('button', { name: 'Topic Lookup' });
    this.clustersButton = page.getByRole('button', { name: 'Clusters' });
    this.monitoringButton = page.getByRole('button', { name: 'Monitoring' });
    this.ingestionButton = page.getByRole('button', { name: 'Ingestion' });
  }

  /**
   * Navigate to a specific page
   */
  async navigateTo(pageName: 'search' | 'lookup' | 'clusters' | 'monitoring' | 'ingestion'): Promise<void> {
    // Wait for page to be fully loaded
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForLoadState('domcontentloaded');
    
    // Wait for navigation buttons container to be visible
    await this.page.waitForSelector('nav.flex.gap-2', { timeout: 10000 });
    
    switch (pageName) {
      case 'search':
        await this.searchButton.waitFor({ state: 'visible', timeout: 10000 });
        await this.searchButton.click();
        break;
      case 'lookup':
        await this.topicLookupButton.waitFor({ state: 'visible', timeout: 10000 });
        await this.topicLookupButton.click();
        break;
      case 'clusters':
        await this.clustersButton.waitFor({ state: 'visible', timeout: 10000 });
        await this.clustersButton.click();
        break;
      case 'monitoring':
        await this.monitoringButton.waitFor({ state: 'visible', timeout: 10000 });
        await this.monitoringButton.click();
        break;
      case 'ingestion':
        await this.ingestionButton.waitFor({ state: 'visible', timeout: 10000 });
        await this.ingestionButton.click();
        break;
    }
    // Wait for navigation to complete (button becomes active)
    await this.page.waitForTimeout(1000);
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Get the currently active page
   */
  async getActivePage(): Promise<string> {
    // Check which button has the active class (bg-sky-500)
    const activeButton = await this.page.locator('button.bg-sky-500').first();
    if (await activeButton.count() > 0) {
      return await activeButton.textContent() || '';
    }
    return '';
  }

  /**
   * Verify page is active
   */
  async verifyPageActive(pageName: string): Promise<boolean> {
    const activeText = await this.getActivePage();
    return activeText.toLowerCase().includes(pageName.toLowerCase());
  }
}

