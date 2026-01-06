import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Topic Lookup page
 */
export class TopicLookupPage {
  readonly page: Page;
  
  // Form elements
  readonly titleInput: Locator;
  readonly analyzeButton: Locator;
  readonly errorMessage: Locator;
  readonly loadingIndicator: Locator;
  readonly resultContainer: Locator;
  readonly clusterIdDisplay: Locator;
  readonly keywordsDisplay: Locator;
  readonly similarArticlesList: Locator;

  constructor(page: Page) {
    this.page = page;
    this.titleInput = page.locator('input[type="text"]').first();
    this.analyzeButton = page.getByRole('button', { name: /Analyze|Looking up/i });
    this.errorMessage = page.locator('span.text-red-400, p.text-red-400').first();
    this.loadingIndicator = page.locator('[class*="animate-spin"]').first();
    this.resultContainer = page.locator('section:has-text("Cluster ID")').or(page.locator('div.border').filter({ hasText: /Cluster/i }));
    this.clusterIdDisplay = page.locator('text=/Cluster ID/i');
    this.keywordsDisplay = page.locator('text=/Keywords|Cluster Keywords/i');
    this.similarArticlesList = page.locator('ul, div:has-text("Similar Articles")');
  }

  /**
   * Navigate to topic lookup page
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
    // Navigate to topic lookup page via navigation
    await this.page.getByRole('button', { name: 'Topic Lookup' }).click();
    await this.titleInput.waitFor({ state: 'visible', timeout: 5000 });
  }

  /**
   * Enter article title
   */
  async enterTitle(title: string): Promise<void> {
    await this.titleInput.fill(title);
  }

  /**
   * Submit lookup form
   */
  async lookupTopic(title: string): Promise<void> {
    await this.enterTitle(title);
    await this.analyzeButton.click();
  }

  /**
   * Wait for result to appear
   */
  async waitForResult(timeout: number = 10000): Promise<void> {
    await this.page.waitForSelector('text=/Cluster ID|Keywords|Similar Articles/i', { timeout });
  }

  /**
   * Get lookup result
   */
  async getResult(): Promise<{
    articleTitle: string;
    clusterId: number | null;
    keywords: string[];
    similarArticles: string[];
  } | null> {
    try {
      const articleTitle = await this.page.locator('h2, h3').first().textContent();
      const clusterIdText = await this.clusterIdDisplay.textContent();
      const clusterId = clusterIdText?.match(/\d+/) ? parseInt(clusterIdText.match(/\d+/)![0], 10) : null;
      
      const keywordsText = await this.keywordsDisplay.locator('..').textContent();
      const keywords = keywordsText ? keywordsText.split(',').map(k => k.trim()) : [];
      
      const similarArticles: string[] = [];
      const articleItems = await this.similarArticlesList.locator('li').all();
      for (const item of articleItems) {
        const text = await item.textContent();
        if (text) similarArticles.push(text.trim());
      }
      
      return {
        articleTitle: articleTitle || '',
        clusterId,
        keywords,
        similarArticles,
      };
    } catch {
      return null;
    }
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
   * Check if result is displayed
   */
  async hasResult(): Promise<boolean> {
    try {
      return await this.resultContainer.isVisible();
    } catch {
      return false;
    }
  }
}

