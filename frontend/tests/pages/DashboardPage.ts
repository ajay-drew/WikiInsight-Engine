import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Dashboard page
 */
export class DashboardPage {
  readonly page: Page;
  
  // UI elements
  readonly dashboardTitle: Locator;
  readonly totalArticlesCard: Locator;
  readonly totalClustersCard: Locator;
  readonly pipelineStatusCard: Locator;
  readonly apiRequestsCard: Locator;
  readonly quickActionsSection: Locator;
  readonly systemStatusSection: Locator;

  constructor(page: Page) {
    this.page = page;
    this.dashboardTitle = page.locator('h1:has-text("Dashboard")');
    this.totalArticlesCard = page.locator('text=Total Articles').locator('..');
    this.totalClustersCard = page.locator('text=Total Clusters').locator('..');
    this.pipelineStatusCard = page.locator('text=Pipeline Status').locator('..');
    this.apiRequestsCard = page.locator('text=API Requests').locator('..');
    this.quickActionsSection = page.locator('text=Quick Actions').locator('..');
    this.systemStatusSection = page.locator('text=System Status').locator('..');
  }

  /**
   * Navigate to dashboard page
   */
  async goto(): Promise<void> {
    await this.page.goto('/dashboard');
    await this.page.waitForLoadState('networkidle');
    await this.dashboardTitle.waitFor({ state: 'visible', timeout: 5000 });
  }

  /**
   * Get total articles count
   */
  async getTotalArticles(): Promise<number> {
    try {
      const text = await this.totalArticlesCard.locator('text=/\\d+/').first().textContent();
      if (text) {
        const match = text.match(/(\d+)/);
        return match ? parseInt(match[1].replace(/,/g, ''), 10) : 0;
      }
    } catch {
      // Return 0 if not found
    }
    return 0;
  }

  /**
   * Get total clusters count
   */
  async getTotalClusters(): Promise<number> {
    try {
      const text = await this.totalClustersCard.locator('text=/\\d+/').first().textContent();
      if (text) {
        const match = text.match(/(\d+)/);
        return match ? parseInt(match[1], 10) : 0;
      }
    } catch {
      // Return 0 if not found
    }
    return 0;
  }

  /**
   * Get pipeline status
   */
  async getPipelineStatus(): Promise<string> {
    try {
      const status = await this.pipelineStatusCard.locator('text=/Ready|Not Ready/i').first().textContent();
      return status?.trim() || '';
    } catch {
      return '';
    }
  }

  /**
   * Click quick action link
   */
  async clickQuickAction(actionName: 'Search Articles' | 'Browse Clusters' | 'Run Pipeline'): Promise<void> {
    const actionLink = this.page.locator(`a:has-text("${actionName}")`);
    await actionLink.click();
    await this.page.waitForLoadState('networkidle');
  }
}

