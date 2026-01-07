import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Clusters Overview page
 */
export class ClustersOverviewPage {
  readonly page: Page;
  
  // UI elements
  readonly loadingIndicator: Locator;
  readonly errorMessage: Locator;
  readonly clustersTable: Locator;
  readonly clusterRows: Locator;
  readonly selectedClusterDetails: Locator;
  readonly tableTab: Locator;
  readonly graphTab: Locator;
  readonly graphContainer: Locator;
  readonly searchInput: Locator;

  constructor(page: Page) {
    this.page = page;
    this.loadingIndicator = page.locator('[class*="animate-spin"]').first();
    this.errorMessage = page.locator('text=/Failed|Error/i').first();
    this.clustersTable = page.locator('table, div:has-text("Cluster")');
    this.clusterRows = page.locator('tr, div[class*="border"]').filter({ hasText: /Cluster|ID/i });
    this.selectedClusterDetails = page.locator('div:has-text("Keywords"), div:has-text("Top Articles")');
    this.tableTab = page.getByRole('button', { name: /Table/i }).or(page.locator('button:has-text("table")'));
    this.graphTab = page.getByRole('button', { name: /Graph/i }).or(page.locator('button:has-text("graph")'));
    this.graphContainer = page.locator('[class*="graph"], canvas, svg').first();
    this.searchInput = page.locator('input[type="text"]').first();
  }

  /**
   * Navigate to clusters page
   */
  async goto(): Promise<void> {
    await this.page.goto('/clusters');
    await this.page.waitForLoadState('networkidle');
    // Wait for page content to be visible
    await this.page.waitForTimeout(500);
  }

  /**
   * Wait for clusters to load
   */
  async waitForClusters(timeout: number = 10000): Promise<void> {
    await this.page.waitForSelector('table, div:has-text("Cluster")', { timeout });
  }

  /**
   * Load clusters overview
   */
  async loadClusters(): Promise<void> {
    await this.goto();
    await this.waitForClusters();
  }

  /**
   * Select a cluster by ID
   */
  async selectCluster(clusterId: number): Promise<void> {
    // Click on cluster row or button with cluster ID
    const clusterElement = this.page.locator(`text=${clusterId}`).or(this.page.locator(`[data-cluster-id="${clusterId}"]`));
    await clusterElement.first().click();
    await this.page.waitForTimeout(500); // Wait for details to load
  }

  /**
   * Switch to graph tab
   */
  async switchToGraphTab(): Promise<void> {
    await this.graphTab.click();
    await this.page.waitForTimeout(500);
  }

  /**
   * Switch to table tab
   */
  async switchToTableTab(): Promise<void> {
    await this.tableTab.click();
    await this.page.waitForTimeout(500);
  }

  /**
   * Get cluster details
   */
  async getClusterDetails(): Promise<{
    clusterId: number | null;
    keywords: string[];
    topArticles: string[];
  } | null> {
    try {
      // Wait for cluster details to appear
      await this.page.waitForTimeout(1000);
      
      // Try multiple selectors for cluster ID
      const clusterIdSelectors = [
        this.page.locator('text=/Cluster ID[\\s:]+(\\d+)/i'),
        this.page.locator('h2, h3').filter({ hasText: /Cluster/i }),
        this.page.locator('[class*="cluster"]').filter({ hasText: /\d+/ }),
      ];
      
      let clusterId: number | null = null;
      for (const selector of clusterIdSelectors) {
        try {
          const text = await selector.first().textContent({ timeout: 1000 });
          const match = text?.match(/\d+/);
          if (match) {
            clusterId = parseInt(match[0], 10);
            break;
          }
        } catch {
          continue;
        }
      }
      
      // Try to find keywords
      const keywords: string[] = [];
      try {
        const keywordsSection = this.page.locator('text=/Keywords/i').locator('..');
        const keywordsText = await keywordsSection.textContent({ timeout: 1000 });
        if (keywordsText) {
          const keywordList = keywordsText.split(',').map(k => k.trim().replace(/Keywords?:?\s*/i, ''));
          keywords.push(...keywordList.filter(k => k.length > 0));
        }
      } catch {
        // Keywords not found, continue
      }
      
      // Try to find articles
      const articles: string[] = [];
      try {
        const articlesSection = this.page.locator('text=/Top Articles|Articles/i').locator('..');
        const articleElements = await articlesSection.locator('li, div, span').all();
        for (const element of articleElements.slice(0, 5)) {
          const text = await element.textContent();
          if (text && text.length > 3 && !text.toLowerCase().includes('article')) {
            articles.push(text.trim());
          }
        }
      } catch {
        // Articles not found, continue
      }
      
      // Return details even if some fields are empty
      return {
        clusterId,
        keywords,
        topArticles: articles,
      };
    } catch {
      return null;
    }
  }

  /**
   * Get all cluster IDs from table
   */
  async getClusterIds(): Promise<number[]> {
    const ids: number[] = [];
    try {
      const rows = await this.clusterRows.all();
      for (const row of rows) {
        const text = await row.textContent();
        const match = text?.match(/\d+/);
        if (match) {
          ids.push(parseInt(match[0], 10));
        }
      }
    } catch {
      // Return empty array if table not found
    }
    return ids;
  }

  /**
   * Search/filter clusters
   */
  async searchClusters(query: string): Promise<void> {
    await this.searchInput.fill(query);
    await this.page.waitForTimeout(300);
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
   * Check if graph is displayed
   */
  async isGraphVisible(): Promise<boolean> {
    try {
      return await this.graphContainer.isVisible();
    } catch {
      return false;
    }
  }
}

