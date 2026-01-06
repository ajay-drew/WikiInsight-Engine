import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Monitoring page
 */
export class MonitoringPage {
  readonly page: Page;
  
  // UI elements
  readonly loadingIndicator: Locator;
  readonly errorMessage: Locator;
  readonly pipelineStatusSection: Locator;
  readonly metricsSection: Locator;
  readonly driftSection: Locator;
  readonly stabilitySection: Locator;

  constructor(page: Page) {
    this.page = page;
    this.loadingIndicator = page.locator('[class*="animate-spin"]').first();
    this.errorMessage = page.locator('text=/Failed|Error/i').first();
    this.pipelineStatusSection = page.locator('text=/Pipeline Status|Ingestion|Preprocessing/i');
    this.metricsSection = page.locator('text=/API Metrics|Total Requests|Endpoints/i');
    this.driftSection = page.locator('text=/Drift|Data Drift/i');
    this.stabilitySection = page.locator('text=/Stability|Cluster Stability/i');
  }

  /**
   * Navigate to monitoring page
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
    await this.page.getByRole('button', { name: 'Monitoring' }).click();
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Load monitoring data
   */
  async loadMonitoringData(): Promise<void> {
    await this.goto();
    // Wait for data to load
    await this.page.waitForTimeout(1000);
  }

  /**
   * Get pipeline status
   */
  async getPipelineStatus(): Promise<{
    ingestion: boolean;
    preprocessing: boolean;
    clustering: boolean;
  } | null> {
    try {
      // Try multiple selectors to find pipeline status
      const statusSelectors = [
        this.pipelineStatusSection,
        this.page.locator('text=/Pipeline Status/i'),
        this.page.locator('text=/Ingestion/i'),
        this.page.locator('[class*="status"]'),
      ];
      
      for (const selector of statusSelectors) {
        try {
          if (await selector.isVisible({ timeout: 1000 })) {
            const statusText = await selector.textContent();
            return {
              ingestion: statusText?.toLowerCase().includes('ingestion') || false,
              preprocessing: statusText?.toLowerCase().includes('preprocessing') || false,
              clustering: statusText?.toLowerCase().includes('clustering') || false,
            };
          }
        } catch {
          continue;
        }
      }
      
      // If no status section found, return default
      return {
        ingestion: false,
        preprocessing: false,
        clustering: false,
      };
    } catch {
      return null;
    }
  }

  /**
   * Get metrics summary
   */
  async getMetrics(): Promise<{
    totalRequests: number | null;
    endpoints: string[];
  } | null> {
    try {
      const metricsText = await this.metricsSection.textContent();
      const totalMatch = metricsText?.match(/(\d+)\s*(total|requests)/i);
      const totalRequests = totalMatch ? parseInt(totalMatch[1], 10) : null;
      
      const endpoints: string[] = [];
      const endpointElements = await this.page.locator('text=/\\/api\\//i').all();
      for (const element of endpointElements) {
        const text = await element.textContent();
        if (text) endpoints.push(text.trim());
      }
      
      return {
        totalRequests,
        endpoints,
      };
    } catch {
      return null;
    }
  }

  /**
   * Get drift report
   */
  async getDriftReport(): Promise<{
    driftDetected: boolean | null;
  } | null> {
    try {
      const driftText = await this.driftSection.textContent();
      return {
        driftDetected: driftText?.toLowerCase().includes('detected') || null,
      };
    } catch {
      return null;
    }
  }

  /**
   * Get stability metrics
   */
  async getStabilityMetrics(): Promise<{
    ari: number | null;
    nmi: number | null;
  } | null> {
    try {
      const stabilityText = await this.stabilitySection.textContent();
      const ariMatch = stabilityText?.match(/ari[:\s]+([\d.]+)/i);
      const nmiMatch = stabilityText?.match(/nmi[:\s]+([\d.]+)/i);
      
      return {
        ari: ariMatch ? parseFloat(ariMatch[1]) : null,
        nmi: nmiMatch ? parseFloat(nmiMatch[1]) : null,
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
   * Wait for monitoring data to load
   */
  async waitForData(timeout: number = 10000): Promise<void> {
    // Try multiple selectors - the page might show different content based on loading state
    try {
      await Promise.race([
        this.page.waitForSelector('h2:has-text("Pipeline Status")', { timeout }),
        this.page.waitForSelector('text=/Pipeline Status/i', { timeout }),
        this.page.waitForSelector('text=/API Metrics/i', { timeout }),
        this.page.waitForSelector('text=/Loading monitoring data/i', { timeout }),
        this.page.waitForSelector('[class*="animate-spin"]', { timeout }),
      ]);
    } catch {
      // If none found, wait a bit for page to settle
      await this.page.waitForTimeout(1000);
    }
  }
}

