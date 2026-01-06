import { Page, Locator } from '@playwright/test';

/**
 * Page Object Model for Ingestion page
 */
export class IngestionPage {
  readonly page: Page;
  
  // Form elements
  readonly queryInputs: Locator;
  readonly addQueryButton: Locator;
  readonly removeQueryButtons: Locator;
  readonly perQueryLimitSlider: Locator;
  readonly maxArticlesSlider: Locator;
  readonly startPipelineButton: Locator;
  readonly errorMessage: Locator;
  readonly progressContainer: Locator;
  readonly progressBar: Locator;
  readonly stageStatuses: Locator;

  constructor(page: Page) {
    this.page = page;
    this.queryInputs = page.locator('input[type="text"][placeholder*="Query"]');
    this.addQueryButton = page.getByRole('button', { name: /Add Query/i });
    this.removeQueryButtons = page.getByRole('button', { name: /Remove/i });
    this.perQueryLimitSlider = page.locator('input[type="range"]').first();
    this.maxArticlesSlider = page.locator('input[type="range"]').last();
    this.startPipelineButton = page.getByRole('button', { name: /Start Pipeline/i });
    this.errorMessage = page.locator('p.text-red-400, div.border-red-500').first();
    this.progressContainer = page.locator('text=/Pipeline Progress|Overall Progress/i');
    this.progressBar = page.locator('[class*="progress"], [style*="width"]').first();
    this.stageStatuses = page.locator('text=/ingestion|preprocessing|clustering|build_graph/i');
  }

  /**
   * Navigate to ingestion page
   */
  async goto(): Promise<void> {
    await this.page.goto('/');
    await this.page.getByRole('button', { name: 'Ingestion' }).click();
    await this.queryInputs.first().waitFor({ state: 'visible', timeout: 5000 });
  }

  /**
   * Configure pipeline with seed queries
   */
  async configurePipeline(config: {
    seedQueries: string[];
    perQueryLimit?: number;
    maxArticles?: number;
  }): Promise<void> {
    // Set seed queries
    const inputs = await this.queryInputs.all();
    for (let i = 0; i < config.seedQueries.length; i++) {
      if (i < inputs.length) {
        await inputs[i].fill(config.seedQueries[i]);
      } else {
        // Add more query inputs if needed
        await this.addQueryButton.click();
        await this.page.waitForTimeout(200);
        const newInputs = await this.queryInputs.all();
        await newInputs[i].fill(config.seedQueries[i]);
      }
    }
    
    // Set per query limit if provided
    if (config.perQueryLimit !== undefined) {
      await this.perQueryLimitSlider.fill(config.perQueryLimit.toString());
    }
    
    // Set max articles if provided
    if (config.maxArticles !== undefined) {
      await this.maxArticlesSlider.fill(config.maxArticles.toString());
    }
  }

  /**
   * Start pipeline
   */
  async startPipeline(): Promise<void> {
    await this.startPipelineButton.click();
  }

  /**
   * Wait for progress updates
   */
  async waitForProgress(timeout: number = 30000): Promise<void> {
    // Wait for any progress indicator - could be progress bar, text, or stage indicator
    try {
      await Promise.race([
        this.page.waitForSelector('text=/Pipeline Progress|Overall Progress|Progress/i', { timeout }),
        this.page.waitForSelector('[class*="progress"]', { timeout }),
        this.page.waitForSelector('text=/ingestion|preprocessing|clustering|build_graph/i', { timeout }),
      ]);
    } catch {
      // If no progress indicator found, wait a bit for page to update
      await this.page.waitForTimeout(1000);
    }
  }

  /**
   * Get progress information
   */
  async getProgress(): Promise<{
    overallProgress: number;
    currentStage: string | null;
    stageProgress: Record<string, number>;
  } | null> {
    try {
      const progressText = await this.progressContainer.textContent();
      const progressMatch = progressText?.match(/(\d+\.?\d*)%/);
      const overallProgress = progressMatch ? parseFloat(progressMatch[1]) : 0;
      
      const currentStageText = await this.stageStatuses.first().textContent();
      const currentStage = currentStageText?.toLowerCase() || null;
      
      const stageProgress: Record<string, number> = {};
      const stages = ['ingestion', 'preprocessing', 'clustering', 'build_graph'];
      for (const stage of stages) {
        const stageElement = this.page.locator(`text=/${stage}/i`).first();
        const stageText = await stageElement.textContent();
        const match = stageText?.match(/(\d+)%/);
        stageProgress[stage] = match ? parseInt(match[1], 10) : 0;
      }
      
      return {
        overallProgress,
        currentStage,
        stageProgress,
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
      // Wait a bit for error to appear
      await this.page.waitForTimeout(1000);
      // Try multiple error message selectors
      const errorSelectors = [
        'p.text-red-400',
        'div.border-red-500',
        'div:has-text("Must have")',
        'div:has-text("3-6")',
        'div:has-text("validation")',
      ];
      
      for (const selector of errorSelectors) {
        try {
          const element = this.page.locator(selector).first();
          if (await element.isVisible({ timeout: 1000 })) {
            const text = await element.textContent();
            if (text && text.trim()) {
              return text.trim();
            }
          }
        } catch {
          continue;
        }
      }
      
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Check if pipeline is running
   */
  async isRunning(): Promise<boolean> {
    try {
      const buttonText = await this.startPipelineButton.textContent();
      return buttonText?.includes('Running') || false;
    } catch {
      return false;
    }
  }

  /**
   * Check if progress is displayed
   */
  async hasProgress(): Promise<boolean> {
    try {
      return await this.progressContainer.isVisible();
    } catch {
      return false;
    }
  }

  /**
   * Wait for pipeline completion
   */
  async waitForCompletion(timeout: number = 300000): Promise<void> {
    // Wait for all stages to be completed
    await this.page.waitForSelector('text=/Complete|completed/i', { timeout });
  }
}

