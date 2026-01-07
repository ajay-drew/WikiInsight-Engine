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
    // Updated to use NavLinks instead of buttons
    this.searchButton = page.locator('nav a[href="/search"]');
    this.topicLookupButton = page.locator('nav a[href="/lookup"]');
    this.clustersButton = page.locator('nav a[href="/clusters"]');
    this.monitoringButton = page.locator('nav a[href="/monitoring"]');
    this.ingestionButton = page.locator('nav a[href="/ingestion"]');
  }

  /**
   * Navigate to a specific page
   */
  async navigateTo(pageName: 'search' | 'lookup' | 'clusters' | 'monitoring' | 'ingestion' | 'dashboard'): Promise<void> {
    // Wait for page to be fully loaded first
    await this.page.waitForLoadState('domcontentloaded');
    
    // Wait for navigation container to be visible
    const navSelector = 'nav';
    await this.page.waitForSelector(navSelector, { timeout: 15000, state: 'visible' });
    
    // Use direct navigation via URL for reliability
    const routes: Record<string, string> = {
      'search': '/search',
      'lookup': '/lookup',
      'clusters': '/clusters',
      'monitoring': '/monitoring',
      'ingestion': '/ingestion',
      'dashboard': '/dashboard',
    };
    
    const route = routes[pageName];
    if (route) {
      await this.page.goto(route);
      await this.page.waitForLoadState('networkidle');
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Get the currently active page
   */
  async getActivePage(): Promise<string> {
    // Check which nav link has the active class (bg-sky-500)
    const activeLink = this.page.locator('nav a.bg-sky-500').first();
    if (await activeLink.count() > 0) {
      const href = await activeLink.getAttribute('href');
      return href?.replace('/', '') || '';
    }
    // Fallback: check URL
    const url = this.page.url();
    const match = url.match(/\/([^/]+)$/);
    return match ? match[1] : '';
  }

  /**
   * Verify page is active
   */
  async verifyPageActive(pageName: string): Promise<boolean> {
    const activePage = await this.getActivePage();
    return activePage.toLowerCase().includes(pageName.toLowerCase());
  }
}

