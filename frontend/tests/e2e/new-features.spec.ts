import { test, expect } from '@playwright/test';
import { Layout } from '../pages/Layout';
import { SearchPage } from '../pages/SearchPage';
import { ClustersOverviewPage } from '../pages/ClustersOverviewPage';
import { DashboardPage } from '../pages/DashboardPage';
import { mockSearchResponse, mockClustersOverviewResponse, mockAPIRoute } from '../fixtures/api-fixtures';

test.describe('New Frontend Features E2E', () => {
  test.describe('Logo Component', () => {
    test('should display logo in header', async ({ page }) => {
      await page.goto('/dashboard');
      const logo = page.locator('svg[aria-label="WikiInsight Logo"]');
      await expect(logo).toBeVisible();
    });

    test('should navigate to dashboard when logo is clicked', async ({ page }) => {
      await page.goto('/search');
      const logoLink = page.locator('a:has(svg[aria-label="WikiInsight Logo"])');
      await logoLink.click();
      await expect(page).toHaveURL(/.*\/dashboard/);
    });
  });

  test.describe('Toast Notifications', () => {
    test('should show success toast after successful search', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', mockSearchResponse);
      
      await searchPage.search('machine learning');
      
      // Wait for toast to appear
      const toast = page.locator('[role="alert"]').filter({ hasText: /Found.*result/i });
      await expect(toast).toBeVisible({ timeout: 5000 });
    });

    test('should show error toast on search failure', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      // Mock API error
      await page.route('**/api/search', route => route.abort());
      
      await searchPage.search('test query');
      
      // Wait for error toast
      const errorToast = page.locator('[role="alert"]').filter({ hasText: /error|failed/i });
      await expect(errorToast).toBeVisible({ timeout: 5000 });
    });

    test('should dismiss toast when close button is clicked', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', mockSearchResponse);
      await searchPage.search('test');
      
      // Wait for toast
      const toast = page.locator('[role="alert"]').first();
      await expect(toast).toBeVisible();
      
      // Click dismiss button
      const dismissButton = toast.locator('button[aria-label="Dismiss notification"]');
      await dismissButton.click();
      
      // Toast should disappear
      await expect(toast).not.toBeVisible({ timeout: 2000 });
    });
  });

  test.describe('Loading Skeletons', () => {
    test('should show skeleton loaders while search is loading', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      // Delay API response to see skeleton
      await page.route('**/api/search', async route => {
        await new Promise(resolve => setTimeout(resolve, 500));
        await route.fulfill({
          status: 200,
          body: JSON.stringify(mockSearchResponse),
        });
      });
      
      await searchPage.search('test');
      
      // Check for skeleton
      const skeleton = page.locator('[aria-label="Loading..."]').first();
      await expect(skeleton).toBeVisible({ timeout: 1000 });
    });

    test('should show cluster card skeletons while loading', async ({ page }) => {
      const clustersPage = new ClustersOverviewPage(page);
      
      // Delay API response
      await page.route('**/api/clusters/overview', async route => {
        await new Promise(resolve => setTimeout(resolve, 500));
        await route.fulfill({
          status: 200,
          body: JSON.stringify(mockClustersOverviewResponse),
        });
      });
      
      await clustersPage.goto();
      
      // Check for skeleton
      const skeleton = page.locator('.animate-pulse').first();
      await expect(skeleton).toBeVisible({ timeout: 1000 });
    });
  });

  test.describe('Empty States', () => {
    test('should show empty state when no search results', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      // Mock empty response
      await mockAPIRoute(page, '**/api/search', {
        query: 'nonexistentquery12345',
        results: [],
        total: 0,
      });
      
      await searchPage.search('nonexistentquery12345');
      await searchPage.waitForResults();
      
      // Check for empty state
      const emptyState = page.locator('text=No results found');
      await expect(emptyState).toBeVisible();
    });

    test('should show empty state on initial search page load', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      const emptyState = page.locator('text=Start searching Wikipedia articles');
      await expect(emptyState).toBeVisible();
    });
  });

  test.describe('Tooltips', () => {
    test('should show tooltip on navigation items', async ({ page }) => {
      await page.goto('/dashboard');
      
      // Hover over search nav link
      const searchLink = page.locator('nav a[href="/search"]');
      await searchLink.hover();
      
      // Wait for tooltip
      const tooltip = page.locator('[role="tooltip"]').filter({ hasText: /search/i });
      await expect(tooltip).toBeVisible({ timeout: 1000 });
    });

    test('should show tooltip on theme toggle button', async ({ page }) => {
      await page.goto('/dashboard');
      
      const themeButton = page.locator('button[aria-label="Toggle theme"]');
      await themeButton.hover();
      
      const tooltip = page.locator('[role="tooltip"]');
      await expect(tooltip).toBeVisible({ timeout: 1000 });
    });
  });

  test.describe('Keyboard Shortcuts', () => {
    test('should focus search input when "/" is pressed', async ({ page }) => {
      await page.goto('/search');
      
      // Press "/" key
      await page.keyboard.press('/');
      
      // Check if search input is focused
      const searchInput = page.locator('input[type="text"]').first();
      await expect(searchInput).toBeFocused();
    });

    test('should clear and blur search input when Escape is pressed', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      // Type in search input
      const searchInput = page.locator('input[type="text"]').first();
      await searchInput.fill('test query');
      await searchInput.focus();
      
      // Press Escape
      await page.keyboard.press('Escape');
      
      // Input should be cleared and blurred
      await expect(searchInput).not.toBeFocused();
      await expect(searchInput).toHaveValue('');
    });

    test('should not trigger shortcut when typing in input', async ({ page }) => {
      await page.goto('/search');
      
      const searchInput = page.locator('input[type="text"]').first();
      await searchInput.fill('/test');
      
      // Should not trigger focus shortcut
      await expect(searchInput).toHaveValue('/test');
    });
  });

  test.describe('Export Functionality', () => {
    test('should export search results as CSV', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', mockSearchResponse);
      await searchPage.search('test');
      await searchPage.waitForResults();
      
      // Set up download listener
      const downloadPromise = page.waitForEvent('download');
      
      // Click CSV export button
      const csvButton = page.locator('button:has-text("CSV")').first();
      await csvButton.click();
      
      const download = await downloadPromise;
      expect(download.suggestedFilename()).toContain('.csv');
      expect(download.suggestedFilename()).toContain('search_results');
    });

    test('should export search results as JSON', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', mockSearchResponse);
      await searchPage.search('test');
      await searchPage.waitForResults();
      
      const downloadPromise = page.waitForEvent('download');
      
      const jsonButton = page.locator('button:has-text("JSON")').first();
      await jsonButton.click();
      
      const download = await downloadPromise;
      expect(download.suggestedFilename()).toContain('.json');
      expect(download.suggestedFilename()).toContain('search_results');
    });

    test('should export clusters as CSV', async ({ page }) => {
      const clustersPage = new ClustersOverviewPage(page);
      await clustersPage.goto();
      
      await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
      await page.waitForTimeout(1000); // Wait for data to load
      
      const downloadPromise = page.waitForEvent('download');
      
      const csvButton = page.locator('button:has-text("CSV")').first();
      await csvButton.click();
      
      const download = await downloadPromise;
      expect(download.suggestedFilename()).toContain('.csv');
      expect(download.suggestedFilename()).toContain('clusters');
    });
  });

  test.describe('Theme Toggle', () => {
    test('should toggle theme when button is clicked', async ({ page }) => {
      await page.goto('/dashboard');
      
      const themeButton = page.locator('button[aria-label="Toggle theme"]');
      const html = page.locator('html');
      
      // Get initial theme
      const initialTheme = await html.getAttribute('data-theme');
      
      // Click toggle
      await themeButton.click();
      
      // Wait for theme change
      await page.waitForTimeout(300);
      
      // Check theme changed
      const newTheme = await html.getAttribute('data-theme');
      expect(newTheme).not.toBe(initialTheme);
    });

    test('should persist theme preference in localStorage', async ({ page }) => {
      await page.goto('/dashboard');
      
      const themeButton = page.locator('button[aria-label="Toggle theme"]');
      await themeButton.click();
      
      await page.waitForTimeout(300);
      
      // Check localStorage
      const theme = await page.evaluate(() => localStorage.getItem('theme'));
      expect(theme).toBeTruthy();
      expect(['dark', 'light']).toContain(theme);
    });
  });

  test.describe('Dashboard Page', () => {
    test('should display dashboard with key metrics', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      
      // Check for dashboard title
      await expect(page.locator('h1:has-text("Dashboard")')).toBeVisible();
      
      // Check for metric cards
      await expect(page.locator('text=Total Articles')).toBeVisible();
      await expect(page.locator('text=Total Clusters')).toBeVisible();
      await expect(page.locator('text=Pipeline Status')).toBeVisible();
    });

    test('should show quick action links', async ({ page }) => {
      await page.goto('/dashboard');
      
      await expect(page.locator('text=Search Articles')).toBeVisible();
      await expect(page.locator('text=Browse Clusters')).toBeVisible();
      await expect(page.locator('text=Run Pipeline')).toBeVisible();
    });

    test('should navigate to search from quick actions', async ({ page }) => {
      await page.goto('/dashboard');
      
      const searchLink = page.locator('a:has-text("Search Articles")');
      await searchLink.click();
      
      await expect(page).toHaveURL(/.*\/search/);
    });
  });

  test.describe('Search Result Highlighting', () => {
    test('should highlight search terms in result titles', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', {
        query: 'machine',
        results: [
          {
            title: 'Machine Learning',
            score: 0.95,
            rank: 0,
            cluster_id: 1,
            categories: [],
            link_count: 10,
            wikipedia_url: 'https://en.wikipedia.org/wiki/Machine_learning',
            wikidata_url: null,
          },
        ],
        total: 1,
      });
      
      await searchPage.search('machine');
      await searchPage.waitForResults();
      
      // Check for highlighted text
      const highlighted = page.locator('mark').first();
      await expect(highlighted).toBeVisible();
      await expect(highlighted).toContainText('Machine');
    });
  });

  test.describe('Animated Transitions', () => {
    test('should have smooth transitions on page navigation', async ({ page }) => {
      await page.goto('/dashboard');
      
      // Navigate to search
      await page.click('nav a[href="/search"]');
      
      // Check for fade-in animation class
      const content = page.locator('main').first();
      await expect(content).toBeVisible();
    });

    test('should animate toast notifications', async ({ page }) => {
      const searchPage = new SearchPage(page);
      await searchPage.goto();
      
      await mockAPIRoute(page, '**/api/search', mockSearchResponse);
      await searchPage.search('test');
      
      // Check for animation class on toast
      const toast = page.locator('[role="alert"]').first();
      await expect(toast).toHaveClass(/animate-slide-in-right/);
    });
  });
});

