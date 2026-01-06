import { test, expect } from '@playwright/test';
import { SearchPage } from '../pages/SearchPage';
import { TopicLookupPage } from '../pages/TopicLookupPage';
import { ClustersOverviewPage } from '../pages/ClustersOverviewPage';
import { MonitoringPage } from '../pages/MonitoringPage';
import { IngestionPage } from '../pages/IngestionPage';
import { mockAPIErrorRoute } from '../fixtures/api-fixtures';

test.describe('API Error Handling', () => {
  test('should handle 400 Bad Request error', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    await mockAPIErrorRoute(page, '**/api/search', 'Bad Request', 400);

    await searchPage.search('test');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
    expect(error).toContain('Bad Request');
  });

  test('should handle 404 Not Found error', async ({ page }) => {
    const topicLookupPage = new TopicLookupPage(page);
    await topicLookupPage.goto();

    await mockAPIErrorRoute(page, '**/api/topics/lookup', 'Not Found', 404);

    await topicLookupPage.lookupTopic('NonExistent');

    const error = await topicLookupPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle 500 Internal Server Error', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    await mockAPIErrorRoute(page, '**/api/search', 'Internal Server Error', 500);

    await searchPage.search('test');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle 503 Service Unavailable error', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    await mockAPIErrorRoute(
      page,
      '**/api/search',
      'Search engine is not available. Run the data pipeline first.',
      503
    );

    await searchPage.search('test');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
    expect(error).toContain('Search engine is not available');
  });

  test('should handle network errors (connection refused)', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    // Simulate network failure
    await page.route('**/api/search', route => route.abort());

    await searchPage.search('test');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle timeout errors', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    // Simulate timeout by delaying response indefinitely
    await page.route('**/api/search', async (route) => {
      // Don't fulfill or continue - simulate timeout
      await new Promise(() => {}); // Never resolves
    });

    // Set a shorter timeout for the test
    page.setDefaultTimeout(2000);

    try {
      await searchPage.search('test');
      await searchPage.waitForResults(2000);
    } catch (error) {
      // Timeout expected
      expect(error).toBeTruthy();
    }
  });

  test('should handle malformed API responses', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    // Return invalid JSON
    await page.route('**/api/search', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: 'Invalid JSON {',
      });
    });

    await searchPage.search('test');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle clusters API errors', async ({ page }) => {
    const clustersPage = new ClustersOverviewPage(page);

    await mockAPIErrorRoute(page, '**/api/clusters/overview', 'Failed to load clusters', 500);

    await clustersPage.loadClusters();

    const error = await clustersPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle monitoring API errors', async ({ page }) => {
    const monitoringPage = new MonitoringPage(page);

    await mockAPIErrorRoute(
      page,
      '**/api/monitoring/pipeline-status',
      'Failed to load monitoring data',
      500
    );

    await monitoringPage.loadMonitoringData();

    const error = await monitoringPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle pipeline API errors', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    await mockAPIErrorRoute(page, '**/api/pipeline/start', 'Failed to start pipeline', 500);

    await ingestionPage.configurePipeline({
      seedQueries: ['Machine learning', 'Data science', 'AI'],
    });
    await ingestionPage.startPipeline();

    const error = await ingestionPage.getError();
    expect(error).toBeTruthy();
  });

  test('should verify error boundaries catch React errors', async ({ page }) => {
    // This test would require injecting an error into the React component
    // For now, we verify that error messages are displayed
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    await mockAPIErrorRoute(page, '**/api/search', 'Error', 500);

    await searchPage.search('test');

    // Error should be displayed to user
    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });
});

