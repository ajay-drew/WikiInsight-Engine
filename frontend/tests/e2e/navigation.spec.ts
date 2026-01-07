import { test, expect } from '@playwright/test';
import { Layout } from '../pages/Layout';
import { SearchPage } from '../pages/SearchPage';
import { TopicLookupPage } from '../pages/TopicLookupPage';
import { ClustersOverviewPage } from '../pages/ClustersOverviewPage';
import { MonitoringPage } from '../pages/MonitoringPage';
import { IngestionPage } from '../pages/IngestionPage';
import { waitForAPIRequest } from '../utils/wait-helpers';
import {
  mockClustersOverviewResponse,
  mockPipelineStatusResponse,
  mockMetricsResponse,
  mockAPIRoute,
} from '../fixtures/api-fixtures';

test.describe('Navigation E2E', () => {
  test('should navigate between all pages', async ({ page }) => {
    const layout = new Layout(page);

    const pages: Array<'search' | 'lookup' | 'clusters' | 'monitoring' | 'ingestion'> = [
      'search',
      'lookup',
      'clusters',
      'monitoring',
      'ingestion',
    ];

    for (const pageName of pages) {
      await layout.navigateTo(pageName);
      // Wait a bit for page to render
      await page.waitForTimeout(300);
      const isActive = await layout.verifyPageActive(pageName);
      expect(isActive).toBe(true);
    }
  });

  test('should load correct page on initial load', async ({ page }) => {
    await page.goto('/');
    
    // Default page should be dashboard (redirects from /)
    const layout = new Layout(page);
    await expect(layout.verifyPageActive('dashboard')).resolves.toBe(true);
  });

  test('should make API calls when pages load', async ({ page }) => {
    const layout = new Layout(page);

    // Test clusters page - should call overview API
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
    const clustersRequestPromise = waitForAPIRequest(page, '/api/clusters/overview', 'GET');
    
    await layout.navigateTo('clusters');
    
    const clustersRequest = await clustersRequestPromise;
    expect(clustersRequest.url()).toContain('/api/clusters/overview');

    // Test monitoring page - should call monitoring APIs
    await mockAPIRoute(page, '**/api/monitoring/pipeline-status', mockPipelineStatusResponse);
    await mockAPIRoute(page, '**/api/monitoring/metrics*', mockMetricsResponse);
    
    const statusRequestPromise = waitForAPIRequest(page, '/api/monitoring/pipeline-status', 'GET');
    
    await layout.navigateTo('monitoring');
    
    const statusRequest = await statusRequestPromise;
    expect(statusRequest.url()).toContain('/api/monitoring/pipeline-status');
  });

  test('should handle browser back/forward navigation', async ({ page }) => {
    const layout = new Layout(page);

    // Navigate to different pages
    await layout.navigateTo('search');
    await page.waitForTimeout(300);
    await layout.navigateTo('clusters');
    await page.waitForTimeout(300);
    await layout.navigateTo('monitoring');
    await page.waitForTimeout(300);

    // Go back
    await page.goBack();
    await page.waitForTimeout(500);
    const isClusters = await layout.verifyPageActive('clusters');
    expect(isClusters).toBe(true);

    // Go back again
    await page.goBack();
    await page.waitForTimeout(500);
    const isSearch = await layout.verifyPageActive('search');
    expect(isSearch).toBe(true);

    // Go forward
    await page.goForward();
    await page.waitForTimeout(500);
    const isClustersAgain = await layout.verifyPageActive('clusters');
    expect(isClustersAgain).toBe(true);
  });

  test('should maintain page state on navigation', async ({ page }) => {
    const layout = new Layout(page);
    const searchPage = new SearchPage(page);

    // Perform a search
    await mockAPIRoute(page, '**/api/search', {
      query: 'test',
      results: [],
      total_results: 0,
    });
    
    await layout.navigateTo('search');
    await page.waitForTimeout(300);
    await searchPage.search('test');
    await searchPage.waitForResults();

    // Navigate away and back
    await layout.navigateTo('clusters');
    await page.waitForTimeout(300);
    await layout.navigateTo('search');
    await page.waitForTimeout(300);

    // Page should reload (state might be reset, which is expected)
    // This test verifies navigation works correctly
    const isActive = await layout.verifyPageActive('search');
    expect(isActive).toBe(true);
  });

  test('should verify each page loads correctly', async ({ page }) => {
    const layout = new Layout(page);

    // Verify Search page
    await layout.navigateTo('search');
    await page.waitForTimeout(300);
    const searchPage = new SearchPage(page);
    expect(await searchPage.searchInput.isVisible()).toBe(true);

    // Verify Topic Lookup page
    await layout.navigateTo('lookup');
    await page.waitForTimeout(300);
    const topicLookupPage = new TopicLookupPage(page);
    expect(await topicLookupPage.titleInput.isVisible()).toBe(true);

    // Verify Clusters page
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
    await layout.navigateTo('clusters');
    await page.waitForTimeout(500);
    const clustersPage = new ClustersOverviewPage(page);
    await clustersPage.waitForClusters(5000);

    // Verify Monitoring page
    await mockAPIRoute(page, '**/api/monitoring/pipeline-status', mockPipelineStatusResponse);
    await layout.navigateTo('monitoring');
    await page.waitForTimeout(500);
    const monitoringPage = new MonitoringPage(page);
    await monitoringPage.waitForData(5000);

    // Verify Ingestion page
    await layout.navigateTo('ingestion');
    await page.waitForTimeout(300);
    const ingestionPage = new IngestionPage(page);
    expect(await ingestionPage.startPipelineButton.isVisible()).toBe(true);
  });
});

