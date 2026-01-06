import { test, expect } from '@playwright/test';
import { ClustersOverviewPage } from '../pages/ClustersOverviewPage';
import { Layout } from '../pages/Layout';
import { waitForAPIRequest } from '../utils/wait-helpers';
import {
  mockClustersOverviewResponse,
  mockClusterSummaryResponse,
  mockGraphVisualizationResponse,
  mockAPIRoute,
} from '../fixtures/api-fixtures';

test.describe('Clusters Workflow E2E', () => {
  test('should complete full clusters exploration workflow', async ({ page }) => {
    const layout = new Layout(page);
    const clustersPage = new ClustersOverviewPage(page);

    // Navigate to clusters page
    await layout.navigateTo('clusters');
    await expect(layout.verifyPageActive('clusters')).resolves.toBe(true);

    // Mock API responses
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
    await mockAPIRoute(page, '**/api/clusters/0', mockClusterSummaryResponse);

    // Verify clusters overview API is called on load
    const overviewRequestPromise = waitForAPIRequest(page, '/api/clusters/overview', 'GET');
    await clustersPage.waitForClusters();
    const overviewRequest = await overviewRequestPromise;
    expect(overviewRequest.url()).toContain('/api/clusters/overview');

    // Select a cluster
    const detailsRequestPromise = waitForAPIRequest(page, '/api/clusters/0', 'GET');
    await clustersPage.selectCluster(0);
    const detailsRequest = await detailsRequestPromise;
    expect(detailsRequest.url()).toContain('/api/clusters/0');

    // Verify cluster details are displayed
    const details = await clustersPage.getClusterDetails();
    expect(details).toBeTruthy();
    expect(details?.clusterId).toBe(0);
  });

  test('should switch to graph tab and load visualization', async ({ page }) => {
    const clustersPage = new ClustersOverviewPage(page);
    
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
    await mockAPIRoute(page, '**/api/clusters/0', mockClusterSummaryResponse);
    await mockAPIRoute(page, '**/api/graph/visualization/0', mockGraphVisualizationResponse);

    await clustersPage.loadClusters();
    await clustersPage.selectCluster(0);

    // Switch to graph tab
    const graphRequestPromise = waitForAPIRequest(page, '/api/graph/visualization/0', 'GET');
    await clustersPage.switchToGraphTab();
    const graphRequest = await graphRequestPromise;
    expect(graphRequest.url()).toContain('/api/graph/visualization/0');

    // Verify graph is displayed
    const isGraphVisible = await clustersPage.isGraphVisible();
    expect(isGraphVisible).toBe(true);
  });

  test('should filter and sort clusters', async ({ page }) => {
    const clustersPage = new ClustersOverviewPage(page);
    
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);

    await clustersPage.loadClusters();
    await clustersPage.waitForClusters();

    // Test search/filter
    await clustersPage.searchClusters('machine');
    
    // Get cluster IDs
    const clusterIds = await clustersPage.getClusterIds();
    expect(clusterIds.length).toBeGreaterThan(0);
  });

  test('should navigate between table and graph views', async ({ page }) => {
    const clustersPage = new ClustersOverviewPage(page);
    
    await mockAPIRoute(page, '**/api/clusters/overview', mockClustersOverviewResponse);
    await mockAPIRoute(page, '**/api/clusters/0', mockClusterSummaryResponse);
    await mockAPIRoute(page, '**/api/graph/visualization/0', mockGraphVisualizationResponse);

    await clustersPage.loadClusters();
    await clustersPage.selectCluster(0);

    // Switch to graph
    await clustersPage.switchToGraphTab();
    expect(await clustersPage.isGraphVisible()).toBe(true);

    // Switch back to table
    await clustersPage.switchToTableTab();
    const details = await clustersPage.getClusterDetails();
    expect(details).toBeTruthy();
  });
});

