import { test, expect } from '@playwright/test';
import { SearchPage } from '../pages/SearchPage';
import { Layout } from '../pages/Layout';
import { waitForAPIRequest } from '../utils/wait-helpers';
import { mockSearchResponse, mockAPIRoute } from '../fixtures/api-fixtures';

test.describe('Search Workflow E2E', () => {
  test('should complete full search workflow', async ({ page }) => {
    const layout = new Layout(page);
    const searchPage = new SearchPage(page);

    // Navigate to search page
    await layout.navigateTo('search');
    await expect(layout.verifyPageActive('search')).resolves.toBe(true);

    // Mock API response
    await mockAPIRoute(page, '**/api/search', mockSearchResponse);

    // Perform search
    const requestPromise = waitForAPIRequest(page, '/api/search', 'POST');
    await searchPage.search('machine learning');
    
    // Verify API request was sent
    const request = await requestPromise;
    expect(request.url()).toContain('/api/search');

    // Verify results are displayed
    await searchPage.waitForResults();
    expect(await searchPage.hasResults()).toBe(true);
    
    const results = await searchPage.getResults();
    expect(results.length).toBeGreaterThan(0);
  });

  test('should perform multiple searches', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    const queries = ['machine learning', 'data science', 'artificial intelligence'];
    
    for (const query of queries) {
      await mockAPIRoute(page, '**/api/search', {
        ...mockSearchResponse,
        query,
        results: [{ ...mockSearchResponse.results[0], title: query }],
      });

      await searchPage.search(query);
      await searchPage.waitForResults();
      
      expect(await searchPage.hasResults()).toBe(true);
    }
  });

  test('should navigate to search page and perform search', async ({ page }) => {
    const layout = new Layout(page);
    const searchPage = new SearchPage(page);

    // Start from a different page
    await layout.navigateTo('monitoring');
    
    // Navigate to search
    await layout.navigateTo('search');
    
    // Perform search
    await mockAPIRoute(page, '**/api/search', mockSearchResponse);
    await searchPage.search('machine learning');
    await searchPage.waitForResults();
    
    expect(await searchPage.hasResults()).toBe(true);
  });
});

