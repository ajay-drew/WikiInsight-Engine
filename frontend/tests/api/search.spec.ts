import { test, expect } from '@playwright/test';
import { SearchPage } from '../pages/SearchPage';
import { waitForAPIRequest, waitForAPIResponse } from '../utils/wait-helpers';
import { verifyRequest } from '../utils/request-verification';
import { mockSearchResponse, mockAPIRoute, mockAPIErrorRoute } from '../fixtures/api-fixtures';

test.describe('Search API Integration', () => {
  let searchPage: SearchPage;

  test.beforeEach(async ({ page }) => {
    searchPage = new SearchPage(page);
    await searchPage.goto();
  });

  test('should send POST request to /api/search with correct body', async ({ page }) => {
    const query = 'machine learning';
    const topK = 20;

    // Wait for and verify the API request
    const requestPromise = waitForAPIRequest(page, '/api/search', 'POST');
    
    await searchPage.search(query);
    
    const request = await requestPromise;
    
    // Verify request details
    verifyRequest(request, {
      url: '/api/search',
      method: 'POST',
      body: { query, top_k: topK },
      headers: { 'content-type': 'application/json' },
    });
  });

  test('should handle successful search response', async ({ page }) => {
    // Mock successful response
    await mockAPIRoute(page, '**/api/search', mockSearchResponse);

    await searchPage.search('machine learning');
    await searchPage.waitForResults();

    const results = await searchPage.getResults();
    expect(results.length).toBeGreaterThan(0);
    expect(await searchPage.hasResults()).toBe(true);
  });

  test('should display search results correctly', async ({ page }) => {
    await mockAPIRoute(page, '**/api/search', mockSearchResponse);

    await searchPage.search('machine learning');
    await searchPage.waitForResults();

    const results = await searchPage.getResults();
    expect(results.length).toBe(2);
    expect(results[0].title).toContain('Machine learning');
  });

  test('should handle empty query validation', async ({ page }) => {
    await searchPage.enterQuery('');
    
    // Button should be disabled when query is empty
    const isDisabled = await searchPage.searchButton.isDisabled();
    expect(isDisabled).toBe(true);
    
    // Try to submit form (should trigger validation)
    await searchPage.searchButton.click({ force: true }).catch(() => {
      // Expected if button is disabled
    });
    
    // Wait for validation error
    await page.waitForTimeout(500);
    
    // Check if error appears (form validation might show error on submit attempt)
    const error = await searchPage.getError();
    // Error might appear or button stays disabled - both are valid
    expect(isDisabled || error !== null).toBe(true);
  });

  test('should handle API error responses', async ({ page }) => {
    await mockAPIErrorRoute(page, '**/api/search', 'Search engine is not available', 503);

    await searchPage.search('machine learning');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
    expect(error).toContain('Search engine');
  });

  test('should handle 404 error', async ({ page }) => {
    await mockAPIErrorRoute(page, '**/api/search', 'Not found', 404);

    await searchPage.search('machine learning');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should handle 500 error', async ({ page }) => {
    await mockAPIErrorRoute(page, '**/api/search', 'Internal server error', 500);

    await searchPage.search('machine learning');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should show loading indicator during search', async ({ page }) => {
    // Delay the response to see loading state
    await page.route('**/api/search', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSearchResponse),
      });
    });

    await searchPage.search('machine learning');
    
    // Check loading indicator appears briefly
    const isLoading = await searchPage.isLoading();
    expect(isLoading).toBe(true);
  });

  test('should handle network errors', async ({ page }) => {
    // Simulate network failure
    await page.route('**/api/search', route => route.abort());

    await searchPage.search('machine learning');

    const error = await searchPage.getError();
    expect(error).toBeTruthy();
  });

  test('should verify request headers include Content-Type', async ({ page }) => {
    const requestPromise = waitForAPIRequest(page, '/api/search', 'POST');
    
    await searchPage.search('machine learning');
    
    const request = await requestPromise;
    const headers = request.headers();
    
    expect(headers['content-type']).toContain('application/json');
  });
});

