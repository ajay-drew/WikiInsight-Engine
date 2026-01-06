import { test, expect } from '@playwright/test';
import { TopicLookupPage } from '../pages/TopicLookupPage';
import { waitForAPIRequest, waitForAPIResponse } from '../utils/wait-helpers';
import { verifyRequest } from '../utils/request-verification';
import { mockTopicLookupResponse, mockAPIRoute, mockAPIErrorRoute } from '../fixtures/api-fixtures';

test.describe('Topic Lookup API Integration', () => {
  let topicLookupPage: TopicLookupPage;

  test.beforeEach(async ({ page }) => {
    topicLookupPage = new TopicLookupPage(page);
    await topicLookupPage.goto();
  });

  test('should send POST request to /api/topics/lookup with correct body', async ({ page }) => {
    const articleTitle = 'Machine learning';

    const requestPromise = waitForAPIRequest(page, '/api/topics/lookup', 'POST');
    
    await topicLookupPage.lookupTopic(articleTitle);
    
    const request = await requestPromise;
    
    verifyRequest(request, {
      url: '/api/topics/lookup',
      method: 'POST',
      body: { article_title: articleTitle },
      headers: { 'content-type': 'application/json' },
    });
  });

  test('should handle successful lookup response', async ({ page }) => {
    await mockAPIRoute(page, '**/api/topics/lookup', mockTopicLookupResponse);

    await topicLookupPage.lookupTopic('Machine learning');
    await topicLookupPage.waitForResult();

    const result = await topicLookupPage.getResult();
    expect(result).toBeTruthy();
    expect(result?.articleTitle).toContain('Machine learning');
    expect(result?.clusterId).toBe(0);
  });

  test('should display cluster information correctly', async ({ page }) => {
    await mockAPIRoute(page, '**/api/topics/lookup', mockTopicLookupResponse);

    await topicLookupPage.lookupTopic('Machine learning');
    await topicLookupPage.waitForResult();

    const result = await topicLookupPage.getResult();
    expect(result?.keywords.length).toBeGreaterThan(0);
    expect(result?.similarArticles.length).toBeGreaterThan(0);
  });

  test('should handle empty title validation', async ({ page }) => {
    await topicLookupPage.enterTitle('');
    await topicLookupPage.analyzeButton.click();

    const error = await topicLookupPage.getError();
    expect(error).toContain('Please enter an article title');
  });

  test('should handle non-existent article error', async ({ page }) => {
    await mockAPIErrorRoute(page, '**/api/topics/lookup', 'Article not found', 404);

    await topicLookupPage.lookupTopic('NonExistentArticle12345');

    const error = await topicLookupPage.getError();
    expect(error).toBeTruthy();
    expect(error).toContain('Article not found');
  });

  test('should handle 500 error', async ({ page }) => {
    await mockAPIErrorRoute(page, '**/api/topics/lookup', 'Internal server error', 500);

    await topicLookupPage.lookupTopic('Machine learning');

    const error = await topicLookupPage.getError();
    expect(error).toBeTruthy();
  });

  test('should show loading indicator during lookup', async ({ page }) => {
    await page.route('**/api/topics/lookup', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockTopicLookupResponse),
      });
    });

    const lookupPromise = topicLookupPage.lookupTopic('Machine learning');
    
    // Check loading indicator immediately after clicking (before response)
    await page.waitForTimeout(100);
    const isLoading = await topicLookupPage.isLoading();
    
    // Loading indicator might be very brief, so just verify it was visible or button shows loading state
    const buttonText = await topicLookupPage.analyzeButton.textContent();
    const isButtonLoading = buttonText?.toLowerCase().includes('looking') || buttonText?.toLowerCase().includes('analyzing');
    
    // Either loading indicator visible or button shows loading state
    expect(isLoading || isButtonLoading).toBe(true);
    
    await lookupPromise;
  });

  test('should handle network errors', async ({ page }) => {
    await page.route('**/api/topics/lookup', route => route.abort());

    await topicLookupPage.lookupTopic('Machine learning');

    const error = await topicLookupPage.getError();
    expect(error).toBeTruthy();
  });

  test('should verify request headers include Content-Type', async ({ page }) => {
    const requestPromise = waitForAPIRequest(page, '/api/topics/lookup', 'POST');
    
    await topicLookupPage.lookupTopic('Machine learning');
    
    const request = await requestPromise;
    const headers = request.headers();
    
    expect(headers['content-type']).toContain('application/json');
  });
});

