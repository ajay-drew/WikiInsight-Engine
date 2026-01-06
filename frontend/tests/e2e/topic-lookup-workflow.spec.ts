import { test, expect } from '@playwright/test';
import { TopicLookupPage } from '../pages/TopicLookupPage';
import { Layout } from '../pages/Layout';
import { waitForAPIRequest } from '../utils/wait-helpers';
import { mockTopicLookupResponse, mockAPIRoute } from '../fixtures/api-fixtures';

test.describe('Topic Lookup Workflow E2E', () => {
  test('should complete full topic lookup workflow', async ({ page }) => {
    const layout = new Layout(page);
    const topicLookupPage = new TopicLookupPage(page);

    // Navigate to topic lookup page
    await layout.navigateTo('lookup');
    await expect(layout.verifyPageActive('lookup')).resolves.toBe(true);

    // Mock API response
    await mockAPIRoute(page, '**/api/topics/lookup', mockTopicLookupResponse);

    // Perform lookup
    const requestPromise = waitForAPIRequest(page, '/api/topics/lookup', 'POST');
    await topicLookupPage.lookupTopic('Machine learning');
    
    // Verify API request was sent
    const request = await requestPromise;
    expect(request.url()).toContain('/api/topics/lookup');

    // Verify result is displayed
    await topicLookupPage.waitForResult();
    expect(await topicLookupPage.hasResult()).toBe(true);
    
    const result = await topicLookupPage.getResult();
    expect(result).toBeTruthy();
    expect(result?.articleTitle).toContain('Machine learning');
    expect(result?.clusterId).toBe(0);
  });

  test('should lookup different article titles', async ({ page }) => {
    const topicLookupPage = new TopicLookupPage(page);
    await topicLookupPage.goto();

    const articles = ['Machine learning', 'Deep learning', 'Neural network'];
    
    for (const article of articles) {
      await mockAPIRoute(page, '**/api/topics/lookup', {
        ...mockTopicLookupResponse,
        article_title: article,
      });

      await topicLookupPage.lookupTopic(article);
      await topicLookupPage.waitForResult();
      
      const result = await topicLookupPage.getResult();
      expect(result?.articleTitle).toContain(article);
    }
  });

  test('should navigate to topic lookup and perform lookup', async ({ page }) => {
    const layout = new Layout(page);
    const topicLookupPage = new TopicLookupPage(page);

    // Start from a different page
    await layout.navigateTo('search');
    
    // Navigate to topic lookup
    await layout.navigateTo('lookup');
    
    // Perform lookup
    await mockAPIRoute(page, '**/api/topics/lookup', mockTopicLookupResponse);
    await topicLookupPage.lookupTopic('Machine learning');
    await topicLookupPage.waitForResult();
    
    expect(await topicLookupPage.hasResult()).toBe(true);
  });
});

