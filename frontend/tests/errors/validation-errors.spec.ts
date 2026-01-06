import { test, expect } from '@playwright/test';
import { SearchPage } from '../pages/SearchPage';
import { TopicLookupPage } from '../pages/TopicLookupPage';
import { IngestionPage } from '../pages/IngestionPage';

test.describe('Form Validation Errors', () => {
  test('should validate empty search query', async ({ page }) => {
    const searchPage = new SearchPage(page);
    await searchPage.goto();

    // Enter empty query
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

  test('should validate empty topic lookup title', async ({ page }) => {
    const topicLookupPage = new TopicLookupPage(page);
    await topicLookupPage.goto();

    // Try to submit empty title
    await topicLookupPage.enterTitle('');
    await topicLookupPage.analyzeButton.click();
    
    // Wait for error to appear
    await page.waitForTimeout(500);

    const error = await topicLookupPage.getError();
    expect(error).toBeTruthy();
    expect(error?.toLowerCase()).toContain('enter an article title');
  });

  test('should validate pipeline query count (minimum 3)', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Try with only 2 queries
    await ingestionPage.configurePipeline({
      seedQueries: ['Query 1', 'Query 2'],
    });
    
    // Wait for validation to trigger
    await page.waitForTimeout(500);
    
    // Check if button is disabled (validation prevents start)
    const isDisabled = await ingestionPage.startPipelineButton.isDisabled();
    const error = await ingestionPage.getError();
    
    // Either button should be disabled OR error message should appear
    expect(isDisabled || (error !== null && error.toLowerCase().includes('3-6'))).toBe(true);
  });

  test('should validate pipeline query count (maximum 6)', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Try with 6 queries (max allowed)
    await ingestionPage.configurePipeline({
      seedQueries: ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'],
    });
    
    await page.waitForTimeout(500);

    // Try to add 7th query - button should be disabled or clicking should not add more
    const addButton = ingestionPage.addQueryButton;
    const initialInputCount = (await ingestionPage.queryInputs.all()).length;
    
    try {
      await addButton.click({ timeout: 1000 });
    } catch {
      // Button might be disabled
    }
    
    await page.waitForTimeout(500);
    
    const finalInputCount = (await ingestionPage.queryInputs.all()).length;
    
    // Should not have more than 6 inputs
    expect(finalInputCount).toBeLessThanOrEqual(6);
  });

  test('should validate per_query_limit range (1-70)', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Set valid queries first
    await ingestionPage.configurePipeline({
      seedQueries: ['Query 1', 'Query 2', 'Query 3'],
    });

    // Try to set per_query_limit to 0 (invalid)
    // The slider should prevent this, but we can test the UI state
    const slider = ingestionPage.perQueryLimitSlider;
    const min = await slider.getAttribute('min');
    const max = await slider.getAttribute('max');
    
    expect(min).toBe('1');
    expect(max).toBe('70');
  });

  test('should validate max_articles range', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    await ingestionPage.configurePipeline({
      seedQueries: ['Query 1', 'Query 2', 'Query 3'],
    });

    const slider = ingestionPage.maxArticlesSlider;
    const min = await slider.getAttribute('min');
    const max = await slider.getAttribute('max');
    
    expect(min).toBe('10');
    expect(max).toBe('1000');
  });

  test('should validate all queries are non-empty', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Set queries with one empty
    const inputs = await ingestionPage.queryInputs.all();
    if (inputs.length >= 3) {
      await inputs[0].fill('Query 1');
      await inputs[1].fill(''); // Empty query
      await inputs[2].fill('Query 3');
    }
    
    // Wait for validation
    await page.waitForTimeout(500);

    // Try to start pipeline - button should be disabled
    const isDisabled = await ingestionPage.startPipelineButton.isDisabled();
    const error = await ingestionPage.getError();
    
    // Button should be disabled OR error should appear
    expect(isDisabled || error !== null).toBe(true);
  });

  test('should show validation message for invalid configuration', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Configure with invalid settings
    await ingestionPage.configurePipeline({
      seedQueries: ['Q1', 'Q2'], // Too few
    });
    
    await page.waitForTimeout(500);
    
    // Try to click button to trigger validation
    try {
      await ingestionPage.startPipelineButton.click({ timeout: 1000, force: true });
    } catch {
      // Expected if disabled
    }
    
    // Wait for validation
    await page.waitForTimeout(1000);

    // Validation error should be shown OR button disabled
    const error = await ingestionPage.getError();
    const isDisabled = await ingestionPage.startPipelineButton.isDisabled();
    expect(isDisabled || error !== null).toBe(true);
  });

  test('should prevent pipeline start with invalid configuration', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    await ingestionPage.configurePipeline({
      seedQueries: ['Q1', 'Q2'], // Invalid: too few queries
    });
    
    // Wait a bit for validation
    await page.waitForTimeout(500);
    
    // Try to click button to trigger validation
    try {
      await ingestionPage.startPipelineButton.click({ timeout: 1000, force: true });
    } catch {
      // Expected if button is disabled
    }
    
    await page.waitForTimeout(500);

    // Start button should be disabled OR error should appear
    const isDisabled = await ingestionPage.startPipelineButton.isDisabled();
    const error = await ingestionPage.getError();
    expect(isDisabled || error !== null).toBe(true);
  });
});

