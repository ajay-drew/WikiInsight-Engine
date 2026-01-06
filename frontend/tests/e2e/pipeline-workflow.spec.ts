import { test, expect } from '@playwright/test';
import { IngestionPage } from '../pages/IngestionPage';
import { Layout } from '../pages/Layout';
import { waitForAPIRequest, waitForSSEConnection } from '../utils/wait-helpers';
import {
  mockPipelineStartResponse,
  mockAPIRoute,
  mockSSERoute,
  createMockPipelineProgress,
} from '../fixtures/api-fixtures';

test.describe('Pipeline Workflow E2E', () => {
  test('should complete full pipeline execution workflow', async ({ page }) => {
    const layout = new Layout(page);
    const ingestionPage = new IngestionPage(page);

    // Navigate to ingestion page
    await layout.navigateTo('ingestion');
    await expect(layout.verifyPageActive('ingestion')).resolves.toBe(true);

    // Mock API responses
    await mockAPIRoute(page, '**/api/pipeline/start', mockPipelineStartResponse);
    
    const progressMessages = [
      { data: createMockPipelineProgress('ingestion', 50, 12.5) },
      { data: createMockPipelineProgress('preprocessing', 75, 25) },
      { data: createMockPipelineProgress('clustering', 100, 75) },
      { data: createMockPipelineProgress('build_graph', 100, 100) },
    ];
    await mockSSERoute(page, '**/api/pipeline/progress', progressMessages);

    // Configure pipeline
    await ingestionPage.configurePipeline({
      seedQueries: ['Machine learning', 'Data science', 'Artificial intelligence'],
      perQueryLimit: 50,
      maxArticles: 1000,
    });

    // Start pipeline
    const startRequestPromise = waitForAPIRequest(page, '/api/pipeline/start', 'POST');
    const ssePromise = waitForSSEConnection(page, '/api/pipeline/progress');
    
    await ingestionPage.startPipeline();
    
    // Verify API request was sent
    const startRequest = await startRequestPromise;
    expect(startRequest.url()).toContain('/api/pipeline/start');

    // Verify SSE connection is established
    const sseResponse = await ssePromise;
    expect(sseResponse.status()).toBe(200);

    // Monitor progress updates
    await ingestionPage.waitForProgress();
    expect(await ingestionPage.hasProgress()).toBe(true);

    // Verify progress UI updates
    const progress = await ingestionPage.getProgress();
    expect(progress).toBeTruthy();
    expect(progress?.overallProgress).toBeGreaterThan(0);
  });

  test('should handle pipeline configuration validation', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    // Test with invalid configuration (too few queries)
    await ingestionPage.configurePipeline({
      seedQueries: ['Query 1', 'Query 2'], // Only 2 queries, need 3-6
    });

    const error = await ingestionPage.getError();
    expect(error).toBeTruthy();
    expect(error).toContain('3-6');
  });

  test('should update progress through all stages', async ({ page }) => {
    const ingestionPage = new IngestionPage(page);
    await ingestionPage.goto();

    await mockAPIRoute(page, '**/api/pipeline/start', mockPipelineStartResponse);
    
    const stages = [
      { stage: 'ingestion', progress: 100, overall: 25 },
      { stage: 'preprocessing', progress: 100, overall: 50 },
      { stage: 'clustering', progress: 100, overall: 75 },
      { stage: 'build_graph', progress: 100, overall: 100 },
    ];
    
    const progressMessages = stages.map(s => ({
      data: createMockPipelineProgress(s.stage, s.progress, s.overall),
    }));
    
    await mockSSERoute(page, '**/api/pipeline/progress', progressMessages);

    await ingestionPage.configurePipeline({
      seedQueries: ['Machine learning', 'Data science', 'AI'],
    });
    await ingestionPage.startPipeline();
    
    await ingestionPage.waitForProgress();
    
    // Verify progress updates
    const progress = await ingestionPage.getProgress();
    expect(progress).toBeTruthy();
    expect(progress?.overallProgress).toBeGreaterThan(0);
  });
});

