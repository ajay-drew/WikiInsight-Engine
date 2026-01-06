import { Page, Route } from '@playwright/test';

/**
 * Mock data generators and fixtures for API responses
 */

// Search API mocks
export const mockSearchResponse = {
  query: 'machine learning',
  results: [
    {
      title: 'Machine learning',
      score: 0.95,
      rank: 1,
      wikipedia_url: 'https://en.wikipedia.org/wiki/Machine_learning',
      wikidata_qid: 'Q2539',
      wikidata_url: 'https://www.wikidata.org/wiki/Q2539',
      cluster_id: 0,
      categories: ['Artificial intelligence', 'Machine learning'],
      link_count: 150,
    },
    {
      title: 'Deep learning',
      score: 0.89,
      rank: 2,
      wikipedia_url: 'https://en.wikipedia.org/wiki/Deep_learning',
      cluster_id: 0,
      categories: ['Machine learning', 'Neural networks'],
      link_count: 120,
    },
  ],
  total_results: 2,
};

export const mockSearchErrorResponse = {
  detail: 'Search engine is not available. Run the data pipeline first.',
};

// Topic Lookup API mocks
export const mockTopicLookupResponse = {
  article_title: 'Machine learning',
  cluster_id: 0,
  similar_articles: ['Deep learning', 'Neural network', 'Artificial intelligence'],
  keywords: ['algorithm', 'data', 'model', 'training', 'neural'],
  explanation: {
    cluster_id: 0,
    keywords: ['algorithm', 'data', 'model'],
  },
};

export const mockTopicLookupErrorResponse = {
  detail: 'Article not found',
};

// Clusters API mocks
export const mockClustersOverviewResponse = [
  {
    cluster_id: 0,
    size: 50,
    keywords: ['machine', 'learning', 'algorithm'],
    top_articles: ['Machine learning', 'Deep learning', 'Neural network'],
  },
  {
    cluster_id: 1,
    size: 45,
    keywords: ['data', 'science', 'analysis'],
    top_articles: ['Data science', 'Statistics', 'Analytics'],
  },
];

export const mockClusterSummaryResponse = {
  cluster_id: 0,
  size: 50,
  keywords: ['machine', 'learning', 'algorithm', 'neural', 'training'],
  top_articles: [
    'Machine learning',
    'Deep learning',
    'Neural network',
    'Artificial intelligence',
    'Supervised learning',
  ],
};

// Graph API mocks
export const mockGraphVisualizationResponse = {
  nodes: [
    { id: 'Machine learning', label: 'Machine learning', cluster_id: 0, x: 100, y: 100 },
    { id: 'Deep learning', label: 'Deep learning', cluster_id: 0, x: 200, y: 150 },
    { id: 'Neural network', label: 'Neural network', cluster_id: 0, x: 150, y: 200 },
  ],
  edges: [
    { source: 'Machine learning', target: 'Deep learning', layer: 2, weight: 0.9, type: 'cluster' },
    { source: 'Deep learning', target: 'Neural network', layer: 3, weight: 0.85, type: 'semantic' },
  ],
};

export const mockGraphNeighborsResponse = {
  article_title: 'Machine learning',
  neighbors: [
    { title: 'Deep learning', layer: 2, type: 'cluster', weight: 0.9 },
    { title: 'Neural network', layer: 3, type: 'semantic', weight: 0.85 },
  ],
};

export const mockGraphPathResponse = {
  from_title: 'Machine learning',
  to_title: 'Data science',
  path: ['Machine learning', 'Artificial intelligence', 'Data science'],
  found: true,
};

// Monitoring API mocks
export const mockPipelineStatusResponse = {
  ingestion: {
    has_artifacts: true,
    artifact_size: 1024000,
    last_modified: '2025-01-15T10:00:00Z',
  },
  preprocessing: {
    has_artifacts: true,
    last_modified: '2025-01-15T10:05:00Z',
  },
  clustering: {
    has_artifacts: true,
    last_modified: '2025-01-15T10:10:00Z',
  },
};

export const mockMetricsResponse = {
  total_requests: 150,
  window_seconds: 3600,
  endpoints: {
    '/api/search': {
      count: 50,
      avg_latency_ms: 120,
      p50_latency_ms: 100,
      p95_latency_ms: 200,
      p99_latency_ms: 300,
      error_rate: 0.02,
      status_codes: { '200': 49, '500': 1 },
    },
  },
};

export const mockDriftReportResponse = {
  drift_detected: false,
  threshold: 0.1,
};

export const mockStabilityMetricsResponse = {
  stability_calculated: true,
  ari: 0.85,
  nmi: 0.82,
  jaccard: 0.78,
};

// Pipeline API mocks
export const mockPipelineStartResponse = {
  status: 'started',
  message: 'Pipeline started successfully',
  config: {
    seed_queries: ['Machine learning', 'Data science'],
    per_query_limit: 50,
    max_articles: 1000,
  },
};

export const mockPipelineProgressResponse = {
  current_stage: 'ingestion',
  stages: {
    ingestion: {
      status: 'running',
      progress: 45,
      message: 'Fetching articles...',
      eta: 120,
    },
    preprocessing: {
      status: 'pending',
      progress: 0,
      message: 'Waiting...',
      eta: null,
    },
    clustering: {
      status: 'pending',
      progress: 0,
      message: 'Waiting...',
      eta: null,
    },
    build_graph: {
      status: 'pending',
      progress: 0,
      message: 'Waiting...',
      eta: null,
    },
  },
  started_at: '2025-01-15T10:00:00Z',
  updated_at: '2025-01-15T10:01:00Z',
  overall_progress: 11.25,
};

/**
 * Mock API route handler
 */
export async function mockAPIRoute(
  page: Page,
  urlPattern: string | RegExp,
  response: any,
  status: number = 200
): Promise<void> {
  await page.route(urlPattern, async (route: Route) => {
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

/**
 * Mock API error route
 */
export async function mockAPIErrorRoute(
  page: Page,
  urlPattern: string | RegExp,
  message: string,
  status: number = 500
): Promise<void> {
  await page.route(urlPattern, async (route: Route) => {
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify({ detail: message }),
    });
  });
}

/**
 * Mock SSE stream for pipeline progress
 */
export async function mockSSERoute(
  page: Page,
  urlPattern: string | RegExp,
  messages: Array<{ data: any; event?: string }>
): Promise<void> {
  await page.route(urlPattern, async (route: Route) => {
    // For SSE, we need to send the first message immediately
    // and keep the connection open
    let messageIndex = 0;
    
    const stream = new ReadableStream({
      start(controller) {
        // Send first message immediately
        if (messages.length > 0) {
          const msg = messages[0];
          const event = msg.event ? `event: ${msg.event}\n` : '';
          const data = `data: ${JSON.stringify(msg.data)}\n\n`;
          controller.enqueue(new TextEncoder().encode(event + data));
        }
        
        // Send remaining messages with delay
        for (let i = 1; i < messages.length; i++) {
          setTimeout(() => {
            const msg = messages[i];
            const event = msg.event ? `event: ${msg.event}\n` : '';
            const data = `data: ${JSON.stringify(msg.data)}\n\n`;
            controller.enqueue(new TextEncoder().encode(event + data));
            
            if (i === messages.length - 1) {
              // Keep connection open for SSE
              // Don't close immediately
            }
          }, i * 200);
        }
      },
    });
    
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
      body: stream,
    });
  });
}

/**
 * Create mock search response with custom query
 */
export function createMockSearchResponse(query: string, results: any[] = mockSearchResponse.results) {
  return {
    ...mockSearchResponse,
    query,
    results,
    total_results: results.length,
  };
}

/**
 * Create mock cluster response with custom cluster ID
 */
export function createMockClusterResponse(clusterId: number) {
  return {
    ...mockClusterSummaryResponse,
    cluster_id: clusterId,
  };
}

/**
 * Create mock pipeline progress response
 */
export function createMockPipelineProgress(
  currentStage: string,
  stageProgress: number,
  overallProgress: number
) {
  return {
    ...mockPipelineProgressResponse,
    current_stage: currentStage,
    stages: {
      ...mockPipelineProgressResponse.stages,
      [currentStage]: {
        ...mockPipelineProgressResponse.stages[currentStage as keyof typeof mockPipelineProgressResponse.stages],
        progress: stageProgress,
      },
    },
    overall_progress: overallProgress,
  };
}

