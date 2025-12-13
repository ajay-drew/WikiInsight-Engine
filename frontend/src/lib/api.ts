export type TopicLookupResponse = {
  article_title: string;
  cluster_id: number | null;
  similar_articles: string[];
  keywords: string[] | null;
  explanation: Record<string, unknown> | null;
};

export type ClusterSummary = {
  cluster_id: number;
  size: number;
  keywords: string[];
  top_articles: string[];
};

const API_BASE = "/api";

export async function lookupTopic(articleTitle: string): Promise<TopicLookupResponse> {
  const resp = await fetch(`${API_BASE}/topics/lookup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ article_title: articleTitle })
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Lookup failed (${resp.status}): ${text || resp.statusText}`);
  }

  return (await resp.json()) as TopicLookupResponse;
}

export async function fetchClustersOverview(): Promise<ClusterSummary[]> {
  const resp = await fetch(`${API_BASE}/clusters/overview`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load clusters overview (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as ClusterSummary[];
}

export async function fetchClusterSummary(clusterId: number): Promise<ClusterSummary> {
  const resp = await fetch(`${API_BASE}/clusters/${clusterId}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load cluster ${clusterId} (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as ClusterSummary;
}

export type SearchResult = {
  title: string;
  score: number;
  rank: number;
};

export type SearchResponse = {
  query: string;
  results: SearchResult[];
  total_results: number;
};

export async function searchArticles(query: string, topK: number = 10): Promise<SearchResponse> {
  const resp = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, top_k: topK })
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Search failed (${resp.status}): ${text || resp.statusText}`);
  }

  return (await resp.json()) as SearchResponse;
}

// Monitoring API types and functions
export type PipelineStatus = {
  ingestion: {
    has_artifacts: boolean;
    artifact_size?: number;
    last_modified?: string;
  };
  preprocessing: {
    has_artifacts: boolean;
    last_modified?: string;
  };
  clustering: {
    has_artifacts: boolean;
    last_modified?: string;
  };
};

export type APIMetricsSummary = {
  total_requests: number;
  window_seconds?: number;
  endpoints: Record<string, {
    count: number;
    avg_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    error_rate: number;
    status_codes: Record<string, number>;
  }>;
};

export type DriftReport = {
  drift_detected: boolean;
  threshold?: number;
  [key: string]: unknown;
};

export type StabilityMetrics = {
  stability_calculated: boolean;
  ari?: number;
  nmi?: number;
  jaccard?: number;
  [key: string]: unknown;
};

export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const resp = await fetch(`${API_BASE}/monitoring/pipeline-status`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load pipeline status (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as PipelineStatus;
}

export async function fetchMetrics(windowSeconds?: number): Promise<APIMetricsSummary> {
  const url = windowSeconds
    ? `${API_BASE}/monitoring/metrics?window_seconds=${windowSeconds}`
    : `${API_BASE}/monitoring/metrics`;
  const resp = await fetch(url);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load metrics (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as APIMetricsSummary;
}

export async function fetchDriftScores(): Promise<DriftReport> {
  const resp = await fetch(`${API_BASE}/monitoring/drift`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load drift scores (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as DriftReport;
}

export async function fetchClusterStability(): Promise<StabilityMetrics> {
  const resp = await fetch(`${API_BASE}/monitoring/stability`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load cluster stability (${resp.status}): ${text || resp.statusText}`);
  }
  return (await resp.json()) as StabilityMetrics;
}


