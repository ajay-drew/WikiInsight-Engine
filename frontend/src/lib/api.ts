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


