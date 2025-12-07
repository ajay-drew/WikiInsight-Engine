import React, { useEffect, useState } from "react";
import { ClusterSummary, fetchClustersOverview, fetchClusterSummary } from "../lib/api";

export function ClustersOverviewPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadOverview();
  }, []);

  async function loadOverview() {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchClustersOverview();
      setClusters(data);
      if (data.length > 0) {
        const firstId = data[0].cluster_id;
        const full = await fetchClusterSummary(firstId);
        setSelectedCluster(full);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load clusters overview.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectCluster(id: number) {
    setError(null);
    try {
      const summary = await fetchClusterSummary(id);
      setSelectedCluster(summary);
    } catch (err: any) {
      setError(err.message || `Failed to load cluster ${id}.`);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Clusters Overview</h1>
        <p className="text-sm text-slate-400">
          Inspect topic clusters, their sizes, and representative articles. For each cluster we
          also show “topic words” – everyday words that appear often in this cluster and much less
          in others, giving a quick feel for what the cluster is about.
        </p>
      </section>

      {loading && <p className="text-sm text-slate-300">Loading clusters…</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {!loading && clusters.length === 0 && !error && (
        <p className="text-sm text-slate-400">
          No clusters available. Make sure you have run the DVC pipeline / clustering step.
        </p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 border border-slate-800 rounded-lg bg-slate-900/70 overflow-hidden">
          <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
            <span className="text-sm font-medium">All Clusters</span>
            <span className="text-xs text-slate-400">
              Total: {clusters.length.toLocaleString()}
            </span>
          </div>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-900/90 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left font-medium text-slate-400">ID</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-400">Size</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-400">Topic words</th>
                  <th className="px-3 py-2 text-left font-medium text-slate-400"></th>
                </tr>
              </thead>
              <tbody>
                {clusters.map((c) => (
                  <tr
                    key={c.cluster_id}
                    className="border-t border-slate-800 hover:bg-slate-800/60"
                  >
                    <td className="px-3 py-1.5 font-mono">{c.cluster_id}</td>
                    <td className="px-3 py-1.5">{c.size}</td>
                    <td className="px-3 py-1.5 text-slate-300">
                      {c.keywords && c.keywords.length > 0
                        ? c.keywords.slice(0, 8).join(", ")
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      <button
                        onClick={() => handleSelectCluster(c.cluster_id)}
                        className="px-2 py-1 rounded bg-slate-800 text-slate-100 hover:bg-slate-700 text-[11px]"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="border border-slate-800 rounded-lg bg-slate-900/70 p-4">
          <h2 className="text-sm font-medium mb-2">Cluster Details</h2>
          {selectedCluster ? (
            <div className="space-y-3 text-xs">
              <p>
                <span className="text-slate-400">Cluster ID:</span>{" "}
                <span className="font-mono text-sky-400">
                  {selectedCluster.cluster_id}
                </span>
              </p>
              <p>
                <span className="text-slate-400">Size:</span>{" "}
                {selectedCluster.size.toLocaleString()}
              </p>
              <div>
                <p className="text-slate-400 mb-1">
                  Topic words (high-frequency and distinctive for this cluster):
                </p>
                <p className="text-slate-200">
                  {selectedCluster.keywords.slice(0, 20).join(", ")}
                </p>
              </div>
              <div>
                <p className="text-slate-400 mb-1">Representative Articles:</p>
                <ul className="list-disc list-inside space-y-0.5 text-slate-200">
                  {selectedCluster.top_articles.slice(0, 10).map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Select a cluster to inspect details.</p>
          )}
        </div>
      </div>
    </div>
  );
}


