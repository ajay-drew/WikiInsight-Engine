import React, { useEffect, useState } from "react";
import {
  APIMetricsSummary,
  DriftReport,
  fetchClusterStability,
  fetchDriftScores,
  fetchMetrics,
  fetchPipelineStatus,
  PipelineStatus,
  StabilityMetrics,
} from "../lib/api";

export function MonitoringPage() {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [metrics, setMetrics] = useState<APIMetricsSummary | null>(null);
  const [driftReport, setDriftReport] = useState<DriftReport | null>(null);
  const [stability, setStability] = useState<StabilityMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadAllData();
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      void loadAllData();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  async function loadAllData() {
    setLoading(true);
    setError(null);
    try {
      const [status, apiMetrics, drift, stabilityData] = await Promise.all([
        fetchPipelineStatus(),
        fetchMetrics(3600), // Last hour
        fetchDriftScores().catch(() => null), // Optional
        fetchClusterStability().catch(() => null), // Optional
      ]);
      setPipelineStatus(status);
      setMetrics(apiMetrics);
      setDriftReport(drift);
      setStability(stabilityData);
    } catch (err: any) {
      setError(err.message || "Failed to load monitoring data.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Monitoring Dashboard</h1>
        <p className="text-sm text-slate-400">
          Real-time monitoring of pipeline status, data drift, cluster stability, and API performance.
        </p>
      </section>

      {/* MLOps Tools */}
      <section>
        <h2 className="text-lg font-semibold mb-3">MLOps Tools</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="border border-slate-700 hover:border-sky-500 rounded-lg p-4 bg-slate-900/70 hover:bg-slate-800/70 transition-colors group"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">MLflow UI</h3>
              <span className="text-sky-400 group-hover:text-sky-300">→</span>
            </div>
            <p className="text-xs text-slate-400">
              View experiment tracking, parameters, metrics, and model artifacts
            </p>
            <p className="text-xs text-slate-500 mt-2 font-mono">http://localhost:5000</p>
          </a>
          
          <div className="border border-slate-700 rounded-lg p-4 bg-slate-900/70">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium">DVC Pipeline</h3>
              <span className="text-slate-500">CLI</span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Data versioning and pipeline management
            </p>
            <div className="space-y-1 text-xs font-mono bg-slate-950 p-2 rounded">
              <div className="text-slate-400">$ dvc dag</div>
              <div className="text-slate-400">$ dvc metrics show</div>
              <div className="text-slate-400">$ dvc repro</div>
            </div>
          </div>
        </div>
      </section>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          <span>Loading monitoring data...</span>
        </div>
      )}

      {error && <p className="text-sm text-red-400">{error}</p>}

      {/* Pipeline Status */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Pipeline Status</h2>
        {pipelineStatus ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(pipelineStatus).map(([stage, info]) => (
              <div
                key={stage}
                className={`border rounded-lg p-4 ${
                  info.has_artifacts
                    ? "border-green-500/50 bg-green-500/10"
                    : "border-red-500/50 bg-red-500/10"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium capitalize">{stage}</h3>
                  {info.has_artifacts ? (
                    <span className="text-green-400 text-xs">✓ Ready</span>
                  ) : (
                    <span className="text-red-400 text-xs">✗ Missing</span>
                  )}
                </div>
                {info.last_modified && (
                  <p className="text-xs text-slate-400">
                    Last modified: {new Date(info.last_modified).toLocaleString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400">No pipeline status available.</p>
        )}
      </section>

      {/* Drift Detection */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Data Drift Detection</h2>
        {driftReport ? (
          <div
            className={`border rounded-lg p-4 ${
              driftReport.drift_detected
                ? "border-yellow-500/50 bg-yellow-500/10"
                : "border-green-500/50 bg-green-500/10"
            }`}
          >
            <div className="flex items-center gap-2 mb-3">
              {driftReport.drift_detected ? (
                <>
                  <span className="text-yellow-400">⚠️</span>
                  <span className="font-medium text-yellow-400">Drift Detected</span>
                </>
              ) : (
                <>
                  <span className="text-green-400">✓</span>
                  <span className="font-medium text-green-400">No Significant Drift</span>
                </>
              )}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              {Object.entries(driftReport)
                .filter(([k, v]) => typeof v === "number" && "pvalue" in k.toLowerCase())
                .slice(0, 4)
                .map(([key, value]) => (
                  <div key={key}>
                    <p className="text-slate-400">{key.replace("_", " ")}</p>
                    <p className="font-mono text-slate-200">{value.toFixed(4)}</p>
                  </div>
                ))}
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-400">No drift report available.</p>
        )}
      </section>

      {/* Cluster Stability */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Cluster Stability</h2>
        {stability?.stability_calculated ? (
          <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {stability.ari !== undefined && (
                <div>
                  <p className="text-xs text-slate-400 mb-1">Adjusted Rand Index (ARI)</p>
                  <p className="text-2xl font-mono text-sky-400">{stability.ari.toFixed(3)}</p>
                </div>
              )}
              {stability.nmi !== undefined && (
                <div>
                  <p className="text-xs text-slate-400 mb-1">Normalized Mutual Information (NMI)</p>
                  <p className="text-2xl font-mono text-sky-400">{stability.nmi.toFixed(3)}</p>
                </div>
              )}
              {stability.jaccard !== undefined && (
                <div>
                  <p className="text-xs text-slate-400 mb-1">Jaccard Similarity</p>
                  <p className="text-2xl font-mono text-sky-400">{stability.jaccard.toFixed(3)}</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-400">No baseline available for stability comparison.</p>
        )}
      </section>

      {/* API Metrics */}
      <section>
        <h2 className="text-lg font-semibold mb-3">API Performance (Last Hour)</h2>
        {metrics && metrics.total_requests > 0 ? (
          <div className="space-y-4">
            <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70">
              <p className="text-sm text-slate-400 mb-2">Total Requests</p>
              <p className="text-2xl font-mono text-sky-400">{metrics.total_requests}</p>
            </div>
            <div className="space-y-2">
              {Object.entries(metrics.endpoints).map(([endpoint, stats]) => (
                <div
                  key={endpoint}
                  className="border border-slate-800 rounded-lg p-3 bg-slate-900/70"
                >
                  <h4 className="font-medium text-sm mb-2">{endpoint}</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div>
                      <p className="text-slate-400">Count</p>
                      <p className="font-mono">{stats.count}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Avg Latency</p>
                      <p className="font-mono">{stats.avg_latency_ms.toFixed(2)}ms</p>
                    </div>
                    <div>
                      <p className="text-slate-400">P95 Latency</p>
                      <p className="font-mono">{stats.p95_latency_ms.toFixed(2)}ms</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Error Rate</p>
                      <p className="font-mono">{(stats.error_rate * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-400">No API metrics available yet.</p>
        )}
      </section>
    </div>
  );
}

