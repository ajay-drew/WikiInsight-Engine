import React, { useEffect, useState } from "react";
import { FaSync } from "react-icons/fa";
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
import { useToast } from "../hooks/useToast";

export function MonitoringPage() {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [metrics, setMetrics] = useState<APIMetricsSummary | null>(null);
  const [driftReport, setDriftReport] = useState<DriftReport | null>(null);
  const [stability, setStability] = useState<StabilityMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

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
      toast.showError(err.message || "Failed to load monitoring data.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-2xl font-semibold">Monitoring Dashboard</h1>
          <button
            onClick={loadAllData}
            disabled={loading}
            className="px-3 py-1.5 rounded flex items-center gap-2 text-sm transition-colors disabled:opacity-50"
            style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            onMouseEnter={(e) => !loading && (e.currentTarget.style.backgroundColor = 'var(--bg-secondary)')}
            onMouseLeave={(e) => !loading && (e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)')}
          >
            <FaSync className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          Real-time monitoring of pipeline status, data drift, cluster stability, and API performance.
        </p>
      </section>

      {loading && (
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-tertiary)' }}>
          <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          <span>Loading monitoring data...</span>
        </div>
      )}

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
                  <h3 className="font-medium capitalize" style={{ color: 'var(--text-primary)' }}>{stage}</h3>
                  {info.has_artifacts ? (
                    <span className="text-green-400 text-xs">✓ Ready</span>
                  ) : (
                    <span className="text-red-400 text-xs">✗ Missing</span>
                  )}
                </div>
                {info.last_modified && (
                  <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                    Last modified: {new Date(info.last_modified).toLocaleString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No pipeline status available.</p>
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
                    <p style={{ color: 'var(--text-tertiary)' }}>{key.replace("_", " ")}</p>
                    <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>{value.toFixed(4)}</p>
                  </div>
                ))}
            </div>
          </div>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No drift report available.</p>
        )}
      </section>

      {/* Cluster Stability */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Cluster Stability</h2>
        {stability?.stability_calculated ? (
          <div 
            className="border rounded-lg p-4"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {stability.ari !== undefined && (
                <div>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>Adjusted Rand Index (ARI)</p>
                  <p className="text-2xl font-mono" style={{ color: 'var(--accent)' }}>{stability.ari.toFixed(3)}</p>
                </div>
              )}
              {stability.nmi !== undefined && (
                <div>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>Normalized Mutual Information (NMI)</p>
                  <p className="text-2xl font-mono" style={{ color: 'var(--accent)' }}>{stability.nmi.toFixed(3)}</p>
                </div>
              )}
              {stability.jaccard !== undefined && (
                <div>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-tertiary)' }}>Jaccard Similarity</p>
                  <p className="text-2xl font-mono" style={{ color: 'var(--accent)' }}>{stability.jaccard.toFixed(3)}</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No baseline available for stability comparison.</p>
        )}
      </section>

      {/* API Metrics */}
      <section>
        <h2 className="text-lg font-semibold mb-3">API Performance (Last Hour)</h2>
        {metrics && metrics.total_requests > 0 ? (
          <div className="space-y-4">
            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>Total Requests</p>
              <p className="text-2xl font-mono" style={{ color: 'var(--accent)' }}>{metrics.total_requests}</p>
            </div>
            <div className="space-y-2">
              {Object.entries(metrics.endpoints).map(([endpoint, stats]) => (
                <div
                  key={endpoint}
                  className="border rounded-lg p-3"
                  style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
                >
                  <h4 className="font-medium text-sm mb-2" style={{ color: 'var(--text-primary)' }}>{endpoint}</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div>
                      <p style={{ color: 'var(--text-tertiary)' }}>Count</p>
                      <p className="font-mono" style={{ color: 'var(--text-primary)' }}>{stats.count}</p>
                    </div>
                    <div>
                      <p style={{ color: 'var(--text-tertiary)' }}>Avg Latency</p>
                      <p className="font-mono" style={{ color: 'var(--text-primary)' }}>{stats.avg_latency_ms.toFixed(2)}ms</p>
                    </div>
                    <div>
                      <p style={{ color: 'var(--text-tertiary)' }}>P95 Latency</p>
                      <p className="font-mono" style={{ color: 'var(--text-primary)' }}>{stats.p95_latency_ms.toFixed(2)}ms</p>
                    </div>
                    <div>
                      <p style={{ color: 'var(--text-tertiary)' }}>Error Rate</p>
                      <p className="font-mono" style={{ color: 'var(--text-primary)' }}>{(stats.error_rate * 100).toFixed(1)}%</p>
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

