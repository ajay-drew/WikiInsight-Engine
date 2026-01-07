import React, { useEffect, useState } from "react";
import { 
  fetchClustersOverview, 
  fetchPipelineStatus, 
  fetchMetrics,
  ClusterSummary 
} from "../lib/api";
import { useToast } from "../hooks/useToast";
import { EmptyState } from "../components/EmptyState";
import { Skeleton } from "../components/Skeleton";
import { FaSearch, FaDatabase, FaChartLine, FaCog, FaArrowRight } from "react-icons/fa";
import { Link } from "react-router-dom";

export function DashboardPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const toast = useToast();

  useEffect(() => {
    void loadDashboardData();
  }, []);

  async function loadDashboardData() {
    setLoading(true);
    try {
      const [clustersData, status, apiMetrics] = await Promise.all([
        fetchClustersOverview().catch(() => []),
        fetchPipelineStatus().catch(() => null),
        fetchMetrics(3600).catch(() => null),
      ]);
      setClusters(clustersData);
      setPipelineStatus(status);
      setMetrics(apiMetrics);
    } catch (err: any) {
      toast.showError("Failed to load dashboard data");
    } finally {
      setLoading(false);
    }
  }

  const totalArticles = clusters.reduce((sum, c) => sum + c.size, 0);
  const totalClusters = clusters.length;
  const pipelineReady = pipelineStatus && 
    pipelineStatus.clustering?.has_artifacts && 
    pipelineStatus.preprocessing?.has_artifacts;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Dashboard</h1>
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          Overview of your WikiInsight system status and key metrics.
        </p>
      </section>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div 
              key={i} 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <Skeleton variant="text" width="60%" height={16} />
              <Skeleton variant="text" width="40%" height={32} className="mt-2" />
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Total Articles</span>
                <FaDatabase style={{ color: 'var(--text-tertiary)' }} />
              </div>
              <p className="text-2xl font-semibold" style={{ color: 'var(--accent)' }}>{totalArticles.toLocaleString()}</p>
            </div>

            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Total Clusters</span>
                <FaChartLine style={{ color: 'var(--text-tertiary)' }} />
              </div>
              <p className="text-2xl font-semibold" style={{ color: 'var(--accent)' }}>{totalClusters}</p>
            </div>

            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Pipeline Status</span>
                <FaCog style={{ color: 'var(--text-tertiary)' }} />
              </div>
              <p className="text-2xl font-semibold" style={{ color: pipelineReady ? '#10b981' : '#f59e0b' }}>
                {pipelineReady ? "Ready" : "Not Ready"}
              </p>
            </div>

            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>API Requests (1h)</span>
                <FaSearch style={{ color: 'var(--text-tertiary)' }} />
              </div>
              <p className="text-2xl font-semibold" style={{ color: 'var(--accent)' }}>
                {metrics?.total_requests?.toLocaleString() || "0"}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <h2 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Quick Actions</h2>
              <div className="space-y-2">
                <Link
                  to="/search"
                  className="flex items-center justify-between p-3 rounded transition-colors group"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  <span className="text-sm">Search Articles</span>
                  <FaArrowRight className="text-xs" style={{ color: 'var(--text-tertiary)' }} />
                </Link>
                <Link
                  to="/clusters"
                  className="flex items-center justify-between p-3 rounded transition-colors group"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  <span className="text-sm">Browse Clusters</span>
                  <FaArrowRight className="text-xs" style={{ color: 'var(--text-tertiary)' }} />
                </Link>
                <Link
                  to="/ingestion"
                  className="flex items-center justify-between p-3 rounded transition-colors group"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  <span className="text-sm">Run Pipeline</span>
                  <FaArrowRight className="text-xs" style={{ color: 'var(--text-tertiary)' }} />
                </Link>
              </div>
            </div>

            <div 
              className="border rounded-lg p-4"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <h2 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>System Status</h2>
              {pipelineStatus ? (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Ingestion</span>
                    <span style={{ color: pipelineStatus.ingestion?.has_artifacts ? '#10b981' : '#ef4444' }}>
                      {pipelineStatus.ingestion?.has_artifacts ? "✓" : "✗"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Preprocessing</span>
                    <span style={{ color: pipelineStatus.preprocessing?.has_artifacts ? '#10b981' : '#ef4444' }}>
                      {pipelineStatus.preprocessing?.has_artifacts ? "✓" : "✗"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-tertiary)' }}>Clustering</span>
                    <span style={{ color: pipelineStatus.clustering?.has_artifacts ? '#10b981' : '#ef4444' }}>
                      {pipelineStatus.clustering?.has_artifacts ? "✓" : "✗"}
                    </span>
                  </div>
                </div>
              ) : (
                <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No pipeline status available</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

