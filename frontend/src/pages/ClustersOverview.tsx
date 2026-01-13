import React, { useEffect, useState, useMemo } from "react";
import { ClusterSummary, fetchClustersOverview, fetchClusterSummary, fetchGraphVisualization, GraphNode, GraphEdge, fetchEmbeddingMap, EmbeddingPoint } from "../lib/api";
import { KnowledgeGraph } from "../components/KnowledgeGraph";
import { GraphControls } from "../components/GraphControls";
import { ClusterCardSkeleton } from "../components/ClusterCardSkeleton";
import { EmptyState } from "../components/EmptyState";
import { Tooltip } from "../components/Tooltip";
import { EmbeddingMap } from "../components/EmbeddingMap";
import { FaSort, FaSortUp, FaSortDown, FaDownload } from "react-icons/fa";
import { usePersistentState } from "../hooks/usePersistentState";
import { useToast } from "../hooks/useToast";
import { exportToCSV, exportToJSON } from "../utils/export";

export function ClustersOverviewPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [persistentUI, setPersistentUI] = usePersistentState<{
    activeTab: "table" | "graph" | "embedding-map";
    selectedClusterId: number | null;
    searchQuery: string;
    showLayers: { link: boolean; cluster: boolean; semantic: boolean };
    sortColumn: "name" | "size" | "id";
    sortDirection: "asc" | "desc";
  }>("clustersPageState", {
    activeTab: "table",
    selectedClusterId: null,
    searchQuery: "",
    showLayers: { link: true, cluster: true, semantic: true },
    sortColumn: "id",
    sortDirection: "asc",
  });
  const { activeTab, selectedClusterId, searchQuery, showLayers, sortColumn, sortDirection } = persistentUI;
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [embeddingMapPoints, setEmbeddingMapPoints] = useState<EmbeddingPoint[]>([]);
  const [embeddingMapLoading, setEmbeddingMapLoading] = useState(false);
  const toast = useToast();

  useEffect(() => {
    void loadOverview();
  }, []);

  async function loadOverview() {
    setLoading(true);
    try {
      const data = await fetchClustersOverview();
      setClusters(data);
      const targetId = persistentUI.selectedClusterId ?? (data.length > 0 ? data[0].cluster_id : null);
      if (targetId !== null) {
        const full = await fetchClusterSummary(targetId);
        setSelectedCluster(full);
        setPersistentUI((prev) => ({ ...prev, selectedClusterId: targetId }));
      }
    } catch (err: any) {
      toast.showError(err.message || "Failed to load clusters overview.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectCluster(id: number) {
    try {
      const summary = await fetchClusterSummary(id);
      setSelectedCluster(summary);
      setPersistentUI((prev) => ({ ...prev, selectedClusterId: id }));
      
      // Load graph data if graph tab is active
      if (activeTab === "graph") {
        await loadGraphData(id);
      }
    } catch (err: any) {
      toast.showError(err.message || `Failed to load cluster ${id}.`);
    }
  }

  async function loadGraphData(clusterId: number) {
    setGraphLoading(true);
    try {
      const data = await fetchGraphVisualization(clusterId);
      setGraphData(data);
    } catch (err: any) {
      toast.showError(err.message || `Failed to load graph for cluster ${clusterId}.`);
      setGraphData(null);
    } finally {
      setGraphLoading(false);
    }
  }

  async function loadEmbeddingMap() {
    setEmbeddingMapLoading(true);
    try {
      const data = await fetchEmbeddingMap();
      setEmbeddingMapPoints(data.points);
    } catch (err: any) {
      toast.showError(err.message || "Failed to load embedding map.");
      setEmbeddingMapPoints([]);
    } finally {
      setEmbeddingMapLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "graph" && selectedCluster) {
      void loadGraphData(selectedCluster.cluster_id);
    }
    if (activeTab === "embedding-map" && embeddingMapPoints.length === 0) {
      void loadEmbeddingMap();
    }
  }, [activeTab, selectedCluster]);

  const filteredGraphNodes = useMemo(() => {
    if (!graphData || !graphData.nodes || !Array.isArray(graphData.nodes)) {
      return [];
    }
    if (!searchQuery.trim()) {
      return graphData.nodes.filter((node) => node && node.id && node.label);
    }
    const query = searchQuery.toLowerCase();
    return graphData.nodes.filter((node) => {
      if (!node || !node.id || !node.label) return false;
      return node.label.toLowerCase().includes(query) || node.id.toLowerCase().includes(query);
    });
  }, [graphData, searchQuery]);

  const filteredGraphEdges = useMemo(() => {
    if (!graphData || !graphData.edges || !Array.isArray(graphData.edges)) {
      return [];
    }
    const nodeIds = new Set(filteredGraphNodes.map((n) => n?.id).filter(Boolean));
    return graphData.edges.filter(
      (edge) => edge && edge.source && edge.target && 
                nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );
  }, [graphData, filteredGraphNodes]);

  // Sort and filter clusters based on selected column, direction, and search query
  const sortedClusters = useMemo(() => {
    if (!clusters || clusters.length === 0) return [];
    
    // Filter by search query if provided (for table view)
    let filtered = clusters;
    if (activeTab === "table" && searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = clusters.filter((c) => {
        const name = (c.top_articles && c.top_articles.length > 0 ? c.top_articles[0] : `Cluster ${c.cluster_id}`).toLowerCase();
        const keywords = (c.keywords || []).join(" ").toLowerCase();
        return name.includes(query) || keywords.includes(query);
      });
    }
    
    // Sort filtered clusters
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;
      
      switch (sortColumn) {
        case "name":
          const nameA = (a.top_articles && a.top_articles.length > 0 ? a.top_articles[0] : `Cluster ${a.cluster_id}`).toLowerCase();
          const nameB = (b.top_articles && b.top_articles.length > 0 ? b.top_articles[0] : `Cluster ${b.cluster_id}`).toLowerCase();
          comparison = nameA.localeCompare(nameB);
          break;
        case "size":
          comparison = a.size - b.size;
          break;
        case "id":
          comparison = a.cluster_id - b.cluster_id;
          break;
      }
      
      return sortDirection === "asc" ? comparison : -comparison;
    });
    
    return sorted;
  }, [clusters, sortColumn, sortDirection, searchQuery, activeTab]);

  function handleSort(column: "name" | "size" | "id") {
    if (sortColumn === column) {
      // Toggle direction if same column
      setPersistentUI((prev) => ({
        ...prev,
        sortDirection: sortDirection === "asc" ? "desc" : "asc",
      }));
    } else {
      // New column, default to ascending
      setPersistentUI((prev) => ({
        ...prev,
        sortColumn: column,
        sortDirection: "asc",
      }));
    }
  }

  function getSortIcon(column: "name" | "size" | "id") {
    if (sortColumn !== column) {
      return <FaSort style={{ color: 'var(--text-tertiary)' }} />;
    }
    return sortDirection === "asc" 
      ? <FaSortUp className="text-sky-400" />
      : <FaSortDown className="text-sky-400" />;
  }

  return (
    <div className="space-y-6">
      <section>
        <div className="flex items-start justify-between mb-3">
          <div>
            <h1 className="text-2xl font-semibold mb-2">Clusters Overview</h1>
            <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
              Inspect topic clusters, their sizes, and representative articles. For each cluster we
              also show "topic words" – everyday words that appear often in this cluster and much less
              in others, giving a quick feel for what the cluster is about.
            </p>
          </div>
          {!loading && clusters.length > 0 && (
            <div className="flex gap-2">
              <Tooltip content="Export clusters as CSV">
                <button
                  onClick={() => {
                    try {
                      const exportData = clusters.map((c) => ({
                        cluster_id: c.cluster_id,
                        size: c.size,
                        keywords: c.keywords.join("; "),
                        top_articles: c.top_articles.join("; "),
                      }));
                      exportToCSV(exportData, "clusters");
                      toast.showSuccess("Clusters exported as CSV");
                    } catch (err: any) {
                      toast.showError(err.message || "Failed to export clusters");
                    }
                  }}
                  className="px-3 py-1.5 rounded text-xs transition-colors flex items-center gap-1.5"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  <FaDownload className="text-[10px]" />
                  CSV
                </button>
              </Tooltip>
              <Tooltip content="Export clusters as JSON">
                <button
                  onClick={() => {
                    try {
                      exportToJSON(clusters, "clusters");
                      toast.showSuccess("Clusters exported as JSON");
                    } catch (err: any) {
                      toast.showError(err.message || "Failed to export clusters");
                    }
                  }}
                  className="px-3 py-1.5 rounded text-xs transition-colors flex items-center gap-1.5"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  <FaDownload className="text-[10px]" />
                  JSON
                </button>
              </Tooltip>
            </div>
          )}
        </div>
        <div 
          className="text-xs border rounded-lg p-3"
          style={{ color: 'var(--text-tertiary)', backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}
        >
          <p className="font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Graph Node Colors:</p>
          <p className="mb-1">
            Each node (article) in the graph is colored based on its cluster assignment. 
            Different colors represent different topic clusters, making it easy to visually 
            identify which articles belong to the same topic group. Articles in the same cluster 
            share similar colors, while articles from different clusters have distinct colors.
          </p>
          <p className="italic" style={{ color: 'var(--text-tertiary)' }}>
            Note: The color palette cycles through 10 colors, so clusters with IDs that differ by 
            multiples of 10 may share the same color.
          </p>
        </div>
      </section>

      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <ClusterCardSkeleton key={i} />
          ))}
        </div>
      )}

      {!loading && clusters.length === 0 && (
        <EmptyState
          icon="database"
          title="No clusters available"
          description="Make sure you have run the DVC pipeline / clustering step. Go to the Ingestion page to start a new pipeline run."
          actionLabel="Go to Ingestion"
          onAction={() => window.location.href = "/ingestion"}
        />
      )}

      <div className="space-y-4">
        {/* Tab selector */}
        <div className="flex gap-2 border-b" style={{ borderColor: 'var(--border-color)' }}>
          <button
            onClick={() => setPersistentUI((prev) => ({ ...prev, activeTab: "table" }))}
            className="px-4 py-2 text-sm font-medium transition-colors"
            style={{
              color: activeTab === "table" ? 'var(--accent)' : 'var(--text-tertiary)',
              borderBottom: activeTab === "table" ? '2px solid var(--accent)' : 'none',
            }}
            onMouseEnter={(e) => {
              if (activeTab !== "table") {
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== "table") {
                e.currentTarget.style.color = 'var(--text-tertiary)';
              }
            }}
          >
            Table View
          </button>
          <button
            onClick={() => setPersistentUI((prev) => ({ ...prev, activeTab: "graph" }))}
            className="px-4 py-2 text-sm font-medium transition-colors"
            style={{
              color: activeTab === "graph" ? 'var(--accent)' : 'var(--text-tertiary)',
              borderBottom: activeTab === "graph" ? '2px solid var(--accent)' : 'none',
            }}
            onMouseEnter={(e) => {
              if (activeTab !== "graph") {
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== "graph") {
                e.currentTarget.style.color = 'var(--text-tertiary)';
              }
            }}
          >
            Graph View
          </button>
          <button
            onClick={() => setPersistentUI((prev) => ({ ...prev, activeTab: "embedding-map" }))}
            className="px-4 py-2 text-sm font-medium transition-colors"
            style={{
              color: activeTab === "embedding-map" ? 'var(--accent)' : 'var(--text-tertiary)',
              borderBottom: activeTab === "embedding-map" ? '2px solid var(--accent)' : 'none',
            }}
            onMouseEnter={(e) => {
              if (activeTab !== "embedding-map") {
                e.currentTarget.style.color = 'var(--text-secondary)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeTab !== "embedding-map") {
                e.currentTarget.style.color = 'var(--text-tertiary)';
              }
            }}
          >
            Embedding Map
          </button>
        </div>

        {activeTab === "table" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div 
              className="lg:col-span-2 border rounded-lg overflow-hidden"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div 
                className="px-3 py-2 border-b flex items-center justify-between"
                style={{ borderColor: 'var(--border-color)' }}
              >
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>All Clusters</span>
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                  Total: {clusters.length.toLocaleString()}
                </span>
              </div>
              <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--border-color)' }}>
                <input
                  type="text"
                  placeholder="Filter clusters by name or keywords..."
                  value={searchQuery}
                  onChange={(e) => setPersistentUI((prev) => ({ ...prev, searchQuery: e.target.value }))}
                  className="w-full px-3 py-2 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-all"
                  style={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                />
              </div>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
                <tr>
                  <th 
                    className="px-3 py-2 text-left font-medium cursor-pointer transition-colors select-none"
                    style={{ color: 'var(--text-tertiary)' }}
                    onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-tertiary)'}
                    onClick={() => handleSort("name")}
                  >
                    <div className="flex items-center gap-2">
                      <span>Cluster Name</span>
                      {getSortIcon("name")}
                    </div>
                  </th>
                  <th 
                    className="px-3 py-2 text-left font-medium cursor-pointer transition-colors select-none"
                    style={{ color: 'var(--text-tertiary)' }}
                    onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-tertiary)'}
                    onClick={() => handleSort("size")}
                  >
                    <div className="flex items-center gap-2">
                      <span>Size</span>
                      {getSortIcon("size")}
                    </div>
                  </th>
                  <th className="px-3 py-2 text-left font-medium" style={{ color: 'var(--text-tertiary)' }}>Topic words</th>
                  <th 
                    className="px-3 py-2 text-left font-medium cursor-pointer transition-colors select-none"
                    style={{ color: 'var(--text-tertiary)' }}
                    onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-tertiary)'}
                    onClick={() => handleSort("id")}
                  >
                    <div className="flex items-center gap-2">
                      <span>ID</span>
                      {getSortIcon("id")}
                    </div>
                  </th>
                  <th className="px-3 py-2 text-left font-medium" style={{ color: 'var(--text-tertiary)' }}></th>
                </tr>
              </thead>
              <tbody>
                {sortedClusters.map((c) => (
                  <tr
                    key={c.cluster_id}
                    className={`border-t transition-colors cursor-pointer group ${
                      selectedCluster?.cluster_id === c.cluster_id
                        ? ""
                        : ""
                    }`}
                    style={{ 
                      borderColor: 'var(--border-color)',
                      backgroundColor: selectedCluster?.cluster_id === c.cluster_id 
                        ? 'rgba(14, 165, 233, 0.1)' 
                        : 'transparent',
                      borderLeft: selectedCluster?.cluster_id === c.cluster_id 
                        ? '3px solid var(--accent)' 
                        : 'none'
                    }}
                    onClick={() => handleSelectCluster(c.cluster_id)}
                    onMouseEnter={(e) => {
                      if (selectedCluster?.cluster_id !== c.cluster_id) {
                        e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedCluster?.cluster_id !== c.cluster_id) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                  >
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-primary)' }}>
                      {c.top_articles && c.top_articles.length > 0
                        ? c.top_articles[0]
                        : `Cluster ${c.cluster_id}`}
                    </td>
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-primary)' }}>{c.size}</td>
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-secondary)' }}>
                      {c.keywords && c.keywords.length > 0
                        ? c.keywords.slice(0, 8).join(", ")
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                      {c.cluster_id}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      <button
                        onClick={() => handleSelectCluster(c.cluster_id)}
                        className="px-2 py-1 rounded text-[11px] transition-colors"
                        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
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

        <div 
          className="border rounded-lg p-4"
          style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
        >
          {selectedCluster ? (
            <>
              <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                {selectedCluster.top_articles && selectedCluster.top_articles.length > 0
                  ? selectedCluster.top_articles[0]
                  : `Cluster ${selectedCluster.cluster_id}`}
              </h2>
              <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>
                Cluster ID:{" "}
                <span className="font-mono" style={{ color: 'var(--accent)' }}>
                  {selectedCluster.cluster_id}
                </span>
              </p>
              <div className="space-y-3 text-xs">
                <p style={{ color: 'var(--text-primary)' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Size:</span>{" "}
                  {selectedCluster.size.toLocaleString()}
                </p>
                <div>
                  <p className="mb-1" style={{ color: 'var(--text-tertiary)' }}>
                    Topic words (high-frequency and distinctive for this cluster):
                  </p>
                  <p style={{ color: 'var(--text-secondary)' }}>
                    {selectedCluster.keywords.slice(0, 20).join(", ")}
                  </p>
                </div>
                <div>
                  <p className="mb-1" style={{ color: 'var(--text-tertiary)' }}>Representative Articles:</p>
                  <ul className="list-disc list-inside space-y-0.5" style={{ color: 'var(--text-secondary)' }}>
                    {selectedCluster.top_articles.slice(0, 10).map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </>
          ) : (
            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Select a cluster to inspect details.</p>
          )}
        </div>
      </div>
        )}

        {activeTab === "graph" && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div 
              className="lg:col-span-3 border rounded-lg overflow-hidden"
              style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
            >
              <div 
                className="px-3 py-2 border-b flex items-center justify-between"
                style={{ borderColor: 'var(--border-color)' }}
              >
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                  Knowledge Graph - {selectedCluster ? `Cluster ${selectedCluster.cluster_id}` : "Select a cluster"}
                </span>
              </div>
              <div className="h-[600px] p-4">
                {graphLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-tertiary)' }}>
                      <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--accent)' }}></div>
                      <span>Loading graph...</span>
                    </div>
                  </div>
                ) : graphData && graphData.nodes && graphData.nodes.length > 0 ? (
                  <KnowledgeGraph
                    nodes={filteredGraphNodes}
                    edges={filteredGraphEdges}
                    showLayers={showLayers}
                    onNodeClick={(node) => {
                      console.log("Clicked node:", node);
                    }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
                      {selectedCluster
                        ? "No graph data available for this cluster."
                        : "Select a cluster to view its knowledge graph."}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <GraphControls
                showLayers={showLayers}
                onLayerToggle={(layer) => {
                  setPersistentUI((prev) => ({
                    ...prev,
                    showLayers: { ...prev.showLayers, [layer]: !prev.showLayers[layer] },
                  }));
                }}
                onResetView={() => {
                  setPersistentUI((prev) => ({ ...prev, searchQuery: "" }));
                  // ReactFlow will handle view reset via fitView
                }}
                searchQuery={searchQuery}
                onSearchChange={(value) => setPersistentUI((prev) => ({ ...prev, searchQuery: value }))}
              />

              {selectedCluster && (
                <div 
                  className="border rounded-lg p-4"
                  style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
                >
                  <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                    {selectedCluster.top_articles && selectedCluster.top_articles.length > 0
                      ? selectedCluster.top_articles[0]
                      : `Cluster ${selectedCluster.cluster_id}`}
                  </h2>
                  <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>
                    Cluster ID:{" "}
                    <span className="font-mono" style={{ color: 'var(--accent)' }}>
                      {selectedCluster.cluster_id}
                    </span>
                  </p>
                  <div className="space-y-3 text-xs">
                    <p style={{ color: 'var(--text-primary)' }}>
                      <span style={{ color: 'var(--text-tertiary)' }}>Size:</span>{" "}
                      {selectedCluster.size.toLocaleString()}
                    </p>
                    <div>
                      <p className="mb-1" style={{ color: 'var(--text-tertiary)' }}>Topic words:</p>
                      <p style={{ color: 'var(--text-secondary)' }}>
                        {selectedCluster.keywords.slice(0, 10).join(", ")}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "embedding-map" && (
          <div className="space-y-4">
            {embeddingMapLoading ? (
              <div className="flex items-center justify-center h-96" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-tertiary)' }}>
                  <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--accent)' }}></div>
                  <span>Loading embedding map...</span>
                </div>
              </div>
            ) : embeddingMapPoints.length > 0 ? (
              <EmbeddingMap
                points={embeddingMapPoints}
                selectedClusterId={selectedClusterId}
                searchQuery={searchQuery}
                onPointClick={(point) => {
                  console.log("Clicked point:", point);
                  // Optionally select the cluster
                  if (point.cluster_id !== selectedClusterId) {
                    void handleSelectCluster(point.cluster_id);
                  }
                }}
              />
            ) : (
              <div
                className="flex items-center justify-center h-96 rounded-lg"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <div className="text-center">
                  <div className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
                    No Embedding Map Available
                  </div>
                  <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    Run the clustering pipeline to generate the 2D embedding visualization.
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


