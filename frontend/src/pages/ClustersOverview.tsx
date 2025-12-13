import React, { useEffect, useState, useMemo } from "react";
import { ClusterSummary, fetchClustersOverview, fetchClusterSummary, fetchGraphVisualization, GraphNode, GraphEdge } from "../lib/api";
import { KnowledgeGraph } from "../components/KnowledgeGraph";
import { GraphControls } from "../components/GraphControls";
import { FaSort, FaSortUp, FaSortDown } from "react-icons/fa";

export function ClustersOverviewPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"table" | "graph">("table");
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [showLayers, setShowLayers] = useState({
    cluster: true,
    semantic: true,
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [sortColumn, setSortColumn] = useState<"name" | "size" | "id">("id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

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
      
      // Load graph data if graph tab is active
      if (activeTab === "graph") {
        await loadGraphData(id);
      }
    } catch (err: any) {
      setError(err.message || `Failed to load cluster ${id}.`);
    }
  }

  async function loadGraphData(clusterId: number) {
    setGraphLoading(true);
    setGraphError(null);
    try {
      const data = await fetchGraphVisualization(clusterId);
      setGraphData(data);
    } catch (err: any) {
      setGraphError(err.message || `Failed to load graph for cluster ${clusterId}.`);
      setGraphData(null);
    } finally {
      setGraphLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "graph" && selectedCluster) {
      void loadGraphData(selectedCluster.cluster_id);
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

  // Sort clusters based on selected column and direction
  const sortedClusters = useMemo(() => {
    if (!clusters || clusters.length === 0) return [];
    
    const sorted = [...clusters].sort((a, b) => {
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
  }, [clusters, sortColumn, sortDirection]);

  function handleSort(column: "name" | "size" | "id") {
    if (sortColumn === column) {
      // Toggle direction if same column
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      // New column, default to ascending
      setSortColumn(column);
      setSortDirection("asc");
    }
  }

  function getSortIcon(column: "name" | "size" | "id") {
    if (sortColumn !== column) {
      return <FaSort className="text-slate-500" />;
    }
    return sortDirection === "asc" 
      ? <FaSortUp className="text-sky-400" />
      : <FaSortDown className="text-sky-400" />;
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Clusters Overview</h1>
        <p className="text-sm text-slate-400 mb-3">
          Inspect topic clusters, their sizes, and representative articles. For each cluster we
          also show "topic words" – everyday words that appear often in this cluster and much less
          in others, giving a quick feel for what the cluster is about.
        </p>
        <div className="text-xs text-slate-400 bg-slate-900/50 border border-slate-800 rounded-lg p-3">
          <p className="font-medium text-slate-300 mb-2">Graph Node Colors:</p>
          <p className="mb-1">
            Each node (article) in the graph is colored based on its cluster assignment. 
            Different colors represent different topic clusters, making it easy to visually 
            identify which articles belong to the same topic group. Articles in the same cluster 
            share similar colors, while articles from different clusters have distinct colors.
          </p>
          <p className="text-slate-500 italic">
            Note: The color palette cycles through 10 colors, so clusters with IDs that differ by 
            multiples of 10 may share the same color.
          </p>
        </div>
      </section>

      {loading && <p className="text-sm text-slate-300">Loading clusters…</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {!loading && clusters.length === 0 && !error && (
        <p className="text-sm text-slate-400">
          No clusters available. Make sure you have run the DVC pipeline / clustering step.
        </p>
      )}

      <div className="space-y-4">
        {/* Tab selector */}
        <div className="flex gap-2 border-b border-slate-800">
          <button
            onClick={() => setActiveTab("table")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === "table"
                ? "text-sky-400 border-b-2 border-sky-400"
                : "text-slate-400 hover:text-slate-300"
            }`}
          >
            Table View
          </button>
          <button
            onClick={() => setActiveTab("graph")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === "graph"
                ? "text-sky-400 border-b-2 border-sky-400"
                : "text-slate-400 hover:text-slate-300"
            }`}
          >
            Graph View
          </button>
        </div>

        {activeTab === "table" && (
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
                  <th 
                    className="px-3 py-2 text-left font-medium text-slate-400 cursor-pointer hover:text-slate-300 transition-colors select-none"
                    onClick={() => handleSort("name")}
                  >
                    <div className="flex items-center gap-2">
                      <span>Cluster Name</span>
                      {getSortIcon("name")}
                    </div>
                  </th>
                  <th 
                    className="px-3 py-2 text-left font-medium text-slate-400 cursor-pointer hover:text-slate-300 transition-colors select-none"
                    onClick={() => handleSort("size")}
                  >
                    <div className="flex items-center gap-2">
                      <span>Size</span>
                      {getSortIcon("size")}
                    </div>
                  </th>
                  <th className="px-3 py-2 text-left font-medium text-slate-400">Topic words</th>
                  <th 
                    className="px-3 py-2 text-left font-medium text-slate-400 cursor-pointer hover:text-slate-300 transition-colors select-none"
                    onClick={() => handleSort("id")}
                  >
                    <div className="flex items-center gap-2">
                      <span>ID</span>
                      {getSortIcon("id")}
                    </div>
                  </th>
                  <th className="px-3 py-2 text-left font-medium text-slate-400"></th>
                </tr>
              </thead>
              <tbody>
                {sortedClusters.map((c) => (
                  <tr
                    key={c.cluster_id}
                    className="border-t border-slate-800 hover:bg-slate-800/60"
                  >
                    <td className="px-3 py-1.5 text-slate-200">
                      {c.top_articles && c.top_articles.length > 0
                        ? c.top_articles[0]
                        : `Cluster ${c.cluster_id}`}
                    </td>
                    <td className="px-3 py-1.5">{c.size}</td>
                    <td className="px-3 py-1.5 text-slate-300">
                      {c.keywords && c.keywords.length > 0
                        ? c.keywords.slice(0, 8).join(", ")
                        : "—"}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-slate-400 text-[10px]">
                      {c.cluster_id}
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
          {selectedCluster ? (
            <>
              <h2 className="text-lg font-semibold mb-1">
                {selectedCluster.top_articles && selectedCluster.top_articles.length > 0
                  ? selectedCluster.top_articles[0]
                  : `Cluster ${selectedCluster.cluster_id}`}
              </h2>
              <p className="text-sm text-slate-400 mb-2">
                Cluster ID:{" "}
                <span className="font-mono text-sky-400">
                  {selectedCluster.cluster_id}
                </span>
              </p>
              <div className="space-y-3 text-xs">
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
            </>
          ) : (
            <p className="text-xs text-slate-400">Select a cluster to inspect details.</p>
          )}
        </div>
      </div>
        )}

        {activeTab === "graph" && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div className="lg:col-span-3 border border-slate-800 rounded-lg bg-slate-900/70 overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                <span className="text-sm font-medium">
                  Knowledge Graph - {selectedCluster ? `Cluster ${selectedCluster.cluster_id}` : "Select a cluster"}
                </span>
              </div>
              <div className="h-[600px] p-4">
                {graphLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                      <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
                      <span>Loading graph...</span>
                    </div>
                  </div>
                ) : graphError ? (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-sm text-red-400">{graphError}</p>
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
                    <p className="text-sm text-slate-400">
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
                  setShowLayers((prev) => ({
                    ...prev,
                    [layer]: !prev[layer],
                  }));
                }}
                onResetView={() => {
                  setSearchQuery("");
                  // ReactFlow will handle view reset via fitView
                }}
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
              />

              {selectedCluster && (
                <div className="border border-slate-800 rounded-lg bg-slate-900/70 p-4">
                  <h2 className="text-lg font-semibold mb-1">
                    {selectedCluster.top_articles && selectedCluster.top_articles.length > 0
                      ? selectedCluster.top_articles[0]
                      : `Cluster ${selectedCluster.cluster_id}`}
                  </h2>
                  <p className="text-sm text-slate-400 mb-2">
                    Cluster ID:{" "}
                    <span className="font-mono text-sky-400">
                      {selectedCluster.cluster_id}
                    </span>
                  </p>
                  <div className="space-y-3 text-xs">
                    <p>
                      <span className="text-slate-400">Size:</span>{" "}
                      {selectedCluster.size.toLocaleString()}
                    </p>
                    <div>
                      <p className="text-slate-400 mb-1">Topic words:</p>
                      <p className="text-slate-200">
                        {selectedCluster.keywords.slice(0, 10).join(", ")}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


