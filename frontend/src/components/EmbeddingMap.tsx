import React, { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ZAxis,
} from "recharts";
import { EmbeddingPoint } from "../lib/api";

type EmbeddingMapProps = {
  points: EmbeddingPoint[];
  selectedClusterId: number | null;
  searchQuery: string;
  onPointClick?: (point: EmbeddingPoint) => void;
};

// Generate distinct colors for clusters
const CLUSTER_COLORS = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#10b981", // green
  "#f59e0b", // amber
  "#8b5cf6", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#6366f1", // indigo
  "#84cc16", // lime
];

function getClusterColor(clusterId: number): string {
  if (clusterId < 0) return "#6b7280"; // gray for unassigned
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

export function EmbeddingMap({
  points,
  selectedClusterId,
  searchQuery,
  onPointClick,
}: EmbeddingMapProps) {
  // Filter and enhance points
  const processedPoints = useMemo(() => {
    const query = searchQuery.toLowerCase().trim();
    
    return points.map((point) => {
      const matchesSearch = !query || point.title.toLowerCase().includes(query);
      const isSelected = selectedClusterId === null || point.cluster_id === selectedClusterId;
      
      return {
        ...point,
        matchesSearch,
        isSelected,
        opacity: isSelected ? 1.0 : 0.3,
        size: matchesSearch && isSelected ? 100 : matchesSearch ? 80 : isSelected ? 60 : 40,
      };
    });
  }, [points, selectedClusterId, searchQuery]);

  // Get unique clusters for legend
  const clusters = useMemo(() => {
    const uniqueClusters = new Map<number, { id: number; count: number; color: string }>();
    
    points.forEach((point) => {
      if (!uniqueClusters.has(point.cluster_id)) {
        uniqueClusters.set(point.cluster_id, {
          id: point.cluster_id,
          count: 0,
          color: getClusterColor(point.cluster_id),
        });
      }
      const cluster = uniqueClusters.get(point.cluster_id)!;
      cluster.count += 1;
    });
    
    return Array.from(uniqueClusters.values()).sort((a, b) => a.id - b.id);
  }, [points]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;
    
    const point = payload[0].payload as EmbeddingPoint & { matchesSearch: boolean; isSelected: boolean };
    
    return (
      <div
        className="rounded-lg shadow-lg p-3 max-w-xs"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
        }}
      >
        <div className="font-semibold text-sm mb-1" style={{ color: 'var(--text-primary)' }}>
          {point.title}
        </div>
        <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
          Cluster: {point.cluster_id}
        </div>
        {point.keywords && point.keywords.length > 0 && (
          <div className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
            Keywords: {point.keywords.slice(0, 3).join(", ")}
          </div>
        )}
      </div>
    );
  };

  if (points.length === 0) {
    return (
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
    );
  }

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div
        className="rounded-lg p-4"
        style={{ backgroundColor: 'var(--bg-secondary)', borderLeft: '4px solid #3b82f6' }}
      >
        <div className="text-sm" style={{ color: 'var(--text-primary)' }}>
          <strong>2D Embedding Map:</strong> Each point represents an article projected to 2D using UMAP.
          Colors indicate clusters. Hover for details, click to highlight.
        </div>
      </div>

      {/* Legend */}
      <div
        className="rounded-lg p-4"
        style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
      >
        <div className="text-sm font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
          Clusters ({clusters.length})
        </div>
        <div className="flex flex-wrap gap-3">
          {clusters.map((cluster) => (
            <div key={cluster.id} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: cluster.color }}
              />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Cluster {cluster.id} ({cluster.count})
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Scatter plot */}
      <div
        className="rounded-lg p-4"
        style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
      >
        <ResponsiveContainer width="100%" height={600}>
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
            <XAxis
              type="number"
              dataKey="x"
              name="UMAP X"
              stroke="var(--text-tertiary)"
              tick={{ fill: 'var(--text-tertiary)' }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="UMAP Y"
              stroke="var(--text-tertiary)"
              tick={{ fill: 'var(--text-tertiary)' }}
            />
            <ZAxis type="number" dataKey="size" range={[40, 100]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter
              data={processedPoints}
              fill="#8884d8"
              onClick={(data) => {
                if (onPointClick && data) {
                  onPointClick(data as EmbeddingPoint);
                }
              }}
              style={{ cursor: 'pointer' }}
            >
              {processedPoints.map((point, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getClusterColor(point.cluster_id)}
                  fillOpacity={point.opacity}
                  stroke={point.matchesSearch && point.isSelected ? "#fff" : "none"}
                  strokeWidth={point.matchesSearch && point.isSelected ? 2 : 0}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Stats */}
      <div
        className="rounded-lg p-4"
        style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
      >
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
              {points.length}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>Total Articles</div>
          </div>
          <div>
            <div className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
              {clusters.length}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>Clusters</div>
          </div>
          <div>
            <div className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
              {processedPoints.filter((p) => p.matchesSearch && p.isSelected).length}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>Visible</div>
          </div>
        </div>
      </div>
    </div>
  );
}
