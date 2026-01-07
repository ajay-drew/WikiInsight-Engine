import React from "react";
import { FaFilter, FaSearch, FaUndo } from "react-icons/fa";
import { Tooltip } from "./Tooltip";

interface GraphControlsProps {
  showLayers: {
    link: boolean;
    cluster: boolean;
    semantic: boolean;
  };
  onLayerToggle: (layer: "link" | "cluster" | "semantic") => void;
  onResetView: () => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

export function GraphControls({
  showLayers,
  onLayerToggle,
  onResetView,
  searchQuery,
  onSearchChange,
}: GraphControlsProps) {
  return (
    <div 
      className="border rounded-lg p-4 space-y-4"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
    >
      <div className="flex items-center gap-2 mb-4">
        <FaFilter style={{ color: 'var(--text-tertiary)' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Graph Controls</h3>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs mb-2 block" style={{ color: 'var(--text-tertiary)' }}>Layer Visibility</label>
          <div className="space-y-2">
            <Tooltip content="Show/hide Wikipedia article links (Layer 1)">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showLayers.link}
                  onChange={() => onLayerToggle("link")}
                  className="rounded focus:ring-2 transition-all"
                  style={{ 
                    borderColor: 'var(--border-color)', 
                    backgroundColor: 'var(--bg-tertiary)',
                    accentColor: 'var(--accent)'
                  }}
                />
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Wikipedia Links (Layer 1)</span>
                <span className="w-3 h-3 rounded bg-blue-500"></span>
              </label>
            </Tooltip>
            <Tooltip content="Show/hide cluster relationships - articles in the same cluster (Layer 2)">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showLayers.cluster}
                  onChange={() => onLayerToggle("cluster")}
                  className="rounded focus:ring-2 transition-all"
                  style={{ 
                    borderColor: 'var(--border-color)', 
                    backgroundColor: 'var(--bg-tertiary)',
                    accentColor: 'var(--accent)'
                  }}
                />
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Cluster Relationships (Layer 2)</span>
                <span className="w-3 h-3 rounded bg-green-500"></span>
              </label>
            </Tooltip>
            <Tooltip content="Show/hide semantic similarity edges - articles with high embedding similarity (Layer 3)">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showLayers.semantic}
                  onChange={() => onLayerToggle("semantic")}
                  className="rounded focus:ring-2 transition-all"
                  style={{ 
                    borderColor: 'var(--border-color)', 
                    backgroundColor: 'var(--bg-tertiary)',
                    accentColor: 'var(--accent)'
                  }}
                />
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Semantic Similarity (Layer 3)</span>
                <span className="w-3 h-3 rounded bg-orange-500"></span>
              </label>
            </Tooltip>
          </div>
        </div>

        <div>
          <label className="text-xs mb-2 block flex items-center gap-1" style={{ color: 'var(--text-tertiary)' }}>
            <FaSearch className="text-[10px]" />
            Search Nodes
          </label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Filter by article title..."
            className="w-full px-3 py-1.5 rounded text-sm focus:outline-none focus:ring-2 transition-all"
            style={{ 
              backgroundColor: 'var(--bg-tertiary)', 
              borderColor: 'var(--border-color)', 
              color: 'var(--text-primary)'
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
          />
        </div>

        <button
          onClick={onResetView}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs transition-colors"
          style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
        >
          <FaUndo className="text-[10px]" />
          Reset View
        </button>
      </div>

      <div className="pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
        <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
          Click nodes to explore connections. Use controls to filter layers and search for specific articles.
        </p>
      </div>
    </div>
  );
}

