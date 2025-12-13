import React from "react";
import { FaFilter, FaSearch, FaUndo } from "react-icons/fa";

interface GraphControlsProps {
  showLayers: {
    cluster: boolean;
    semantic: boolean;
  };
  onLayerToggle: (layer: "cluster" | "semantic") => void;
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
    <div className="border border-slate-800 rounded-lg bg-slate-900/70 p-4 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <FaFilter className="text-slate-400" />
        <h3 className="text-sm font-medium text-slate-200">Graph Controls</h3>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs text-slate-400 mb-2 block">Layer Visibility</label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showLayers.cluster}
                onChange={() => onLayerToggle("cluster")}
                className="rounded border-slate-600 bg-slate-800 text-sky-500 focus:ring-sky-500"
              />
              <span className="text-xs text-slate-300">Cluster Relationships (Layer 2)</span>
              <span className="w-3 h-3 rounded bg-green-500"></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showLayers.semantic}
                onChange={() => onLayerToggle("semantic")}
                className="rounded border-slate-600 bg-slate-800 text-sky-500 focus:ring-sky-500"
              />
              <span className="text-xs text-slate-300">Semantic Similarity (Layer 3)</span>
              <span className="w-3 h-3 rounded bg-orange-500"></span>
            </label>
          </div>
        </div>

        <div>
          <label className="text-xs text-slate-400 mb-2 block flex items-center gap-1">
            <FaSearch className="text-[10px]" />
            Search Nodes
          </label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Filter by article title..."
            className="w-full px-3 py-1.5 rounded bg-slate-800 border border-slate-700 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent"
          />
        </div>

        <button
          onClick={onResetView}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition-colors"
        >
          <FaUndo className="text-[10px]" />
          Reset View
        </button>
      </div>

      <div className="pt-3 border-t border-slate-800">
        <p className="text-xs text-slate-500">
          Click nodes to explore connections. Use controls to filter layers and search for specific articles.
        </p>
      </div>
    </div>
  );
}

