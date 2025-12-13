import React, { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  ConnectionMode,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { GraphNode, GraphEdge } from "../lib/api";

interface KnowledgeGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (node: GraphNode) => void;
  showLayers?: {
    cluster: boolean;
    semantic: boolean;
  };
}

// Color palette for clusters
const CLUSTER_COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
];

function getClusterColor(clusterId: number): string {
  if (clusterId < 0) return "#64748b"; // Gray for unclustered
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

export function KnowledgeGraph({
  nodes,
  edges,
  onNodeClick,
  showLayers = { cluster: true, semantic: true },
}: KnowledgeGraphProps) {
  // Filter edges based on layer visibility and validity
  const filteredEdges = useMemo(() => {
    if (!edges || !Array.isArray(edges)) {
      return [];
    }
    return edges.filter((edge) => {
      if (!edge || !edge.source || !edge.target) return false;
      
      // Ensure layer is a number for comparison
      const layer = typeof edge.layer === 'number' ? edge.layer : parseInt(String(edge.layer), 10);
      
      if (layer === 2 && !showLayers.cluster) return false;
      if (layer === 3 && !showLayers.semantic) return false;
      
      // If layer is 0 or invalid, exclude it
      if (layer === 0 || isNaN(layer)) return false;
      
      return true;
    });
  }, [edges, showLayers]);

  // Convert to ReactFlow format
  const flowNodes: Node[] = useMemo(() => {
    // Validate nodes array
    if (!nodes || !Array.isArray(nodes)) {
      return [];
    }

    // Filter out any invalid nodes (null, undefined, or missing required fields)
    const validNodes = nodes.filter(
      (node) => node && node.id && node.label !== undefined && node.cluster_id !== undefined
    );

    return validNodes
      .map((node, index) => {
        // Ensure node is valid before accessing properties
        if (!node || !node.id) {
          return null;
        }

        // Provide default position if not available (ReactFlow needs positions)
        // Use a simple grid layout as fallback
        const defaultX = (index % 10) * 150;
        const defaultY = Math.floor(index / 10) * 100;

        const position = 
          node.x !== undefined && 
          node.y !== undefined && 
          !isNaN(node.x) && 
          !isNaN(node.y) &&
          typeof node.x === 'number' &&
          typeof node.y === 'number'
            ? { x: node.x, y: node.y }
            : { x: defaultX, y: defaultY };

        return {
          id: String(node.id),
          data: {
            label: String(node.label || node.id),
            clusterId: typeof node.cluster_id === 'number' ? node.cluster_id : -1,
          },
          position,
          style: {
            background: getClusterColor(typeof node.cluster_id === 'number' ? node.cluster_id : -1),
            color: "#fff",
            border: "2px solid #1e293b",
            borderRadius: "8px",
            padding: "8px 12px",
            fontSize: "12px",
            fontWeight: 500,
          },
        };
      })
      .filter((node): node is Node => node !== null && node !== undefined);
  }, [nodes]);

  const flowEdges: Edge[] = useMemo(() => {
    // Filter out invalid edges (missing source/target)
    const validEdges = filteredEdges.filter(
      (edge) => edge && edge.source && edge.target
    );

    return validEdges.map((edge, idx) => {
      const edgeStyle: Partial<Edge["style"]> = {
        strokeWidth: Math.max(1, Math.min(3, (edge.weight || 1) * 2)),
        opacity: 0.6,
      };

      // Color edges by layer (ensure layer is a number)
      const layer = typeof edge.layer === 'number' ? edge.layer : parseInt(String(edge.layer), 10);
      
      if (layer === 2) {
        edgeStyle.stroke = "#10b981"; // Green for cluster
      } else if (layer === 3) {
        edgeStyle.stroke = "#f59e0b"; // Orange for semantic
      } else {
        edgeStyle.stroke = "#64748b"; // Gray for unknown
      }

      return {
        id: `edge-${idx}`,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        animated: layer === 3, // Animate semantic edges
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeStyle.stroke || "#64748b",
        },
        style: edgeStyle,
      };
    });
  }, [filteredEdges]);

  const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [reactFlowEdges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Update nodes/edges when props change
  React.useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes, setNodes]);

  React.useEffect(() => {
    setEdges(flowEdges);
  }, [flowEdges, setEdges]);

  const onNodeClickHandler = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        const originalNode = nodes.find((n) => n.id === node.id);
        if (originalNode) {
          onNodeClick(originalNode);
        }
      }
    },
    [onNodeClick, nodes]
  );

  // Don't render if no valid nodes
  if (!reactFlowNodes || reactFlowNodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center border border-slate-800 rounded-lg bg-slate-900/50">
        <p className="text-sm text-slate-400">No graph nodes to display</p>
      </div>
    );
  }

  // Ensure all nodes have valid positions before rendering
  const safeNodes = reactFlowNodes.filter((node) => {
    return node && node.id && node.position && 
           typeof node.position.x === 'number' && 
           typeof node.position.y === 'number' &&
           !isNaN(node.position.x) && 
           !isNaN(node.position.y);
  });

  if (safeNodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center border border-slate-800 rounded-lg bg-slate-900/50">
        <p className="text-sm text-slate-400">No valid graph nodes to display</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full border border-slate-800 rounded-lg bg-slate-900/50">
      <ReactFlow
        nodes={safeNodes}
        edges={reactFlowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClickHandler}
        connectionMode={ConnectionMode.Loose}
        fitView
        attributionPosition="bottom-left"
      >
        <Background color="#1e293b" gap={16} />
        <Controls className="bg-slate-800 border-slate-700" />
        <MiniMap
          className="bg-slate-800 border-slate-700"
          nodeColor={(node) => {
            try {
              const clusterId = (node.data as { clusterId?: number })?.clusterId ?? -1;
              return getClusterColor(clusterId);
            } catch {
              return "#64748b";
            }
          }}
        />
      </ReactFlow>
    </div>
  );
}

