import React from "react";
import { FaInbox, FaSearch, FaDatabase, FaChartLine } from "react-icons/fa";

interface EmptyStateProps {
  icon?: "inbox" | "search" | "database" | "chart";
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}

const icons = {
  inbox: FaInbox,
  search: FaSearch,
  database: FaDatabase,
  chart: FaChartLine,
};

export function EmptyState({ 
  icon = "inbox", 
  title, 
  description, 
  actionLabel, 
  onAction 
}: EmptyStateProps) {
  const Icon = icons[icon];

  return (
    <div 
      className="border rounded-lg p-12 text-center"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
    >
      <div className="flex justify-center mb-4">
        <Icon className="text-6xl" style={{ color: 'var(--text-tertiary)' }} />
      </div>
      <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>{title}</h3>
      <p className="text-sm mb-6 max-w-md mx-auto" style={{ color: 'var(--text-tertiary)' }}>{description}</p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="px-4 py-2 rounded-lg text-white text-sm font-medium transition-colors"
          style={{ backgroundColor: 'var(--accent)' }}
          onMouseEnter={(e) => e.currentTarget.style.opacity = '0.9'}
          onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}

