import React from "react";
import { SearchResult } from "../lib/api";
import { FaExternalLinkAlt, FaLink, FaTag } from "react-icons/fa";
import { HighlightedText } from "../utils/highlight";

interface SearchResultCardProps {
  result: SearchResult;
  index: number;
  query?: string;
}

export function SearchResultCard({ result, index, query }: SearchResultCardProps) {
  return (
    <div 
      className="border rounded-lg p-4 transition-colors hover:scale-[1.01] hover:shadow-lg"
      style={{ 
        borderColor: 'var(--border-color)', 
        backgroundColor: 'var(--bg-secondary)',
      }}
      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-3 mb-2">
            <h3 className="font-medium flex-1 truncate" style={{ color: 'var(--text-primary)' }}>
              {query ? (
                <HighlightedText text={result.title} query={query} />
              ) : (
                result.title
              )}
            </h3>
            <div className="flex items-center gap-2 flex-shrink-0">
              {result.cluster_id !== undefined && (
                <span className="px-2 py-1 rounded bg-sky-900/50 text-xs text-sky-300 border border-sky-700/50">
                  Cluster {result.cluster_id}
                </span>
              )}
              <span 
                className="px-2 py-1 rounded text-xs"
                style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
              >
                #{index + 1}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
            <span>
              Score: <span className="font-mono text-sky-400">{result.score.toFixed(4)}</span>
            </span>
            <span>
              Rank: <span className="font-mono">{result.rank + 1}</span>
            </span>
            {result.link_count > 0 && (
              <span className="flex items-center gap-1">
                <FaLink className="text-slate-500" />
                {result.link_count} links
              </span>
            )}
          </div>

          {result.categories.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {result.categories.slice(0, 5).map((category, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-800/50 text-xs text-slate-300"
                >
                  <FaTag className="text-slate-500 text-[10px]" />
                  {category}
                </span>
              ))}
            </div>
          )}

          <div className="flex items-center gap-2">
            <a
              href={result.wikipedia_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
            >
              <FaExternalLinkAlt className="text-[10px]" />
              Wikipedia
            </a>
            {result.wikidata_url && (
              <a
                href={result.wikidata_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
              >
                <FaExternalLinkAlt className="text-[10px]" />
                Wikidata
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

