import React from "react";
import { SearchResult } from "../lib/api";
import { FaExternalLinkAlt, FaLink, FaTag } from "react-icons/fa";

interface SearchResultCardProps {
  result: SearchResult;
  index: number;
}

export function SearchResultCard({ result, index }: SearchResultCardProps) {
  return (
    <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70 hover:bg-slate-900 transition-colors">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-3 mb-2">
            <h3 className="font-medium text-slate-100 flex-1 truncate">
              {result.title}
            </h3>
            <div className="flex items-center gap-2 flex-shrink-0">
              {result.cluster_id !== undefined && (
                <span className="px-2 py-1 rounded bg-sky-900/50 text-xs text-sky-300 border border-sky-700/50">
                  Cluster {result.cluster_id}
                </span>
              )}
              <span className="px-2 py-1 rounded bg-slate-800 text-xs text-slate-300">
                #{index + 1}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-slate-400 mb-3">
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
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition-colors"
            >
              <FaExternalLinkAlt className="text-[10px]" />
              Wikipedia
            </a>
            {result.wikidata_url && (
              <a
                href={result.wikidata_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition-colors"
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

