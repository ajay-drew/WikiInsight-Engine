import React, { useState, useRef } from "react";
import { searchArticles, SearchResult } from "../lib/api";
import { SearchResultCard } from "../components/SearchResultCard";
import { SearchResultSkeleton } from "../components/SearchResultSkeleton";
import { EmptyState } from "../components/EmptyState";
import { Tooltip } from "../components/Tooltip";
import { FaDownload, FaTimes } from "react-icons/fa";
import { usePersistentState } from "../hooks/usePersistentState";
import { useToast } from "../hooks/useToast";
import { exportToCSV, exportToJSON } from "../utils/export";

export function SearchPage() {
  const [state, setState] = usePersistentState<{
    query: string;
    results: SearchResult[];
    hasSearched: boolean;
    error: string | null;
  }>("searchPageState", {
    query: "",
    results: [],
    hasSearched: false,
    error: null,
  });
  const [loading, setLoading] = useState(false);
  const { query, results, hasSearched } = state;
  const toast = useToast();
  const searchInputRef = useRef<HTMLInputElement>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setState((prev) => ({ ...prev, results: [], hasSearched: false }));

    if (!query.trim()) {
      toast.showWarning("Please enter a search query.");
      return;
    }

    setLoading(true);
    try {
      const data = await searchArticles(query.trim(), 20);
      setState((prev) => ({ ...prev, results: data.results, hasSearched: true }));
      if (data.results.length > 0) {
        toast.showSuccess(`Found ${data.results.length} result${data.results.length === 1 ? "" : "s"}`);
      }
    } catch (err: any) {
      toast.showError(err.message || "Failed to perform search.");
      setState((prev) => ({
        ...prev,
        results: [],
      }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Hybrid Search</h1>
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          Search Wikipedia articles using semantic (vector) and keyword (BM25) search combined with Reciprocal Rank Fusion.
          Find articles by meaning, not just exact keywords.
        </p>
      </section>

      <form onSubmit={handleSearch} className="flex flex-col gap-3 max-w-2xl">
        <div className="relative">
          <div className="absolute -top-6 right-0 text-xs" style={{ color: 'var(--text-tertiary)' }}>
            Press <kbd className="px-1.5 py-0.5 rounded border" style={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)' }}>/</kbd> to focus
          </div>
        </div>
        <div className="flex gap-2 relative">
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(e) => setState((prev) => ({ ...prev, query: e.target.value }))}
            placeholder="e.g. machine learning algorithms, cooking pasta recipes, space exploration... (Press / to focus)"
            className="flex-1 px-4 py-2.5 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all pr-10"
            style={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
            disabled={loading}
          />
          {query && (
            <button
              type="button"
              onClick={() => setState((prev) => ({ ...prev, query: "" }))}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 transition-colors"
              style={{ color: 'var(--text-tertiary)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-tertiary)'}
              aria-label="Clear search"
            >
              <FaTimes className="w-4 h-4" />
            </button>
          )}
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-2.5 rounded-lg bg-sky-500 text-white text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed hover:bg-sky-600 transition-colors"
          >
            {loading ? "Searching..." : "Search"}
            </button>
        </div>
      </form>

      {loading && (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <SearchResultSkeleton key={i} />
          ))}
        </div>
      )}

      {hasSearched && !loading && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              Search Results
              {results.length > 0 && (
                <span className="px-2 py-0.5 rounded-full text-xs font-medium" style={{ backgroundColor: 'rgba(14, 165, 233, 0.2)', color: 'rgb(125, 211, 252)' }}>
                  {results.length}
                </span>
              )}
            </h2>
            {results.length > 0 && (
              <div className="flex gap-2">
                <Tooltip content="Export results as CSV">
                  <button
                    onClick={() => {
                      try {
                        const exportData = results.map((r) => ({
                          title: r.title,
                          score: r.score,
                          rank: r.rank + 1,
                          cluster_id: r.cluster_id ?? "",
                          categories: r.categories.join("; "),
                          link_count: r.link_count,
                          wikipedia_url: r.wikipedia_url,
                        }));
                        exportToCSV(exportData, "search_results");
                        toast.showSuccess("Search results exported as CSV");
                      } catch (err: any) {
                        toast.showError(err.message || "Failed to export results");
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
                <Tooltip content="Export results as JSON">
                  <button
                    onClick={() => {
                      try {
                        exportToJSON(results, "search_results");
                        toast.showSuccess("Search results exported as JSON");
                      } catch (err: any) {
                        toast.showError(err.message || "Failed to export results");
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

          {results.length === 0 ? (
            <EmptyState
              icon="search"
              title={`No results found for "${query}"`}
              description="Try different keywords or a more general query. Consider using synonyms or broader terms."
            />
          ) : (
            <div className="space-y-3">
              {results.map((result, idx) => (
                <SearchResultCard 
                  key={`${result.title}-${idx}`} 
                  result={result} 
                  index={idx} 
                  query={query}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {!hasSearched && !loading && (
        <div className="text-center py-12">
          <EmptyState
            icon="search"
            title="Start searching Wikipedia articles"
            description="Enter a search query above to find relevant articles using semantic and keyword search combined with Reciprocal Rank Fusion."
          />
          <div className="mt-6 text-sm" style={{ color: 'var(--text-tertiary)' }}>
            <p className="mb-2">Try searching for:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              {["machine learning", "space exploration", "cooking recipes", "history"].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setState((prev) => ({ ...prev, query: suggestion }));
                    searchInputRef.current?.focus();
                  }}
                  className="px-3 py-1 rounded transition-colors"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

