import React, { useState } from "react";
import { searchArticles, SearchResult } from "../lib/api";
import { SearchResultCard } from "../components/SearchResultCard";
import { usePersistentState } from "../hooks/usePersistentState";

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
  const { query, results, hasSearched, error } = state;

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setState((prev) => ({ ...prev, error: null, results: [], hasSearched: false }));

    if (!query.trim()) {
      setState((prev) => ({ ...prev, error: "Please enter a search query." }));
      return;
    }

    setLoading(true);
    try {
      const data = await searchArticles(query.trim(), 20);
      setState((prev) => ({ ...prev, results: data.results, hasSearched: true }));
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        error: err.message || "Failed to perform search.",
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
        <p className="text-sm text-slate-400">
          Search Wikipedia articles using semantic (vector) and keyword (BM25) search combined with Reciprocal Rank Fusion.
          Find articles by meaning, not just exact keywords.
        </p>
      </section>

      <form onSubmit={handleSearch} className="flex flex-col gap-3 max-w-2xl">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setState((prev) => ({ ...prev, query: e.target.value }))}
            placeholder="e.g. machine learning algorithms, cooking pasta recipes, space exploration..."
            className="flex-1 px-4 py-2.5 rounded-lg bg-slate-900 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-2.5 rounded-lg bg-sky-500 text-white text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed hover:bg-sky-600 transition-colors"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
      </form>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          <span>Searching articles...</span>
        </div>
      )}

      {hasSearched && !loading && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Search Results
              {results.length > 0 && (
                <span className="ml-2 text-sm font-normal text-slate-400">
                  ({results.length} {results.length === 1 ? "result" : "results"})
                </span>
              )}
            </h2>
          </div>

          {results.length === 0 ? (
            <div className="border border-slate-800 rounded-lg p-8 bg-slate-900/70 text-center">
              <p className="text-slate-400">No results found for "{query}"</p>
              <p className="text-xs text-slate-500 mt-2">
                Try different keywords or a more general query.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((result, idx) => (
                <SearchResultCard key={`${result.title}-${idx}`} result={result} index={idx} />
              ))}
            </div>
          )}
        </section>
      )}

      {!hasSearched && !loading && (
        <div className="border border-slate-800 rounded-lg p-8 bg-slate-900/70">
          <p className="text-sm text-slate-400 text-center">
            Enter a search query above to find relevant Wikipedia articles.
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="p-3 rounded bg-slate-800/50">
              <p className="font-medium text-slate-300 mb-1">Semantic Search</p>
              <p className="text-slate-500">Finds articles by meaning and context</p>
            </div>
            <div className="p-3 rounded bg-slate-800/50">
              <p className="font-medium text-slate-300 mb-1">Keyword Search</p>
              <p className="text-slate-500">Finds articles with exact term matches</p>
            </div>
            <div className="p-3 rounded bg-slate-800/50">
              <p className="font-medium text-slate-300 mb-1">Hybrid Fusion</p>
              <p className="text-slate-500">Combines both for best results</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

