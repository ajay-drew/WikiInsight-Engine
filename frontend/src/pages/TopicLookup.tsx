import React, { useState } from "react";
import { lookupTopic, TopicLookupResponse } from "../lib/api";
import { EmptyState } from "../components/EmptyState";
import { Skeleton } from "../components/Skeleton";
import { usePersistentState } from "../hooks/usePersistentState";
import { useToast } from "../hooks/useToast";

export function TopicLookupPage() {
  const [state, setState] = usePersistentState<{
    title: string;
    result: TopicLookupResponse | null;
    error: string | null;
  }>("topicLookupState", {
    title: "",
    result: null,
    error: null,
  });
  const [loading, setLoading] = useState(false);
  const { title, result } = state;
  const toast = useToast();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setState((prev) => ({ ...prev, result: null }));
    if (!title.trim()) {
      toast.showWarning("Please enter an article title.");
      return;
    }
    setLoading(true);
    try {
      const data = await lookupTopic(title.trim());
      setState((prev) => ({ ...prev, result: data }));
      toast.showSuccess("Topic lookup completed successfully.");
    } catch (err: any) {
      toast.showError(err.message || "Failed to look up topic.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Topic Lookup</h1>
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          Enter a Wikipedia article title to see its topic cluster, distinctive topic words, and
          similar articles. The topic words are terms that show up a lot in this cluster but much
          less in other clusters.
        </p>
      </section>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 max-w-xl">
        <input
          type="text"
          value={title}
          onChange={(e) => setState((prev) => ({ ...prev, title: e.target.value }))}
          placeholder="e.g. Machine learning"
          className="px-3 py-2 rounded text-sm focus:outline-none focus:ring-2 transition-all"
          style={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
          onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
          onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
        />
        <div className="flex gap-2 items-center">
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 rounded bg-sky-500 text-white text-sm font-medium disabled:opacity-60"
          >
            {loading ? "Looking up..." : "Analyze"}
          </button>
        </div>
      </form>

      {loading && (
        <div 
          className="border rounded-lg p-4 space-y-3"
          style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
        >
          <Skeleton variant="text" width="40%" height={24} />
          <Skeleton variant="text" width="60%" height={16} />
          <Skeleton variant="text" width="100%" height={16} />
          <Skeleton variant="text" width="80%" height={16} />
        </div>
      )}

      {result && (
        <section className="mt-4 space-y-4">
          <div 
            className="border rounded-lg p-4"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
          >
            <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{result.article_title}</h2>
            <p className="text-sm mb-2" style={{ color: 'var(--text-tertiary)' }}>
              Cluster ID:{" "}
              <span className="font-mono" style={{ color: 'var(--accent)' }}>
                {result.cluster_id !== null ? result.cluster_id : "Unknown"}
              </span>
            </p>
            {result.keywords && result.keywords.length > 0 && (
              <div className="mb-3">
                <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Cluster Keywords</h3>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {result.keywords.slice(0, 20).join(", ")}
                </p>
              </div>
            )}
            {result.similar_articles.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Similar Articles</h3>
                <ul className="text-xs list-disc list-inside space-y-0.5" style={{ color: 'var(--text-secondary)' }}>
                  {result.similar_articles.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}

      {!result && !loading && (
        <EmptyState
          icon="search"
          title="Enter an article title to analyze"
          description="Enter a Wikipedia article title above to see its topic cluster, distinctive keywords, and similar articles."
        />
      )}
    </div>
  );
}


