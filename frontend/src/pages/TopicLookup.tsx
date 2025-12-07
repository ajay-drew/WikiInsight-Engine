import React, { useState } from "react";
import { lookupTopic, TopicLookupResponse } from "../lib/api";

export function TopicLookupPage() {
  const [title, setTitle] = useState("");
  const [result, setResult] = useState<TopicLookupResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!title.trim()) {
      setError("Please enter an article title.");
      return;
    }
    setLoading(true);
    try {
      const data = await lookupTopic(title.trim());
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to look up topic.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Topic Lookup</h1>
        <p className="text-sm text-slate-400">
          Enter a Wikipedia article title to see its topic cluster, distinctive topic words, and
          similar articles. The topic words are terms that show up a lot in this cluster but much
          less in other clusters.
        </p>
      </section>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 max-w-xl">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Machine learning"
          className="px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500"
        />
        <div className="flex gap-2 items-center">
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 rounded bg-sky-500 text-white text-sm font-medium disabled:opacity-60"
          >
            {loading ? "Looking up..." : "Analyze"}
          </button>
          {error && <span className="text-xs text-red-400">{error}</span>}
        </div>
      </form>

      {result && (
        <section className="mt-4 space-y-4">
          <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70">
            <h2 className="text-lg font-semibold mb-1">{result.article_title}</h2>
            <p className="text-sm text-slate-400 mb-2">
              Cluster ID:{" "}
              <span className="font-mono text-sky-400">
                {result.cluster_id !== null ? result.cluster_id : "Unknown"}
              </span>
            </p>
            {result.keywords && result.keywords.length > 0 && (
              <div className="mb-3">
                <h3 className="text-sm font-medium mb-1">Cluster Keywords</h3>
                <p className="text-xs text-slate-300">
                  {result.keywords.slice(0, 20).join(", ")}
                </p>
              </div>
            )}
            {result.similar_articles.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-1">Similar Articles</h3>
                <ul className="text-xs text-slate-300 list-disc list-inside space-y-0.5">
                  {result.similar_articles.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}


