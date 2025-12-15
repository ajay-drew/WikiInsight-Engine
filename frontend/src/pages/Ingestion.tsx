import React, { useEffect, useState } from "react";
import { connectPipelineProgress, PipelineConfig, PipelineProgress, startPipeline } from "../lib/api";

export function IngestionPage() {
  const [queries, setQueries] = useState<string[]>(["Machine learning", "Artificial intelligence", "Data science"]);
  const [perQueryLimit, setPerQueryLimit] = useState<number>(50);
  const [maxArticles] = useState<number>(1000);
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  // Calculate total potential articles
  const totalPotential = queries.length * perQueryLimit;
  const actualMax = Math.min(totalPotential, maxArticles);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [eventSource]);

  function handleAddQuery() {
    if (queries.length < 6) {
      setQueries([...queries, ""]);
    }
  }

  function handleRemoveQuery(index: number) {
    if (queries.length > 3) {
      setQueries(queries.filter((_, i) => i !== index));
    }
  }

  function handleQueryChange(index: number, value: string) {
    const newQueries = [...queries];
    newQueries[index] = value;
    setQueries(newQueries);
  }

  function validateConfig(): string | null {
    if (queries.length < 3 || queries.length > 6) {
      return "Must have 3-6 seed queries";
    }
    if (queries.some(q => !q.trim())) {
      return "All queries must be non-empty";
    }
    if (perQueryLimit < 1 || perQueryLimit > 70) {
      return "Per-query limit must be between 1 and 70";
    }
    if (totalPotential > maxArticles) {
      return `Total potential articles (${totalPotential}) exceeds max (${maxArticles}). System will cap at ${maxArticles}.`;
    }
    return null;
  }

  async function handleStartPipeline() {
    const validationError = validateConfig();
    if (validationError) {
      setError(validationError);
      return;
    }

    setError(null);
    setIsRunning(true);

    try {
      // Start pipeline
      await startPipeline({
        seed_queries: queries.filter(q => q.trim()),
        per_query_limit: perQueryLimit,
        max_articles: maxArticles,
      });

      // Connect to SSE stream
      const es = connectPipelineProgress();
      setEventSource(es);

      es.onmessage = (event) => {
        try {
          const progressData = JSON.parse(event.data) as PipelineProgress;
          setProgress(progressData);

          // Check if pipeline is complete
          const stages = progressData.stages;
          const allDone = Object.values(stages).every(
            stage => stage.status === "completed" || stage.status === "error"
          );
          if (allDone) {
            setIsRunning(false);
            es.close();
            setEventSource(null);
          }
        } catch (err) {
          console.error("Failed to parse progress:", err);
        }
      };

      es.onerror = (err) => {
        console.error("SSE connection error:", err);
        setError("Lost connection to progress stream");
        setIsRunning(false);
        es.close();
        setEventSource(null);
      };
    } catch (err: any) {
      setError(err.message || "Failed to start pipeline");
      setIsRunning(false);
    }
  }

  function formatTime(seconds: number | null): string {
    if (seconds === null || seconds < 0) return "Calculating...";
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  }

  const currentStage = progress?.current_stage;
  const currentStageData = currentStage ? progress?.stages[currentStage as keyof typeof progress.stages] : null;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold mb-2">Pipeline Configuration</h1>
        <p className="text-sm text-slate-400">
          Configure and start the data pipeline. Set 3-6 seed queries, per-query article limit (up to 70),
          and the system will fetch articles (capped at 1000 total).
        </p>
      </section>

      {/* Configuration Form */}
      <div className="border border-slate-800 rounded-lg p-6 bg-slate-900/70 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Seed Queries ({queries.length}/6)
          </label>
          <div className="space-y-2">
            {queries.map((query, index) => (
              <div key={index} className="flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => handleQueryChange(index, e.target.value)}
                  placeholder={`Query ${index + 1} (e.g. Machine learning)`}
                  className="flex-1 px-3 py-2 rounded bg-slate-900 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500"
                  disabled={isRunning}
                />
                {queries.length > 3 && (
                  <button
                    onClick={() => handleRemoveQuery(index)}
                    className="px-3 py-2 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 text-sm"
                    disabled={isRunning}
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
          </div>
          {queries.length < 6 && (
            <button
              onClick={handleAddQuery}
              className="mt-2 px-3 py-1.5 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 text-sm"
              disabled={isRunning}
            >
              + Add Query
            </button>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Per-Query Limit: {perQueryLimit} articles
          </label>
          <input
            type="range"
            min="1"
            max="70"
            value={perQueryLimit}
            onChange={(e) => setPerQueryLimit(parseInt(e.target.value))}
            className="w-full"
            disabled={isRunning}
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>1</span>
            <span>70</span>
          </div>
        </div>

        <div className="border-t border-slate-800 pt-4">
          <div className="flex justify-between items-center text-sm">
            <span className="text-slate-400">Total Potential Articles:</span>
            <span className="font-mono text-sky-400">
              {totalPotential > maxArticles ? (
                <>
                  <span className="text-yellow-400">{totalPotential}</span> → {maxArticles} (capped)
                </>
              ) : (
                totalPotential
              )}
            </span>
          </div>
        </div>

        {error && (
          <div className="border border-red-500/50 rounded p-3 bg-red-500/10">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        <button
          onClick={handleStartPipeline}
          disabled={isRunning || !!validateConfig()}
          className="w-full px-4 py-2.5 rounded-lg bg-sky-500 text-white font-medium disabled:opacity-60 disabled:cursor-not-allowed hover:bg-sky-600 transition-colors"
        >
          {isRunning ? "Pipeline Running..." : "Start Pipeline"}
        </button>
      </div>

      {/* Progress Display */}
      {progress && (
        <div className="border border-slate-800 rounded-lg p-6 bg-slate-900/70 space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-lg font-semibold">Pipeline Progress</h2>
              <span className="text-sm font-mono text-sky-400">
                {progress.overall_progress.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2.5">
              <div
                className="bg-sky-500 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progress.overall_progress}%` }}
              />
            </div>
          </div>

          {currentStageData && (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium capitalize">
                    {currentStage?.replace("_", " ")}: {currentStageData.message}
                  </span>
                  {currentStageData.eta !== null && (
                    <span className="text-xs text-slate-400">
                      ETA: {formatTime(currentStageData.eta)}
                    </span>
                  )}
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                    className="bg-sky-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentStageData.progress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Stage Status */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-800">
            {Object.entries(progress.stages).map(([stage, stageData]) => (
              <div
                key={stage}
                className={`p-3 rounded border ${
                  stageData.status === "completed"
                    ? "border-green-500/50 bg-green-500/10"
                    : stageData.status === "running"
                    ? "border-sky-500/50 bg-sky-500/10"
                    : stageData.status === "error"
                    ? "border-red-500/50 bg-red-500/10"
                    : "border-slate-800 bg-slate-800/50"
                }`}
              >
                <div className="text-xs font-medium capitalize mb-1">{stage.replace("_", " ")}</div>
                <div className="text-xs text-slate-400">
                  {stageData.status === "completed" && "✓ Complete"}
                  {stageData.status === "running" && "⟳ Running"}
                  {stageData.status === "error" && "✗ Error"}
                  {stageData.status === "pending" && "○ Pending"}
                </div>
                {stageData.progress > 0 && (
                  <div className="text-xs text-slate-500 mt-1">
                    {stageData.progress.toFixed(0)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

