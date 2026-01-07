import React, { useEffect, useState, useRef } from "react";
import { connectPipelineProgress, PipelineConfig, PipelineProgress, startPipeline, getPipelineLogs, deletePipelineLogs, PipelineLogEntry } from "../lib/api";
import { usePersistentState } from "../hooks/usePersistentState";
import { useToast } from "../hooks/useToast";

export function IngestionPage() {
  const [config, setConfig] = usePersistentState<{
    queries: string[];
    perQueryLimit: number;
    maxArticles: number;
  }>("ingestionConfig", {
    queries: ["Machine learning", "Artificial intelligence", "Data science"],
    perQueryLimit: 50,
    maxArticles: 1000,
  });
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const toast = useToast();
  const [logs, setLogs] = useState<PipelineLogEntry[]>([]);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  // Calculate total potential articles
  const totalPotential = config.queries.length * config.perQueryLimit;
  const actualMax = Math.min(totalPotential, config.maxArticles);

  useEffect(() => {
    // Delete all logs on page load (webpage reload)
    deletePipelineLogs().catch(() => {
      // Ignore errors - logs might not exist yet
    });
    
    // Cleanup on unmount
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, []); // Only run on mount

  // Poll for logs while pipeline is running
  useEffect(() => {
    if (!isRunning) return;

    const pollLogs = async () => {
      try {
        const logsResponse = await getPipelineLogs();
        if (logsResponse.run_id) {
          setCurrentRunId(logsResponse.run_id);
          setLogs(logsResponse.logs);
        }
      } catch (err) {
        console.debug("Failed to fetch logs:", err);
      }
    };

    // Poll immediately and then every 2 seconds
    pollLogs();
    const interval = setInterval(pollLogs, 2000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // Auto-scroll logs container to bottom when new logs arrive (only scrolls the container, not the page)
  useEffect(() => {
    if (logs.length > 0 && logsContainerRef.current) {
      const container = logsContainerRef.current;
      // Scroll to bottom of the container, not the page
      container.scrollTop = container.scrollHeight;
    }
  }, [logs]);

  function handleAddQuery() {
    if (config.queries.length < 6) {
      setConfig((prev) => ({ ...prev, queries: [...prev.queries, ""] }));
    }
  }

  function handleRemoveQuery(index: number) {
    if (config.queries.length > 3) {
      setConfig((prev) => ({
        ...prev,
        queries: prev.queries.filter((_, i) => i !== index),
      }));
    }
  }

  function handleQueryChange(index: number, value: string) {
    setConfig((prev) => {
      const newQueries = [...prev.queries];
      newQueries[index] = value;
      return { ...prev, queries: newQueries };
    });
  }

  function validateConfig(): string | null {
    if (config.queries.length < 3 || config.queries.length > 6) {
      return "Must have 3-6 seed queries";
    }
    if (config.queries.some((q) => !q.trim())) {
      return "All queries must be non-empty";
    }
    if (config.perQueryLimit < 1 || config.perQueryLimit > 70) {
      return "Per-query limit must be between 1 and 70";
    }
    if (config.maxArticles < 50) {
      return "Max articles must be at least 50 for meaningful clustering. Please increase max articles or adjust your seed queries/per-query limit.";
    }
    if (totalPotential > config.maxArticles) {
      return `Total potential articles (${totalPotential}) exceeds max (${config.maxArticles}). System will cap at ${config.maxArticles}.`;
    }
    return null;
  }

  async function handleStartPipeline() {
    const validationError = validateConfig();
    if (validationError) {
      toast.showError(validationError);
      return;
    }

    setIsRunning(true);

    try {
      // Start pipeline
      await startPipeline({
        seed_queries: config.queries.filter((q) => q.trim()),
        per_query_limit: config.perQueryLimit,
        max_articles: config.maxArticles,
      });

      // Connect to SSE stream
      const es = connectPipelineProgress();
      setEventSource(es);

      es.onopen = () => {
        console.log("SSE connection opened");
        toast.showSuccess("Pipeline started successfully");
      };

      es.onmessage = async (event) => {
        try {
          // Ignore keep-alive comments
          if (event.data.trim() === "" || event.data.startsWith(":")) {
            return;
          }

          const progressData = JSON.parse(event.data) as PipelineProgress;
          
          // Handle special event types
          if (progressData.type === "reload") {
            console.log("Data reload:", progressData.message);
            // Don't update progress, just log the reload message
            return;
          }

          if (progressData.type === "error") {
            console.warn("Stream error:", progressData.message);
            // Don't close connection on stream errors - allow it to recover
            return;
          }

          setProgress(progressData);

          // Fetch logs from database
          try {
            const logsResponse = await getPipelineLogs();
            if (logsResponse.run_id) {
              setCurrentRunId(logsResponse.run_id);
              setLogs(logsResponse.logs);
            }
          } catch (logErr) {
            // Ignore log fetch errors
            console.debug("Failed to fetch logs:", logErr);
          }

          // Check if pipeline is complete
          const stages = progressData.stages;
          const allDone = Object.values(stages).every(
            stage => stage.status === "completed" || stage.status === "error"
          );
          if (allDone) {
            setIsRunning(false);
            // Wait a moment before closing to ensure final message is received
            setTimeout(() => {
              es.close();
              setEventSource(null);
            }, 1000);
            // Final log fetch
            try {
              const logsResponse = await getPipelineLogs();
              if (logsResponse.run_id) {
                setLogs(logsResponse.logs);
              }
            } catch (logErr) {
              console.debug("Failed to fetch final logs:", logErr);
            }
          }
        } catch (err) {
          console.error("Failed to parse progress:", err);
          // Don't close connection on parse errors - might be temporary
        }
      };

      es.onerror = (err) => {
        console.error("SSE connection error:", err);
        // EventSource automatically attempts to reconnect
        // Only show error if connection is actually closed and not recovering
        if (es.readyState === EventSource.CLOSED) {
          toast.showError("Lost connection to progress stream. EventSource will attempt to reconnect automatically.");
          // EventSource will automatically try to reconnect, so we don't need manual reconnection
          // Just update the error message to be less alarming
          setTimeout(() => {
            if (es.readyState === EventSource.CONNECTING || es.readyState === EventSource.OPEN) {
              // Connection recovered
            }
          }, 5000);
        } else if (es.readyState === EventSource.CONNECTING) {
          // Connection is reconnecting
        }
      };
    } catch (err: any) {
      toast.showError(err.message || "Failed to start pipeline");
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
        <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
          Configure and start the data pipeline. Set 3-6 seed queries, per-query article limit (up to 70),
          and the system will fetch articles (capped at 1000 total).
        </p>
      </section>

      {/* Configuration Form */}
      <div 
        className="border rounded-lg p-6 space-y-4"
        style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
      >
        <div>
          <label className="block text-sm font-medium mb-2">
            Seed Queries ({config.queries.length}/6)
          </label>
          <div className="space-y-2">
            {config.queries.map((query, index) => (
              <div key={index} className="flex gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => handleQueryChange(index, e.target.value)}
                  placeholder={`Query ${index + 1} (e.g. Machine learning)`}
                  className="flex-1 px-3 py-2 rounded text-sm focus:outline-none focus:ring-2 transition-all"
                  style={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                  onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
                  onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
                  disabled={isRunning}
                />
                {config.queries.length > 3 && (
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
          {config.queries.length < 6 && (
            <button
              onClick={handleAddQuery}
              className="mt-2 px-3 py-1.5 rounded text-sm transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
              disabled={isRunning}
            >
              + Add Query
            </button>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Per-Query Limit: {config.perQueryLimit} articles
          </label>
          <input
            type="range"
            min="1"
            max="70"
            value={config.perQueryLimit}
            onChange={(e) => setConfig((prev) => ({ ...prev, perQueryLimit: parseInt(e.target.value) }))}
            className="w-full"
            disabled={isRunning}
          />
          <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
            <span>1</span>
            <span>70</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Max Articles: {config.maxArticles}
          </label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={config.maxArticles}
            onChange={(e) => setConfig((prev) => ({ ...prev, maxArticles: parseInt(e.target.value) }))}
            className="w-full"
            disabled={isRunning}
          />
          <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
            <span>10</span>
            <span>1000</span>
          </div>
          <p className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Maximum total articles to fetch (hard cap: 1000)
          </p>
        </div>

        <div className="border-t pt-4" style={{ borderColor: 'var(--border-color)' }}>
          <div className="flex justify-between items-center text-sm">
            <span style={{ color: 'var(--text-tertiary)' }}>Total Potential Articles:</span>
            <span className="font-mono text-sky-400">
              {totalPotential > config.maxArticles ? (
                <>
                  <span className="text-yellow-400">{totalPotential}</span> → {config.maxArticles} (capped)
                </>
              ) : (
                totalPotential
              )}
            </span>
          </div>
        </div>

        {validateConfig() && (
          <div className="border rounded-lg p-3 text-sm" style={{ borderColor: 'rgba(234, 179, 8, 0.5)', backgroundColor: 'rgba(234, 179, 8, 0.1)', color: 'rgb(250, 204, 21)' }}>
            ⚠️ {validateConfig()}
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
        <div 
          className="border rounded-lg p-6 space-y-4"
          style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
        >
          <div>
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Pipeline Progress</h2>
              <span className="text-sm font-mono" style={{ color: 'var(--accent)' }}>
                {progress.overall_progress.toFixed(1)}%
              </span>
            </div>
            <div className="w-full rounded-full h-2.5" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
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
                    <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                      ETA: {formatTime(currentStageData.eta)}
                    </span>
                  )}
                </div>
                <div className="w-full rounded-full h-2" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
                  <div
                    className="bg-sky-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentStageData.progress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Stage Status */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t" style={{ borderColor: 'var(--border-color)' }}>
            {Object.entries(progress.stages).map(([stage, stageData]) => {
              const statusStyles = {
                completed: { borderColor: 'rgba(16, 185, 129, 0.5)', backgroundColor: 'rgba(16, 185, 129, 0.1)' },
                running: { borderColor: 'rgba(14, 165, 233, 0.5)', backgroundColor: 'rgba(14, 165, 233, 0.1)' },
                error: { borderColor: 'rgba(239, 68, 68, 0.5)', backgroundColor: 'rgba(239, 68, 68, 0.1)' },
                pending: { borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-tertiary)' },
              };
              const style = statusStyles[stageData.status as keyof typeof statusStyles] || statusStyles.pending;
              return (
                <div
                  key={stage}
                  className="p-3 rounded border"
                  style={style}
                >
                  <div className="text-xs font-medium capitalize mb-1" style={{ color: 'var(--text-primary)' }}>{stage.replace("_", " ")}</div>
                  <div className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                    {stageData.status === "completed" && "✓ Complete"}
                    {stageData.status === "running" && "⟳ Running"}
                    {stageData.status === "error" && "✗ Error"}
                    {stageData.status === "pending" && "○ Pending"}
                  </div>
                  {stageData.progress > 0 && (
                    <div className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
                      {stageData.progress.toFixed(0)}%
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Pipeline Logs Section */}
      {(isRunning || logs.length > 0) && (
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Pipeline Logs</h2>
            {currentRunId && (
              <span className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>Run ID: {currentRunId.substring(0, 8)}...</span>
            )}
          </div>
          <div 
            ref={logsContainerRef}
            className="border rounded-lg p-4 max-h-[400px] overflow-auto"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
          >
            {logs.length === 0 ? (
              <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No logs available yet...</p>
            ) : (
              <div className="space-y-1 font-mono text-xs">
                {logs.map((log, idx) => (
                  <div
                    key={idx}
                    style={{
                      color: log.log_level === "ERROR" 
                        ? "#ef4444" 
                        : log.log_level === "WARNING"
                        ? "#f59e0b"
                        : log.log_level === "DEBUG"
                        ? 'var(--text-tertiary)'
                        : 'var(--text-secondary)'
                    }}
                  >
                    <span style={{ color: 'var(--text-tertiary)' }}>
                      [{new Date(log.timestamp).toLocaleTimeString()}]
                    </span>
                    {log.stage_name && (
                      <span className="ml-2" style={{ color: 'var(--accent)' }}>[{log.stage_name}]</span>
                    )}
                    <span className={`ml-2 ${
                      log.log_level === "ERROR" ? "font-bold" : ""
                    }`}>
                      {log.message}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

