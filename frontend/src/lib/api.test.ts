import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { startPipeline, connectPipelineProgress, PipelineConfig } from "./api";

describe("api client", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("startPipeline", () => {
    it("sends correct request to backend", async () => {
      const mockConfig: PipelineConfig = {
        seed_queries: ["AI", "ML", "Deep Learning"],
        per_query_limit: 8,
        max_articles: 24,
      };

      const mockResponse = {
        status: "started",
        message: "Pipeline started successfully",
        config: mockConfig,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      // @ts-expect-error override global fetch for test
      global.fetch = fetchMock;

      const result = await startPipeline(mockConfig);

      expect(result).toEqual(mockResponse);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, options] = fetchMock.mock.calls[0];
      expect(url).toBe("/api/pipeline/start");
      expect((options as RequestInit).method).toBe("POST");
      expect((options as RequestInit).headers).toEqual({
        "Content-Type": "application/json",
      });
      expect(JSON.parse((options as RequestInit).body as string)).toEqual(mockConfig);
    });

    it("handles max_articles correctly", async () => {
      const mockConfig: PipelineConfig = {
        seed_queries: ["Test1", "Test2", "Test3"],
        per_query_limit: 8,
        max_articles: 24,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "started", message: "OK", config: mockConfig }),
      } as Response);

      // @ts-expect-error override global fetch for test
      global.fetch = fetchMock;

      await startPipeline(mockConfig);

      const requestBody = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
      expect(requestBody.max_articles).toBe(24);
    });

    it("throws error when request fails", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        text: async () => "Invalid configuration",
      } as Response);

      // @ts-expect-error override global fetch for test
      global.fetch = fetchMock;

      await expect(
        startPipeline({
          seed_queries: ["Test1", "Test2", "Test3"],
          per_query_limit: 8,
          max_articles: 24,
        })
      ).rejects.toThrow("Failed to start pipeline");
    });

    it("handles network errors", async () => {
      const fetchMock = vi.fn().mockRejectedValue(new Error("Network error"));

      // @ts-expect-error override global fetch for test
      global.fetch = fetchMock;

      await expect(
        startPipeline({
          seed_queries: ["Test1", "Test2", "Test3"],
          per_query_limit: 8,
          max_articles: 24,
        })
      ).rejects.toThrow();
    });
  });

  describe("connectPipelineProgress", () => {
    it("creates EventSource with correct URL", () => {
      // Mock EventSource
      const mockEventSource = {
        addEventListener: vi.fn(),
        close: vi.fn(),
      };

      // @ts-expect-error override global EventSource for test
      global.EventSource = vi.fn(() => mockEventSource) as any;

      const eventSource = connectPipelineProgress();

      expect(global.EventSource).toHaveBeenCalledWith("/api/pipeline/progress");
      expect(eventSource).toBe(mockEventSource);
    });
  });

  describe("lookupTopic", () => {
    it("calls /api/topics/lookup with the right payload", async () => {
      const mockResponse = {
        article_title: "Example",
        cluster_id: 0,
        similar_articles: ["Other"],
        keywords: ["kw"],
        explanation: {},
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      // @ts-expect-error override global fetch for test
      global.fetch = fetchMock;

      const { lookupTopic } = await import("./api");
      const result = await lookupTopic("Example");

      expect(result.article_title).toBe("Example");
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, options] = fetchMock.mock.calls[0];
      expect(url).toBe("/api/topics/lookup");
      expect((options as RequestInit).method).toBe("POST");
    });
  });
});
