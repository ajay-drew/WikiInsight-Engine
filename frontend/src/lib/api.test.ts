import { describe, it, expect, vi, afterEach } from "vitest";
import { lookupTopic } from "./api";

describe("api client", () => {
  afterEach(() => {
    // reset any mocks between tests
    vi.restoreAllMocks();
  });

  it("calls /api/topics/lookup with the right payload", async () => {
    const mockResponse = {
      article_title: "Example",
      cluster_id: 0,
      similar_articles: ["Other"],
      keywords: ["kw"],
      explanation: {}
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    } as Response);

    // @ts-expect-error override global fetch for test
    global.fetch = fetchMock;

    const result = await lookupTopic("Example");
    expect(result.article_title).toBe("Example");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/topics/lookup");
    expect((options as RequestInit).method).toBe("POST");
  });
});


