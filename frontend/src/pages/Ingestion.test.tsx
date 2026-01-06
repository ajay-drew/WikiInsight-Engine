import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { IngestionPage } from "./Ingestion";
import * as api from "../lib/api";

// Mock the API module
vi.mock("../lib/api", () => ({
  startPipeline: vi.fn(),
  connectPipelineProgress: vi.fn(),
}));

describe("IngestionPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders pipeline configuration form", () => {
    render(<IngestionPage />);
    
    expect(screen.getByText("Pipeline Configuration")).toBeInTheDocument();
    expect(screen.getByText("Seed Queries")).toBeInTheDocument();
    expect(screen.getByText("Per-Query Limit")).toBeInTheDocument();
    expect(screen.getByText("Max Articles")).toBeInTheDocument();
    expect(screen.getByText("Start Pipeline")).toBeInTheDocument();
  });

  it("allows editing seed queries", () => {
    render(<IngestionPage />);
    
    const queryInputs = screen.getAllByPlaceholderText(/Query \d+/);
    expect(queryInputs.length).toBeGreaterThanOrEqual(3);
    
    const firstInput = queryInputs[0] as HTMLInputElement;
    fireEvent.change(firstInput, { target: { value: "New Query" } });
    expect(firstInput.value).toBe("New Query");
  });

  it("allows adding and removing queries", () => {
    render(<IngestionPage />);
    
    // Add a query
    const addButton = screen.getByText("+ Add Query");
    fireEvent.click(addButton);
    
    // Should have 4 queries now
    const queryInputs = screen.getAllByPlaceholderText(/Query \d+/);
    expect(queryInputs.length).toBe(4);
    
    // Remove a query (should only work if > 3)
    const removeButtons = screen.getAllByText("Remove");
    if (removeButtons.length > 0) {
      fireEvent.click(removeButtons[0]);
      const remainingInputs = screen.getAllByPlaceholderText(/Query \d+/);
      expect(remainingInputs.length).toBe(3);
    }
  });

  it("allows changing per-query limit", () => {
    render(<IngestionPage />);
    
    const slider = screen.getByLabelText(/Per-Query Limit/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    expect(slider).toBeInTheDocument();
    
    fireEvent.change(slider, { target: { value: "25" } });
    expect(slider.value).toBe("25");
    expect(screen.getByText(/Per-Query Limit: 25 articles/)).toBeInTheDocument();
  });

  it("allows changing max articles", () => {
    render(<IngestionPage />);
    
    const maxArticlesSlider = screen.getByLabelText(/Max Articles/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    expect(maxArticlesSlider).toBeInTheDocument();
    
    fireEvent.change(maxArticlesSlider, { target: { value: "24" } });
    expect(maxArticlesSlider.value).toBe("24");
    expect(screen.getByText(/Max Articles: 24/)).toBeInTheDocument();
  });

  it("validates configuration before starting pipeline", () => {
    render(<IngestionPage />);
    
    // Clear all queries to trigger validation
    const queryInputs = screen.getAllByPlaceholderText(/Query \d+/);
    queryInputs.forEach(input => {
      fireEvent.change(input, { target: { value: "" } });
    });
    
    const startButton = screen.getByText("Start Pipeline");
    expect(startButton).toBeDisabled();
  });

  it("calls startPipeline with correct config when form is submitted", async () => {
    const mockStartPipeline = vi.mocked(api.startPipeline);
    mockStartPipeline.mockResolvedValue({
      status: "started",
      message: "Pipeline started",
      config: {
        seed_queries: ["Test1", "Test2", "Test3"],
        per_query_limit: 8,
        max_articles: 24,
      },
    });

    const mockEventSource = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: Event) => void) | null,
      close: vi.fn(),
    };
    vi.mocked(api.connectPipelineProgress).mockReturnValue(mockEventSource as any);

    render(<IngestionPage />);
    
    // Set max articles to 24
    const maxArticlesSlider = screen.getByLabelText(/Max Articles/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    fireEvent.change(maxArticlesSlider, { target: { value: "24" } });
    
    // Set per-query limit to 8
    const perQuerySlider = screen.getByLabelText(/Per-Query Limit/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    fireEvent.change(perQuerySlider, { target: { value: "8" } });
    
    // Click start
    const startButton = screen.getByText("Start Pipeline");
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(mockStartPipeline).toHaveBeenCalledWith({
        seed_queries: expect.arrayContaining([expect.any(String)]),
        per_query_limit: 8,
        max_articles: 24,
      });
    });
  });

  it("shows error when pipeline start fails", async () => {
    const mockStartPipeline = vi.mocked(api.startPipeline);
    mockStartPipeline.mockRejectedValue(new Error("Failed to start"));

    render(<IngestionPage />);
    
    const startButton = screen.getByText("Start Pipeline");
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByText(/Failed to start/)).toBeInTheDocument();
    });
  });

  it("calculates total potential articles correctly", () => {
    render(<IngestionPage />);
    
    // Set 3 queries, 8 per query = 24 total potential
    const perQuerySlider = screen.getByLabelText(/Per-Query Limit/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    fireEvent.change(perQuerySlider, { target: { value: "8" } });
    
    // Should show total potential
    expect(screen.getByText(/Total Potential Articles/)).toBeInTheDocument();
  });

  it("shows capped message when total potential exceeds max articles", () => {
    render(<IngestionPage />);
    
    // Set max articles to 24
    const maxArticlesSlider = screen.getByLabelText(/Max Articles/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    fireEvent.change(maxArticlesSlider, { target: { value: "24" } });
    
    // Set per-query limit high enough to exceed max
    const perQuerySlider = screen.getByLabelText(/Per-Query Limit/).parentElement?.querySelector('input[type="range"]') as HTMLInputElement;
    fireEvent.change(perQuerySlider, { target: { value: "50" } });
    
    // Should show capped message
    expect(screen.getByText(/capped/)).toBeInTheDocument();
  });

  it("disables form when pipeline is running", async () => {
    const mockStartPipeline = vi.mocked(api.startPipeline);
    mockStartPipeline.mockResolvedValue({
      status: "started",
      message: "Pipeline started",
      config: {
        seed_queries: ["Test1", "Test2", "Test3"],
        per_query_limit: 8,
        max_articles: 24,
      },
    });

    const mockEventSource = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: Event) => void) | null,
      close: vi.fn(),
    };
    vi.mocked(api.connectPipelineProgress).mockReturnValue(mockEventSource as any);

    render(<IngestionPage />);
    
    const startButton = screen.getByText("Start Pipeline");
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByText("Pipeline Running...")).toBeInTheDocument();
    });

    // Form should be disabled
    const queryInputs = screen.getAllByPlaceholderText(/Query \d+/);
    queryInputs.forEach(input => {
      expect(input).toBeDisabled();
    });
  });
});

