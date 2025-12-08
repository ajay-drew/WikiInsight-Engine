import React from "react";

type ErrorBoundaryState = {
  hasError: boolean;
  message: string | null;
};

type ErrorBoundaryProps = {
  children: React.ReactNode;
};

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : "An unexpected error occurred.",
    };
  }

  componentDidCatch(error: unknown, errorInfo: React.ErrorInfo) {
    // In a real app we might send this to a logging service.
    // For now we just log to the console to aid debugging.
    // eslint-disable-next-line no-console
    console.error("Unhandled React error:", error, errorInfo);
  }

  handleReload = () => {
    this.setState({ hasError: false, message: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-100">
          <div className="max-w-md w-full px-6 py-8 border border-slate-800 rounded-lg bg-slate-900/80 shadow-lg space-y-4">
            <h1 className="text-lg font-semibold">Something went wrong</h1>
            <p className="text-sm text-slate-400">
              The topic explorer hit an unexpected error while rendering the page.
            </p>
            {this.state.message && (
              <p className="text-xs text-slate-500 break-words">
                Details: {this.state.message}
              </p>
            )}
            <button
              type="button"
              onClick={this.handleReload}
              className="mt-2 px-4 py-2 rounded bg-sky-500 text-white text-sm font-medium hover:bg-sky-600 transition-colors"
            >
              Reload app
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}



