import React from "react";

type LayoutProps = {
  activePage: "lookup" | "clusters" | "search";
  onChangePage: (page: "lookup" | "clusters" | "search") => void;
  children: React.ReactNode;
};

export function Layout({ activePage, onChangePage, children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl font-semibold">WikiInsight</span>
            <span className="text-xs text-slate-400 border border-slate-700 px-1.5 py-0.5 rounded">
              Topic Explorer
            </span>
          </div>
          <nav className="flex gap-2">
            <button
              onClick={() => onChangePage("search")}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                activePage === "search"
                  ? "bg-sky-500 text-white"
                  : "bg-slate-800 text-slate-200 hover:bg-slate-700"
              }`}
            >
              Search
            </button>
            <button
              onClick={() => onChangePage("lookup")}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                activePage === "lookup"
                  ? "bg-sky-500 text-white"
                  : "bg-slate-800 text-slate-200 hover:bg-slate-700"
              }`}
            >
              Topic Lookup
            </button>
            <button
              onClick={() => onChangePage("clusters")}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                activePage === "clusters"
                  ? "bg-sky-500 text-white"
                  : "bg-slate-800 text-slate-200 hover:bg-slate-700"
              }`}
            >
              Clusters
            </button>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl mx-auto px-4 py-6">{children}</main>
      <footer className="border-t border-slate-800 text-xs text-slate-500 py-3 text-center">
        Backend: FastAPI · Frontend: React + Vite + Tailwind
      </footer>
    </div>
  );
}


