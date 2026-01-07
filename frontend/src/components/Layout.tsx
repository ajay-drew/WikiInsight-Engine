import React from "react";
import { NavLink, Outlet } from "react-router-dom";

const linkClasses =
  "px-3 py-1.5 rounded text-sm transition-colors bg-slate-800 text-slate-200 hover:bg-slate-700";
const activeLinkClasses = "bg-sky-500 text-white";

export function Layout() {
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
            <NavLink
              to="/search"
              className={({ isActive }) => `${linkClasses} ${isActive ? activeLinkClasses : ""}`.trim()}
            >
              Search
            </NavLink>
            <NavLink
              to="/lookup"
              className={({ isActive }) => `${linkClasses} ${isActive ? activeLinkClasses : ""}`.trim()}
            >
              Topic Lookup
            </NavLink>
            <NavLink
              to="/clusters"
              className={({ isActive }) => `${linkClasses} ${isActive ? activeLinkClasses : ""}`.trim()}
            >
              Clusters
            </NavLink>
            <NavLink
              to="/monitoring"
              className={({ isActive }) => `${linkClasses} ${isActive ? activeLinkClasses : ""}`.trim()}
            >
              Monitoring
            </NavLink>
            <NavLink
              to="/ingestion"
              className={({ isActive }) => `${linkClasses} ${isActive ? activeLinkClasses : ""}`.trim()}
            >
              Ingestion
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl mx-auto px-4 py-6">
        <Outlet />
      </main>
      <footer className="border-t border-slate-800 text-xs text-slate-500 py-3 text-center">
        Built by:{" "}
        <a href="mailto:drewjay05@gmail.com" className="text-sky-400 hover:text-sky-300">
          Ajay A
        </a>{" "}
        ·{" "}
        <a href="mailto:drewjay05@gmail.com" className="text-sky-400 hover:text-sky-300">
          drewjay05@gmail.com
        </a>{" "}
        ·{" "}
        <a
          href="https://www.linkedin.com/in/ajay-drew"
          className="text-sky-400 hover:text-sky-300"
          target="_blank"
          rel="noreferrer"
        >
          linkedin.com/in/ajay-drew
        </a>
      </footer>
    </div>
  );
}


