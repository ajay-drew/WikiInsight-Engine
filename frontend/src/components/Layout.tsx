import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import { FaMoon, FaSun } from "react-icons/fa";
import { Tooltip } from "./Tooltip";
import { useTheme } from "../contexts/ThemeContext";
import { Logo } from "./Logo";

const linkClasses =
  "px-3 py-1.5 rounded text-sm transition-colors";

// Styled NavLink component that properly handles theme colors
function ThemedNavLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => {
        const baseClasses = linkClasses;
        if (isActive) {
          return `${baseClasses} bg-[var(--accent)] text-white shadow-lg`;
        }
        return `${baseClasses} bg-[var(--bg-tertiary)] text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]`;
      }}
    >
      {children}
    </NavLink>
  );
}

export function Layout() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <header className="border-b border-[var(--border-color)] bg-[var(--bg-secondary)]/80 backdrop-blur relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <NavLink to="/dashboard" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
              <Logo size={28} className="flex-shrink-0" />
              <span className="text-xl font-semibold">WikiInsight</span>
            </NavLink>
            <span className="text-xs text-[var(--text-tertiary)] border border-[var(--border-color)] px-1.5 py-0.5 rounded">
              Topic Explorer
            </span>
          </div>
          <div className="flex items-center gap-3">
            <nav className="flex gap-2">
            <Tooltip content="Dashboard with system overview">
              <ThemedNavLink to="/dashboard">Dashboard</ThemedNavLink>
            </Tooltip>
            <Tooltip content="Hybrid search combining semantic and keyword search">
              <ThemedNavLink to="/search">Search</ThemedNavLink>
            </Tooltip>
            <Tooltip content="Look up topic cluster for a specific article">
              <ThemedNavLink to="/lookup">Topic Lookup</ThemedNavLink>
            </Tooltip>
            <Tooltip content="Browse and explore topic clusters">
              <ThemedNavLink to="/clusters">Clusters</ThemedNavLink>
            </Tooltip>
            <Tooltip content="Monitor pipeline status and system metrics">
              <ThemedNavLink to="/monitoring">Monitoring</ThemedNavLink>
            </Tooltip>
            <Tooltip content="Run data ingestion and clustering pipeline">
              <ThemedNavLink to="/ingestion">Ingestion</ThemedNavLink>
            </Tooltip>
          </nav>
          <Tooltip content={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            <button
              onClick={toggleTheme}
              className="p-2 rounded bg-[var(--bg-tertiary)] hover:bg-[var(--bg-secondary)] text-[var(--text-primary)] transition-colors"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <FaSun /> : <FaMoon />}
            </button>
          </Tooltip>
          </div>
        </div>
      </header>
      <main className="flex-1 max-w-6xl mx-auto px-4 py-6">
        <Outlet />
      </main>
      <footer className="border-t border-[var(--border-color)] text-xs text-[var(--text-tertiary)] py-3 text-center">
        Built by:{" "}
        <a href="mailto:drewjay05@gmail.com" className="text-[var(--accent)] hover:opacity-80 transition-opacity">
          Ajay A
        </a>{" "}
        ·{" "}
        <a href="mailto:drewjay05@gmail.com" className="text-[var(--accent)] hover:opacity-80 transition-opacity">
          drewjay05@gmail.com
        </a>{" "}
        ·{" "}
        <a
          href="https://www.linkedin.com/in/ajay-drew"
          className="text-[var(--accent)] hover:opacity-80 transition-opacity"
          target="_blank"
          rel="noreferrer"
        >
          linkedin.com/in/ajay-drew
        </a>
      </footer>
    </div>
  );
}


