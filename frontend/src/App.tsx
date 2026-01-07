import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ToastProvider, useToastContext } from "./contexts/ToastContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { ToastContainer } from "./components/ToastContainer";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";
import { SearchPage } from "./pages/Search";
import { MonitoringPage } from "./pages/Monitoring";
import { IngestionPage } from "./pages/Ingestion";
import { DashboardPage } from "./pages/Dashboard";

function AppContent() {
  const { toasts, dismissToast } = useToastContext();
  
  return (
    <>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/lookup" element={<TopicLookupPage />} />
          <Route path="/clusters" element={<ClustersOverviewPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/ingestion" element={<IngestionPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <ToastProvider>
          <AppContent />
        </ToastProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}


