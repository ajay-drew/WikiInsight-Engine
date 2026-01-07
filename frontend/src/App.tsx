import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";
import { SearchPage } from "./pages/Search";
import { MonitoringPage } from "./pages/Monitoring";
import { IngestionPage } from "./pages/Ingestion";

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/search" replace />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/lookup" element={<TopicLookupPage />} />
          <Route path="/clusters" element={<ClustersOverviewPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/ingestion" element={<IngestionPage />} />
          <Route path="*" element={<Navigate to="/search" replace />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}


