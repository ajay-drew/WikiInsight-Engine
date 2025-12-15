import React, { useState } from "react";
import { Layout } from "./components/Layout";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";
import { SearchPage } from "./pages/Search";
import { MonitoringPage } from "./pages/Monitoring";
import { IngestionPage } from "./pages/Ingestion";

export default function App() {
  const [page, setPage] = useState<"lookup" | "clusters" | "search" | "monitoring" | "ingestion">("search");

  return (
    <ErrorBoundary>
      <Layout activePage={page} onChangePage={setPage}>
        {page === "search" && <SearchPage />}
        {page === "lookup" && <TopicLookupPage />}
        {page === "clusters" && <ClustersOverviewPage />}
        {page === "monitoring" && <MonitoringPage />}
        {page === "ingestion" && <IngestionPage />}
      </Layout>
    </ErrorBoundary>
  );
}


