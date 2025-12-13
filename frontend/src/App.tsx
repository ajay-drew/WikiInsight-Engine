import React, { useState } from "react";
import { Layout } from "./components/Layout";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";
import { SearchPage } from "./pages/Search";
import { MonitoringPage } from "./pages/Monitoring";

export default function App() {
  const [page, setPage] = useState<"lookup" | "clusters" | "search" | "monitoring">("search");

  return (
    <ErrorBoundary>
      <Layout activePage={page} onChangePage={setPage}>
        {page === "search" && <SearchPage />}
        {page === "lookup" && <TopicLookupPage />}
        {page === "clusters" && <ClustersOverviewPage />}
        {page === "monitoring" && <MonitoringPage />}
      </Layout>
    </ErrorBoundary>
  );
}


