import React, { useState } from "react";
import { Layout } from "./components/Layout";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";
import { SearchPage } from "./pages/Search";

export default function App() {
  const [page, setPage] = useState<"lookup" | "clusters" | "search">("search");

  return (
    <Layout activePage={page} onChangePage={setPage}>
      {page === "search" && <SearchPage />}
      {page === "lookup" && <TopicLookupPage />}
      {page === "clusters" && <ClustersOverviewPage />}
    </Layout>
  );
}


