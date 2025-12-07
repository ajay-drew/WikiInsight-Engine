import React, { useState } from "react";
import { Layout } from "./components/Layout";
import { TopicLookupPage } from "./pages/TopicLookup";
import { ClustersOverviewPage } from "./pages/ClustersOverview";

export default function App() {
  const [page, setPage] = useState<"lookup" | "clusters">("lookup");

  return (
    <Layout activePage={page} onChangePage={setPage}>
      {page === "lookup" ? <TopicLookupPage /> : <ClustersOverviewPage />}
    </Layout>
  );
}


