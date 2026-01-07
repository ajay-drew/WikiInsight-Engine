import React from "react";
import { Skeleton } from "./Skeleton";

export function ClusterCardSkeleton() {
  return (
    <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70 animate-pulse">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Skeleton variant="text" width="40%" height={20} />
          <Skeleton variant="text" width={60} height={20} />
        </div>
        <Skeleton variant="text" width="100%" height={16} />
        <div className="flex flex-wrap gap-2">
          <Skeleton variant="text" width={70} height={24} />
          <Skeleton variant="text" width={80} height={24} />
          <Skeleton variant="text" width={60} height={24} />
        </div>
      </div>
    </div>
  );
}

