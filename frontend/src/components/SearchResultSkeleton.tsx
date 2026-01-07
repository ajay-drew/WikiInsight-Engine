import React from "react";
import { Skeleton } from "./Skeleton";

export function SearchResultSkeleton() {
  return (
    <div className="border border-slate-800 rounded-lg p-4 bg-slate-900/70 animate-pulse">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0 space-y-3">
          <div className="flex items-start gap-3">
            <Skeleton variant="text" width="60%" height={20} />
            <div className="flex items-center gap-2 flex-shrink-0 ml-auto">
              <Skeleton variant="text" width={80} height={24} />
              <Skeleton variant="text" width={40} height={24} />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Skeleton variant="text" width={120} height={16} />
            <Skeleton variant="text" width={80} height={16} />
            <Skeleton variant="text" width={100} height={16} />
          </div>

          <div className="flex flex-wrap gap-2">
            <Skeleton variant="text" width={80} height={24} />
            <Skeleton variant="text" width={100} height={24} />
            <Skeleton variant="text" width={90} height={24} />
          </div>

          <div className="flex items-center gap-2">
            <Skeleton variant="text" width={100} height={32} />
            <Skeleton variant="text" width={100} height={32} />
          </div>
        </div>
      </div>
    </div>
  );
}

