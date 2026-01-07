import React from "react";

interface SkeletonProps {
  className?: string;
  variant?: "text" | "card" | "circle";
  width?: string | number;
  height?: string | number;
}

export function Skeleton({ 
  className = "", 
  variant = "text", 
  width, 
  height 
}: SkeletonProps) {
  const baseClasses = "animate-pulse bg-slate-800 rounded";
  
  const variantClasses = {
    text: "h-4",
    card: "h-24",
    circle: "rounded-full",
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === "number" ? `${width}px` : width;
  if (height) style.height = typeof height === "number" ? `${height}px` : height;

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
      aria-label="Loading..."
      role="status"
    />
  );
}

