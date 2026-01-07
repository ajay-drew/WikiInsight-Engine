import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
}

export function Logo({ className = "", size = 32 }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="WikiInsight Logo"
    >
      {/* Gradient definition */}
      <defs>
        <linearGradient id="wGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0ea5e9" />
          <stop offset="100%" stopColor="#0284c7" />
        </linearGradient>
      </defs>
      
      {/* W letter - crisp geometric design */}
      <path
        d="M4 26 L6 10 L10 20 L14 8 L18 20 L22 10 L26 26 L22 26 L18 16 L14 26 L10 16 L6 26 Z"
        fill="url(#wGradient)"
        strokeWidth="0"
      />
    </svg>
  );
}

