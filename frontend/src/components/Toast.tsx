import React from "react";
import { FaCheckCircle, FaExclamationCircle, FaInfoCircle, FaTimes, FaExclamationTriangle } from "react-icons/fa";

export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

interface ToastProps {
  toast: Toast;
  onDismiss: (id: string) => void;
}

const icons = {
  success: FaCheckCircle,
  error: FaExclamationCircle,
  info: FaInfoCircle,
  warning: FaExclamationTriangle,
};

const colors = {
  success: "bg-green-500/20 border-green-500/50 text-green-300",
  error: "bg-red-500/20 border-red-500/50 text-red-300",
  info: "bg-blue-500/20 border-blue-500/50 text-blue-300",
  warning: "bg-yellow-500/20 border-yellow-500/50 text-yellow-300",
};

export function ToastComponent({ toast, onDismiss }: ToastProps) {
  const Icon = icons[toast.type];
  const colorClass = colors[toast.type];

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg border backdrop-blur-sm shadow-lg min-w-[300px] max-w-md ${colorClass}`}
      role="alert"
      aria-live="polite"
    >
      <Icon className="flex-shrink-0 mt-0.5" />
      <div className="flex-1 text-sm">{toast.message}</div>
      <button
        onClick={() => onDismiss(toast.id)}
        className="flex-shrink-0 text-current opacity-70 hover:opacity-100 transition-opacity"
        aria-label="Dismiss notification"
      >
        <FaTimes className="text-xs" />
      </button>
    </div>
  );
}

