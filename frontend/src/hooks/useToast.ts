import { useToastContext } from "../contexts/ToastContext";
import { ToastType } from "../components/Toast";

export function useToast() {
  const { showToast } = useToastContext();

  return {
    showSuccess: (message: string, duration?: number) => showToast("success", message, duration),
    showError: (message: string, duration?: number) => showToast("error", message, duration),
    showInfo: (message: string, duration?: number) => showToast("info", message, duration),
    showWarning: (message: string, duration?: number) => showToast("warning", message, duration),
  };
}

