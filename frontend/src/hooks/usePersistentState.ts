import { useEffect, useState } from "react";

// Simple sessionStorage-backed state to persist UI between route changes.
export function usePersistentState<T>(key: string, initialValue: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [state, setState] = useState<T>(() => {
    if (typeof window === "undefined") return initialValue;
    try {
      const stored = window.sessionStorage.getItem(key);
      return stored ? (JSON.parse(stored) as T) : initialValue;
    } catch (err) {
      console.warn(`Failed to read persisted state for ${key}`, err);
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(state));
    } catch (err) {
      console.warn(`Failed to persist state for ${key}`, err);
    }
  }, [key, state]);

  return [state, setState];
}

