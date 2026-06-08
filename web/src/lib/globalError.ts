// Minimal app-wide error channel for mutation failures that no page renders itself.
//
// The QueryClient's default `mutations.onError` (main.tsx) reports here, and the
// GlobalErrorToast in App.tsx subscribes. Mutations that pass their own `onError`
// override the default (TanStack merges defaults per option key), so pages with a
// dedicated error surface keep it and never double-report.

type Listener = (message: string) => void;

let listener: Listener | null = null;

/** Show a message in the app-wide error toast (no-op until the toast mounts). */
export function reportGlobalError(message: string): void {
  listener?.(message);
}

/** Subscribe the (single) toast component; returns an unsubscribe function. */
export function subscribeGlobalError(fn: Listener): () => void {
  listener = fn;
  return () => {
    if (listener === fn) listener = null;
  };
}
