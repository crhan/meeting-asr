// Minimal app-wide message channels (error + informational notice) for events no page
// renders itself.
//
// The QueryClient's default `mutations.onError` (main.tsx) reports errors here, and the
// GlobalErrorToast in App.tsx subscribes. Mutations that pass their own `onError`
// override the default (TanStack merges defaults per option key), so pages with a
// dedicated error surface keep it and never double-report. The notice channel carries
// non-error outcomes (e.g. "learned N contexts") that outlive the page that caused them.

type Listener = (message: string) => void;

let listener: Listener | null = null;
let noticeListener: Listener | null = null;

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

/** Show a non-error message in the app-wide notice toast. */
export function reportGlobalNotice(message: string): void {
  noticeListener?.(message);
}

/** Subscribe the (single) notice toast component; returns an unsubscribe function. */
export function subscribeGlobalNotice(fn: Listener): () => void {
  noticeListener = fn;
  return () => {
    if (noticeListener === fn) noticeListener = null;
  };
}
