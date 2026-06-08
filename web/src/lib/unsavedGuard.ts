// Module-level "are there unsaved edits?" flag, shared between SpeakerReviewPage (which
// owns the dirty state) and app chrome that would destroy it (LangToggle's full reload).
//
// Deliberately a plain module flag rather than context: only one page produces dirty
// state, and the consumers need a synchronous read at event time, not re-renders.
// Note: in-app NavLink switches remain unguarded -- useBlocker needs a data router
// (createBrowserRouter) and we intentionally stay on plain <BrowserRouter>; this flag
// plus the page's beforeunload handler covers the reload/close/lang-toggle paths.

let unsaved = false;

export function setUnsavedEdits(value: boolean): void {
  unsaved = value;
}

export function hasUnsavedEdits(): boolean {
  return unsaved;
}
