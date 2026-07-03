// Module-level "are there unsaved edits?" flag, shared between SpeakerReviewPage (which
// owns the dirty state) and app chrome that would destroy it (topbar GuardedNavLink,
// LangToggle's full reload).
//
// Deliberately a plain module flag rather than context: only one page produces dirty
// state, and the consumers need a synchronous read at event time, not re-renders.
// Covered paths: reload/close (beforeunload), lang toggle, topbar nav links, and the
// page's own cross-project sentence jump. Browser back/forward remains unguarded --
// useBlocker needs a data router (createBrowserRouter) and we intentionally stay on
// plain <BrowserRouter>.

let unsaved = false;

export function setUnsavedEdits(value: boolean): void {
  unsaved = value;
}

export function hasUnsavedEdits(): boolean {
  return unsaved;
}
