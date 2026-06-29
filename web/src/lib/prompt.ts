// App-wide text-input dialog channel, mirroring confirm.ts. A single PromptHost
// (mounted in App) subscribes and renders the styled Modal with an input;
// promptDialog() resolves the entered string, or null on cancel / Escape /
// backdrop click — same contract as the native window.prompt() it replaces,
// whose browser chrome does not match the app theme.

export interface PromptRequest {
  message: string;
  /** Dialog heading. Defaults to a localized "Input". */
  title?: string;
  /** Prefilled value (window.prompt's second arg). */
  defaultValue?: string;
  /** Placeholder shown when the field is empty. */
  placeholder?: string;
  /** Confirm button label. Defaults to a localized "OK". */
  confirmLabel?: string;
  /** Cancel button label. Defaults to a localized "Cancel". */
  cancelLabel?: string;
  /** Render a multi-line textarea instead of a single-line input. */
  multiline?: boolean;
}

type Listener = (req: PromptRequest, resolve: (value: string | null) => void) => void;

let listener: Listener | null = null;

/**
 * Ask the user for a string. Resolves the entered text, or null if cancelled.
 * Falls back to the native prompt if no host is mounted, so a missing provider
 * degrades instead of silently dropping the action.
 */
export function promptDialog(request: PromptRequest | string): Promise<string | null> {
  const req = typeof request === "string" ? { message: request } : request;
  if (!listener) return Promise.resolve(window.prompt(req.message, req.defaultValue));
  return new Promise((resolve) => listener!(req, resolve));
}

/** Subscribe the (single) PromptHost; returns an unsubscribe function. */
export function subscribePrompt(fn: Listener): () => void {
  listener = fn;
  return () => {
    if (listener === fn) listener = null;
  };
}
