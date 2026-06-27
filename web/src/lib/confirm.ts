// App-wide confirm-dialog channel, mirroring globalError.ts. A single
// ConfirmHost (mounted in App) subscribes and renders the styled Modal;
// confirmDialog() resolves true when the user confirms, false on cancel /
// Escape / backdrop click. This replaces the native window.confirm(), whose
// browser chrome does not match the app theme.

export interface ConfirmRequest {
  message: string;
  /** Dialog heading. Defaults to a localized "Confirm". */
  title?: string;
  /** Confirm button label. Defaults to a localized "Confirm". */
  confirmLabel?: string;
  /** Cancel button label. Defaults to a localized "Cancel". */
  cancelLabel?: string;
  /** Render the confirm button in the destructive (red) style. */
  danger?: boolean;
}

type Listener = (req: ConfirmRequest, resolve: (ok: boolean) => void) => void;

let listener: Listener | null = null;

/**
 * Ask the user to confirm an action. Resolves true if confirmed, false
 * otherwise. Falls back to the native confirm if no host is mounted, so a
 * missing provider degrades instead of silently dropping the action.
 */
export function confirmDialog(request: ConfirmRequest | string): Promise<boolean> {
  const req = typeof request === "string" ? { message: request } : request;
  if (!listener) return Promise.resolve(window.confirm(req.message));
  return new Promise((resolve) => listener!(req, resolve));
}

/** Subscribe the (single) ConfirmHost; returns an unsubscribe function. */
export function subscribeConfirm(fn: Listener): () => void {
  listener = fn;
  return () => {
    if (listener === fn) listener = null;
  };
}
