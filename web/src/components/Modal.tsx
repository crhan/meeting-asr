import { useEffect, useRef, type ReactNode } from "react";
import { tr } from "../lib/i18n";

// Module-level count of mounted modals (mirrors unsavedGuard's plain-flag pattern):
// page-level keyboard shortcuts need a synchronous "is any dialog open?" read at event
// time so they don't act on the page behind a modal — including ConfirmHost/PromptHost,
// which pages can't know about.
let openModalCount = 0;

export function anyModalOpen(): boolean {
  return openModalCount > 0;
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  closeDisabled?: boolean;
}

/** Centered modal dialog. Closes on Escape and backdrop click. */
export function Modal({
  title,
  onClose,
  children,
  footer,
  closeDisabled = false,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    openModalCount += 1;
    return () => {
      openModalCount -= 1;
    };
  }, []);

  // Return focus to the opener on unmount -- otherwise closing drops focus on <body>
  // and keyboard users lose their place.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    return () => opener?.focus?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // IME composition cancel must not close the dialog.
      if (e.key === "Escape" && !e.isComposing && !closeDisabled) onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [closeDisabled, onClose]);

  // Minimal focus trap: Tab cycles inside the dialog instead of escaping to the
  // (visually inert but still focusable) page behind it.
  const trapTab = (e: React.KeyboardEvent) => {
    if (e.key !== "Tab") return;
    const root = dialogRef.current;
    if (!root) return;
    const focusables = [
      ...root.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      ),
    ].filter((el) => !el.hasAttribute("disabled"));
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      className="modal-backdrop"
      onMouseDown={closeDisabled ? undefined : onClose}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        ref={dialogRef}
        onKeyDown={trapTab}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <span>{title}</span>
          <button
            className="icon-btn"
            onClick={onClose}
            aria-label={tr("Close", "关闭")}
            disabled={closeDisabled}
          >
            ✕
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>
  );
}
