import { useEffect, type ReactNode } from "react";

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
  useEffect(() => {
    openModalCount += 1;
    return () => {
      openModalCount -= 1;
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !closeDisabled) onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [closeDisabled, onClose]);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={closeDisabled ? undefined : onClose}
    >
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>{title}</span>
          <button
            className="icon-btn"
            onClick={onClose}
            aria-label="close"
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
