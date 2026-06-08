import { useEffect, type ReactNode } from "react";

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
