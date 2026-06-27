import { useEffect, useState } from "react";
import { Modal } from "./Modal";
import { subscribeConfirm, type ConfirmRequest } from "../lib/confirm";
import { tr } from "../lib/i18n";

interface Pending {
  req: ConfirmRequest;
  resolve: (ok: boolean) => void;
}

/**
 * Single host for app-wide styled confirm dialogs. Mount once in App; pages
 * call confirmDialog() and await the user's choice. Cancel / Escape / backdrop
 * all resolve false.
 */
export function ConfirmHost() {
  const [pending, setPending] = useState<Pending | null>(null);

  useEffect(
    () =>
      subscribeConfirm((req, resolve) =>
        setPending((prev) => {
          // A new request supersedes an unanswered one — cancel the old promise
          // so nothing is left hanging.
          prev?.resolve(false);
          return { req, resolve };
        }),
      ),
    [],
  );

  if (!pending) return null;
  const { req, resolve } = pending;
  const settle = (ok: boolean) => {
    resolve(ok);
    setPending(null);
  };

  return (
    <Modal
      title={req.title ?? tr("Confirm", "确认")}
      onClose={() => settle(false)}
      footer={
        <>
          <button className="btn ghost" onClick={() => settle(false)}>
            {req.cancelLabel ?? tr("Cancel", "取消")}
          </button>
          <button
            className={`btn ${req.danger ? "danger" : "primary"}`}
            autoFocus
            onClick={() => settle(true)}
          >
            {req.confirmLabel ?? tr("Confirm", "确认")}
          </button>
        </>
      }
    >
      <div style={{ lineHeight: 1.5 }}>{req.message}</div>
    </Modal>
  );
}
