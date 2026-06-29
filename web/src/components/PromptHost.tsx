import { useEffect, useRef, useState } from "react";
import { Modal } from "./Modal";
import { subscribePrompt, type PromptRequest } from "../lib/prompt";
import { tr } from "../lib/i18n";

interface Pending {
  req: PromptRequest;
  resolve: (value: string | null) => void;
}

/**
 * Single host for app-wide styled text-input dialogs. Mount once in App; pages
 * call promptDialog() and await the entered string (null on cancel). Cancel /
 * Escape / backdrop all resolve null.
 */
export function PromptHost() {
  const [pending, setPending] = useState<Pending | null>(null);
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);

  useEffect(
    () =>
      subscribePrompt((req, resolve) => {
        // A new request supersedes an unanswered one — cancel the old promise.
        setPending((prev) => {
          prev?.resolve(null);
          return { req, resolve };
        });
        setValue(req.defaultValue ?? "");
      }),
    [],
  );

  // Focus and select the field once the dialog mounts, so editing a prefilled
  // value is immediate (matching native prompt behavior).
  useEffect(() => {
    if (pending) inputRef.current?.select();
  }, [pending]);

  if (!pending) return null;
  const { req, resolve } = pending;
  const settle = (result: string | null) => {
    resolve(result);
    setPending(null);
  };

  return (
    <Modal
      title={req.title ?? tr("Input", "输入")}
      onClose={() => settle(null)}
      footer={
        <>
          <button className="btn ghost" onClick={() => settle(null)}>
            {req.cancelLabel ?? tr("Cancel", "取消")}
          </button>
          <button className="btn primary" onClick={() => settle(value)}>
            {req.confirmLabel ?? tr("OK", "确定")}
          </button>
        </>
      }
    >
      {req.message && (
        <div className="subtle" style={{ marginBottom: 8 }}>
          {req.message}
        </div>
      )}
      {req.multiline ? (
        <textarea
          ref={(el) => {
            inputRef.current = el;
          }}
          className="text-edit-area"
          style={{ minHeight: 96 }}
          placeholder={req.placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
      ) : (
        <input
          ref={(el) => {
            inputRef.current = el;
          }}
          className="search"
          style={{ marginBottom: 0 }}
          placeholder={req.placeholder}
          value={value}
          // Enter submits, Escape cancels (Modal also closes on Escape).
          onKeyDown={(e) => {
            if (e.key === "Enter") settle(value);
          }}
          onChange={(e) => setValue(e.target.value)}
        />
      )}
    </Modal>
  );
}
