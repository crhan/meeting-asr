import { useEffect, useState } from "react";
import { useJobStream } from "../lib/useJobStream";
import { cancelJob } from "../api/client";
import { confirmDialog } from "../lib/confirm";
import { tr } from "../lib/i18n";

interface Props {
  jobId: string;
  onDone?: (result: unknown) => void;
  onError?: (error: string) => void;
  /** Called on user cancellation; falls back to onError so pages always unlock. */
  onCancelled?: () => void;
  /** Hide the cancel affordance (e.g. for jobs that must not be interrupted). */
  canCancel?: boolean;
}

function statusLabel(status: string): string {
  if (status === "running") return tr("Running…", "运行中…");
  if (status === "done") return tr("Done", "完成");
  if (status === "error") return tr("Failed", "失败");
  if (status === "cancelled") return tr("Cancelled", "已取消");
  return tr("Queued…", "排队中…");
}

/** Live job progress: current step, description, and a bar driven by SSE. */
export function JobProgress({ jobId, onDone, onError, onCancelled, canCancel = true }: Props) {
  const state = useJobStream(jobId);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (!state.done) return;
    if (state.status === "error") onError?.(state.error ?? "job failed");
    else if (state.status === "cancelled") {
      // Never fall through to onDone: pages treat onDone's result as a success payload.
      if (onCancelled) onCancelled();
      else onError?.(tr("Job cancelled.", "任务已取消。"));
    } else onDone?.(state.result);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.done, state.status]);

  const requestCancel = async () => {
    if (
      !(await confirmDialog({
        message: tr(
          "Cancel this job? A queued job stops immediately; a running one stops at its next progress step. Completed stages can be reused by a re-run.",
          "取消这个任务？排队中的任务立即停止；运行中的任务会在下一个进度点停止。已完成的阶段可被重跑复用。",
        ),
        confirmLabel: tr("Cancel job", "取消任务"),
        cancelLabel: tr("Keep running", "继续运行"),
      }))
    )
      return;
    setCancelling(true);
    try {
      await cancelJob(jobId);
    } catch {
      setCancelling(false);
    }
  };

  const ev = state.latest;
  const stepIndex = (ev?.step_index as number | undefined) ?? null;
  const stepTotal = (ev?.step_total as number | undefined) ?? null;
  const total = (ev?.total as number | undefined) ?? null;
  const completed = (ev?.completed as number | undefined) ?? null;
  const pct =
    total && completed != null ? Math.min(100, (completed / total) * 100) : null;
  const active = state.status === "queued" || state.status === "running";
  const cancelPending = cancelling || state.cancelRequested;

  return (
    <div className="job-progress">
      <div className="job-progress-head">
        <span className={`status-dot status-${state.status === "done" ? "matched" : state.status === "error" ? "conflict" : state.status === "cancelled" ? "ignored" : "review"}`} />
        <span>{statusLabel(state.status)}</span>
        {stepIndex != null && stepTotal != null && (
          <span className="subtle mono">
            {tr("step", "步骤")} {stepIndex}/{stepTotal}
          </span>
        )}
        {canCancel && active && (
          <button
            className="chip danger"
            style={{ marginLeft: "auto" }}
            disabled={cancelPending}
            onClick={() => void requestCancel()}
          >
            {cancelPending ? tr("Cancelling…", "取消中…") : tr("Cancel", "取消")}
          </button>
        )}
      </div>
      {state.steps.length > 0 && stepIndex != null && (
        <div className="job-steps">
          {state.steps.map((step, index) => {
            const n = index + 1;
            const marker = n < stepIndex ? "✓" : n === stepIndex ? "▸" : "·";
            return (
              <div
                key={index}
                className={`job-step subtle ${n === stepIndex ? "now" : n < stepIndex ? "done" : ""}`}
              >
                {marker} {step}
              </div>
            );
          })}
        </div>
      )}
      {state.status === "queued" && state.waitingOn.length > 0 && (
        <div className="job-progress-desc subtle">
          {tr("Waiting for", "正在等待")}{" "}
          {state.waitingOn
            .map((j) => `${j.kind}${j.project_id ? ` (${j.project_id.split("/").pop()})` : ""}`)
            .join(", ")}{" "}
          {tr("to finish…", "完成…")}
        </div>
      )}
      {ev?.description && <div className="job-progress-desc">{ev.description as string}</div>}
      {pct != null && (
        <div className="seg-progress" style={{ marginTop: 6 }}>
          <div className="seg-progress-bar" style={{ width: `${pct}%` }} />
        </div>
      )}
      {state.error && state.status === "error" && (
        <div className="error-box" style={{ marginTop: 8 }}>{state.error}</div>
      )}
    </div>
  );
}
