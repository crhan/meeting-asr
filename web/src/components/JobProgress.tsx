import { useEffect } from "react";
import { useJobStream } from "../lib/useJobStream";
import { tr } from "../lib/i18n";

interface Props {
  jobId: string;
  onDone?: (result: unknown) => void;
  onError?: (error: string) => void;
}

/** Live job progress: current step, description, and a bar driven by SSE. */
export function JobProgress({ jobId, onDone, onError }: Props) {
  const state = useJobStream(jobId);

  useEffect(() => {
    if (!state.done) return;
    if (state.status === "error") onError?.(state.error ?? "job failed");
    else onDone?.(state.result);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.done, state.status]);

  const ev = state.latest;
  const stepIndex = (ev?.step_index as number | undefined) ?? null;
  const stepTotal = (ev?.step_total as number | undefined) ?? null;
  const total = (ev?.total as number | undefined) ?? null;
  const completed = (ev?.completed as number | undefined) ?? null;
  const pct =
    total && completed != null ? Math.min(100, (completed / total) * 100) : null;

  return (
    <div className="job-progress">
      <div className="job-progress-head">
        <span className={`status-dot status-${state.status === "done" ? "matched" : state.status === "error" ? "conflict" : "review"}`} />
        <span>
          {state.status === "running"
            ? tr("Running…", "运行中…")
            : state.status === "done"
              ? tr("Done", "完成")
              : state.status === "error"
                ? tr("Failed", "失败")
                : tr("Queued…", "排队中…")}
        </span>
        {stepIndex != null && stepTotal != null && (
          <span className="subtle mono">
            {tr("step", "步骤")} {stepIndex}/{stepTotal}
          </span>
        )}
      </div>
      {ev?.description && <div className="job-progress-desc">{ev.description as string}</div>}
      {pct != null && (
        <div className="seg-progress" style={{ marginTop: 6 }}>
          <div className="seg-progress-bar" style={{ width: `${pct}%` }} />
        </div>
      )}
      {state.error && <div className="error-box" style={{ marginTop: 8 }}>{state.error}</div>}
    </div>
  );
}
