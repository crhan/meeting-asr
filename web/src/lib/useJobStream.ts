import { useEffect, useState } from "react";
import { ApiError, getJob, type ProgressEvent } from "../api/client";
import { withToken } from "./auth";
import { tr } from "./i18n";

export interface JobStreamState {
  status: string; // queued | running | done | error | cancelled
  latest: ProgressEvent | null;
  steps: string[]; // step plan, if any
  result: unknown;
  error: string | null;
  done: boolean;
  /** Running jobs this one queues behind (intersecting lock keys). */
  waitingOn: { kind: string; project_id: string | null }[];
  cancelRequested: boolean;
}

const INITIAL: JobStreamState = {
  status: "queued",
  latest: null,
  steps: [],
  result: null,
  error: null,
  done: false,
  waitingOn: [],
  cancelRequested: false,
};

/** Subscribe to a job's SSE progress; fetches the final result when it ends. */
export function useJobStream(jobId: string | null): JobStreamState {
  const [state, setState] = useState<JobStreamState>(INITIAL);

  useEffect(() => {
    if (!jobId) {
      setState(INITIAL);
      return;
    }
    setState(INITIAL);
    const src = new EventSource(withToken(`/api/jobs/${jobId}/events`));
    const isTerminal = (status: string) =>
      status === "done" || status === "error" || status === "cancelled";
    const syncSnapshot = () => {
      getJob(jobId)
        .then((job) => {
          const terminal = isTerminal(job.status);
          setState((prev) => ({
            ...prev,
            status: job.status,
            result: terminal ? job.result : prev.result,
            error: job.error ?? prev.error,
            done: terminal ? true : prev.done,
          }));
          if (terminal) src.close();
        })
        .catch((e) => {
          if (e instanceof ApiError && e.status === 404) {
            // Jobs live in server memory; a 404 here means the server restarted and the
            // job is gone. Terminal-error instead of hanging in "running" forever behind
            // an endlessly reconnecting EventSource (which locked the launching UI).
            src.close();
            setState((prev) => ({
              ...prev,
              status: "error",
              error: tr(
                "Job no longer exists (server restarted?)",
                "任务已不存在（服务器可能重启过）",
              ),
              done: true,
            }));
            return;
          }
          setState((prev) => ({
            ...prev,
            error: (e as Error).message,
          }));
        });
    };

    src.onmessage = (msg) => {
      let ev: ProgressEvent;
      try {
        ev = JSON.parse(msg.data) as ProgressEvent;
      } catch {
        return;
      }
      if (ev.type === "end") {
        src.close();
        // Fetch the terminal job snapshot for status + result.
        getJob(jobId)
          .then((job) =>
            setState((prev) => ({
              ...prev,
              status: job.status,
              result: job.result,
              error: job.error,
              done: true,
            })),
          )
          .catch(() => setState((prev) => ({ ...prev, done: true })));
        return;
      }
      if (ev.type === "status") {
        setState((prev) => ({
          ...prev,
          status: (ev.status as string) ?? prev.status,
          error: (ev.error as string | undefined) ?? prev.error,
          waitingOn: Array.isArray(ev.waiting_on)
            ? (ev.waiting_on as JobStreamState["waitingOn"])
            : prev.waitingOn,
          cancelRequested: Boolean(ev.cancel_requested) || prev.cancelRequested,
        }));
        return;
      }
      // progress event
      setState((prev) => ({
        ...prev,
        latest: ev,
        steps:
          Array.isArray(ev.step_descriptions) && ev.step_descriptions.length
            ? (ev.step_descriptions as string[])
            : prev.steps,
      }));
    };

    src.onerror = () => {
      // EventSource auto-reconnects, but if the terminal frame was missed the page would
      // otherwise hang in running state. A one-shot snapshot closes that gap.
      syncSnapshot();
    };

    return () => src.close();
  }, [jobId]);

  return state;
}
