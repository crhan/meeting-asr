import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { listJobs, type JobInfo } from "../api/client";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";
import { JobProgress } from "./JobProgress";

const ACTIVE = new Set(["queued", "running"]);

const KIND_LABEL: Record<string, [string, string]> = {
  "pipeline-run": ["Pipeline run", "管线转写"],
  "pipeline-summarize": ["Summarize", "生成纪要"],
  "correction-polish": ["Polish", "润色"],
  "voiceprint-capture": ["Voiceprint capture", "声纹采集"],
};

/** Query-key prefixes each job kind invalidates when it completes. Unknown kinds are
 *  only displayed, never trigger refreshes. */
const INVALIDATE_BY_KIND: Record<string, string[][]> = {
  "pipeline-run": [["projects"], ["speakers"], ["artifacts"]],
  "pipeline-summarize": [["projects"], ["speakers"], ["artifacts"]],
  "correction-polish": [["proposal"]],
};

function kindLabel(kind: string): string {
  const pair = KIND_LABEL[kind];
  return pair ? tr(pair[0], pair[1]) : kind;
}

/** pipeline jobs carry the project_dir absolute path; show its basename. */
function projectName(projectId: string | null): string | null {
  if (!projectId) return null;
  return projectId.split("/").filter(Boolean).pop() ?? projectId;
}

function statusBadge(job: JobInfo): string {
  if (job.status === "running") return tr("running", "运行中");
  if (job.status === "queued") return tr("queued", "排队中");
  if (job.status === "done") return tr("done", "完成");
  if (job.status === "cancelled") return tr("cancelled", "已取消");
  return tr("failed", "失败");
}

/**
 * Topbar jobs indicator: a job's progress used to live only in the component that
 * launched it, so closing that dialog (or reloading) lost all visibility. This polls
 * GET /api/jobs, re-attaches to any job's SSE stream (the server replays history), and
 * centrally refreshes the affected queries when a job finishes in the background.
 */
export function JobsIndicator() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: listJobs,
    retry: false,
    refetchInterval: (query) => {
      // Not yet authenticated (topbar mounts outside AuthGate) or server briefly down:
      // keep probing slowly and recover silently.
      if (query.state.status === "error") return 30_000;
      const jobs = query.state.data?.jobs ?? [];
      return jobs.some((job) => ACTIVE.has(job.status)) ? 2_000 : 15_000;
    },
  });

  // Centralized completion refresh: when a job transitions into "done", invalidate the
  // queries its kind affects -- pages update even when the launching component is gone.
  const prevStatuses = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    const jobs = jobsQuery.data?.jobs ?? [];
    for (const job of jobs) {
      const prev = prevStatuses.current.get(job.id);
      if (prev !== undefined && ACTIVE.has(prev) && job.status === "done") {
        for (const key of INVALIDATE_BY_KIND[job.kind] ?? []) {
          queryClient.invalidateQueries({ queryKey: key });
        }
      }
      prevStatuses.current.set(job.id, job.status);
    }
  }, [jobsQuery.data, queryClient]);

  if (jobsQuery.isError || !jobsQuery.data) return null;
  const jobs = [...jobsQuery.data.jobs].sort((a, b) => b.created_at - a.created_at);
  if (jobs.length === 0) return null;
  const activeCount = jobs.filter((job) => ACTIVE.has(job.status)).length;
  const errorCount = jobs.filter((job) => job.status === "error").length;

  return (
    <>
      <button className="btn ghost jobs-indicator" onClick={() => setOpen(true)}>
        {tr("Jobs", "任务")}
        {activeCount > 0 && <span className="jobs-count active">{activeCount}</span>}
        {activeCount === 0 && errorCount > 0 && (
          <span className="jobs-count error">{errorCount}</span>
        )}
      </button>
      {open && <JobsModal jobs={jobs} onClose={() => setOpen(false)} />}
    </>
  );
}

function JobsModal({ jobs, onClose }: { jobs: JobInfo[]; onClose: () => void }) {
  const [expanded, setExpanded] = useState<string | null>(
    jobs.find((job) => ACTIVE.has(job.status))?.id ?? null,
  );
  return (
    <Modal title={tr("Background jobs", "后台任务")} onClose={onClose}>
      <div className="jobs-list">
        {jobs.map((job) => {
          const name = projectName(job.project_id);
          const isOpen = expanded === job.id;
          return (
            <div key={job.id} className="jobs-row-wrap">
              <button
                className={`jobs-row ${isOpen ? "on" : ""}`}
                onClick={() => setExpanded(isOpen ? null : job.id)}
              >
                <span className="jobs-row-kind">{kindLabel(job.kind)}</span>
                {name && <span className="subtle mono">{name}</span>}
                <span className={`badge state-${job.status}`}>{statusBadge(job)}</span>
              </button>
              {isOpen && (
                <div className="jobs-row-detail">
                  {/* Re-attach to the SSE stream; the server replays buffered history,
                      so even a finished job shows its last progress and error. */}
                  <JobProgress jobId={job.id} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Modal>
  );
}
