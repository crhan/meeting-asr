import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { listProjects, runPipeline, type ProjectSummary } from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { Modal } from "../components/Modal";
import { JobProgress } from "../components/JobProgress";

/** Terminal payload of the pipeline-run job (see routers/pipeline.py work()). */
interface RunSummary {
  project_id?: string;
  detected_speaker_count?: number;
  sentence_count?: number;
  applied_speaker_count?: number;
  has_summary?: boolean;
  polished?: boolean;
}

function RunDialog({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [path, setPath] = useState("");
  const [title, setTitle] = useState("");
  const [summarize, setSummarize] = useState(true);
  const [polish, setPolish] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<RunSummary | null>(null);

  const runMut = useMutation({
    mutationFn: () =>
      runPipeline({
        input_path: path.trim(),
        title: title.trim() || null,
        summarize,
        polish,
      }),
    onSuccess: (r) => {
      setJobError(null);
      setJobId(r.job_id);
    },
    // This dialog renders runMut.error inline below the form; an explicit onError
    // replaces the QueryClient's default global-toast handler so it isn't shown twice.
    onError: () => {},
  });

  // While the job runs, an accidental Esc/backdrop click would hide the only progress
  // view (there is no other place to re-attach yet) -- ask first. The job itself keeps
  // running server-side either way.
  const requestClose = () => {
    if (!jobId) {
      onClose();
      return;
    }
    void confirmDialog({
      message: tr(
        "The pipeline keeps running in the background, but this dialog is currently the only progress view. Close it anyway?",
        "管线会继续在后台运行，但这个对话框目前是唯一的进度视图。仍要关闭吗？",
      ),
      confirmLabel: tr("Close", "关闭"),
    }).then((ok) => {
      if (ok) onClose();
    });
  };

  return (
    <Modal title={tr("Run pipeline (new transcription)", "运行管线（新转写）")} onClose={requestClose}>
      {runResult ? (
        <div>
          <div className="capture-result">
            <div>
              {tr("Pipeline finished.", "管线已完成。")}{" "}
              <span className="mono subtle">{runResult.project_id}</span>
            </div>
            <div className="subtle" style={{ marginTop: 6 }}>
              {runResult.detected_speaker_count ?? "—"} {tr("speakers detected", "位发言人")} ·{" "}
              {runResult.sentence_count ?? "—"} {tr("sentences", "句")} ·{" "}
              {runResult.has_summary ? tr("summary ready", "纪要已生成") : tr("no summary", "无纪要")} ·{" "}
              {runResult.polished ? tr("polished", "已润色") : tr("not polished", "未润色")}
            </div>
          </div>
          <div className="row gap" style={{ marginTop: 12 }}>
            <button className="btn ghost" onClick={onClose}>
              {tr("Back to list", "返回列表")}
            </button>
            {runResult.project_id && (
              <button
                className="btn primary"
                onClick={() => {
                  onClose();
                  navigate(`/projects/${encodeURIComponent(runResult.project_id!)}/speakers`);
                }}
              >
                {tr("Review speakers", "去复核发言人")}
              </button>
            )}
          </div>
        </div>
      ) : jobId ? (
        <JobProgress
          jobId={jobId}
          onDone={(result) => {
            queryClient.invalidateQueries({ queryKey: ["projects"] });
            // Show a completion summary with a "review speakers" entry instead of
            // silently vanishing back to the list.
            setRunResult((result ?? {}) as RunSummary);
            setJobId(null);
          }}
          // Keep the terminal error after the job panel unmounts; clearing jobId alone would
          // drop the only explanation and bounce the user back to a blank form.
          onError={(e) => {
            setJobError(e);
            setJobId(null);
          }}
        />
      ) : (
        <>
          {jobError && (
            <div className="error-box" style={{ marginBottom: 10 }}>
              {jobError}
            </div>
          )}
          <input
            className="search"
            autoFocus
            placeholder={tr("Server-side media file path…", "服务器上的媒体文件路径…")}
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
          <input
            className="search"
            placeholder={tr("Title (optional)", "标题（可选）")}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <label className="row gap" style={{ marginBottom: 6 }}>
            <input type="checkbox" checked={summarize} onChange={(e) => setSummarize(e.target.checked)} />
            {tr("Summarize", "生成纪要")}
          </label>
          <label className="row gap" style={{ marginBottom: 12 }}>
            <input type="checkbox" checked={polish} onChange={(e) => setPolish(e.target.checked)} />
            {tr("Polish", "润色")}
          </label>
          {runMut.error && (
            <div className="error-box" style={{ marginBottom: 10 }}>
              {(runMut.error as Error).message}
            </div>
          )}
          <button
            className="btn primary"
            disabled={!path.trim() || runMut.isPending}
            onClick={() => runMut.mutate()}
          >
            {runMut.isPending ? tr("Starting…", "启动中…") : tr("Run", "运行")}
          </button>
        </>
      )}
    </Modal>
  );
}

function StateBadge({ project }: { project: ProjectSummary }) {
  const key = project.workflow?.state_key ?? project.status;
  const label = project.workflow?.state ?? project.status;
  return <span className={`badge state-${key}`}>{label}</span>;
}

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const [showRun, setShowRun] = useState(false);
  const { data, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: listProjects,
  });

  if (isLoading) {
    return <div className="placeholder">{tr("Loading projects…", "正在加载项目…")}</div>;
  }
  if (error) {
    return (
      <div className="error-box">
        {tr("Failed to load projects: ", "加载项目失败：")}
        {(error as Error).message}
      </div>
    );
  }
  const projects = data?.projects ?? [];

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1>{tr("Projects", "项目")}</h1>
          <div className="subtle mono">
            {data?.projects_dir} · {projects.length} {tr("projects", "个项目")}
          </div>
        </div>
        <button className="btn primary" onClick={() => setShowRun(true)}>
          + {tr("Run pipeline", "运行管线")}
        </button>
      </div>
      {showRun && <RunDialog onClose={() => setShowRun(false)} />}
      {projects.length === 0 ? (
        <div className="placeholder">{tr("No projects yet.", "暂无项目。")}</div>
      ) : (
        <table className="projects">
          <thead>
            <tr>
              <th>{tr("ID", "ID")}</th>
              <th>{tr("Title", "标题")}</th>
              <th>{tr("State", "状态")}</th>
              <th>{tr("Meeting time", "会议时间")}</th>
              <th>{tr("Outputs", "产物")}</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr
                key={p.project_id}
                className="clickable"
                onClick={() => navigate(`/projects/${p.project_id}/speakers`)}
                title={tr("Open speaker review", "打开 speaker review")}
              >
                <td className="mono nowrap">{p.project_id}</td>
                <td>{p.title || tr("(untitled)", "（无标题）")}</td>
                <td>
                  <StateBadge project={p} />
                </td>
                <td className="mono nowrap">{formatTime(p.meeting_time)}</td>
                <td className="mono">
                  {p.workflow?.outputs.length
                    ? p.workflow.outputs.map((o, i) => (
                        // Break only between artifacts, never inside one (the
                        // hyphen in "corrected-srt" is a default break point).
                        <span key={o}>
                          {i > 0 ? ", " : ""}
                          <span className="nowrap">{o}</span>
                        </span>
                      ))
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
