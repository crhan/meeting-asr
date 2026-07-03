import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  listProjects,
  runPipeline,
  summarizeProject,
  type ProjectSummary,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { ExportsModal } from "../components/ExportsModal";
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
  // paths[0] is input_path; the rest are extra_inputs. Order matters: the multi-input
  // project id is a combined hash over the ordered per-file hashes -- never sort/dedup.
  const [paths, setPaths] = useState<string[]>([""]);
  const [title, setTitle] = useState("");
  const [meetingTime, setMeetingTime] = useState("");
  const [speakerCount, setSpeakerCount] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [summarize, setSummarize] = useState(true);
  const [polish, setPolish] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<RunSummary | null>(null);

  const setPath = (index: number, value: string) =>
    setPaths((prev) => prev.map((p, i) => (i === index ? value : p)));
  const removePath = (index: number) =>
    setPaths((prev) => prev.filter((_, i) => i !== index));

  const runMut = useMutation({
    mutationFn: () =>
      runPipeline({
        input_path: paths[0].trim(),
        extra_inputs: paths.slice(1).map((p) => p.trim()).filter(Boolean),
        title: title.trim() || null,
        meeting_time: meetingTime.trim() || null,
        speaker_count: speakerCount.trim() ? Number(speakerCount) : null,
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
        "The pipeline keeps running in the background; you can re-attach from the Jobs indicator in the top bar. Close this dialog?",
        "管线会继续在后台运行，可从顶栏「任务」指示器重新查看进度。关闭这个对话框吗？",
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
          onCancelled={() => {
            setJobError(tr("Run cancelled.", "转写已取消。"));
            setJobId(null);
          }}
        />
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (paths[0].trim() && !runMut.isPending) runMut.mutate();
          }}
        >
          {jobError && (
            <div className="error-box" style={{ marginBottom: 10 }}>
              {jobError}
            </div>
          )}
          {paths.map((path, index) => (
            <div key={index} className="row gap" style={{ alignItems: "center" }}>
              <input
                className="search"
                autoFocus={index === 0}
                placeholder={
                  index === 0
                    ? tr("Server-side media file path…", "服务器上的媒体文件路径…")
                    : tr(`Segment ${index + 1} path…`, `第 ${index + 1} 段路径…`)
                }
                value={path}
                onChange={(e) => setPath(index, e.target.value)}
              />
              {index > 0 && (
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={tr("Remove segment", "移除该段")}
                  onClick={() => removePath(index)}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
          <div className="row gap" style={{ marginBottom: 10 }}>
            {/* One meeting recorded as several consecutive files (e.g. a call split into
                two captures): segments are concatenated BEFORE ASR, so diarization and
                voiceprints run once over the whole meeting. */}
            <button
              type="button"
              className="chip"
              onClick={() => setPaths((prev) => [...prev, ""])}
            >
              + {tr("Add another segment", "添加另一段")}
            </button>
            {paths.length > 1 && (
              <span className="subtle" style={{ fontSize: 11.5 }}>
                {tr("Segments are transcribed as ONE meeting, in this order.", "各段按此顺序拼成一场会议转写。")}
              </span>
            )}
          </div>
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
          <label className="row gap" style={{ marginBottom: 6 }}>
            <input type="checkbox" checked={polish} onChange={(e) => setPolish(e.target.checked)} />
            {tr("Polish", "润色")}
          </label>
          <button
            type="button"
            className="chip"
            style={{ marginBottom: 8 }}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? tr("Hide advanced", "收起高级选项") : tr("Advanced…", "高级选项…")}
          </button>
          {showAdvanced && (
            <div style={{ marginBottom: 10 }}>
              <label className="subtle" style={{ display: "block", marginBottom: 4 }}>
                {tr("Meeting time (optional)", "会议时间（可选）")}
                <input
                  className="search"
                  type="datetime-local"
                  style={{ marginTop: 4 }}
                  value={meetingTime}
                  onChange={(e) => setMeetingTime(e.target.value)}
                />
              </label>
              <label className="subtle" style={{ display: "block" }}>
                {tr("Speaker count hint (optional)", "发言人数提示（可选）")}
                <input
                  className="search"
                  type="number"
                  min={1}
                  style={{ marginTop: 4 }}
                  value={speakerCount}
                  onChange={(e) => setSpeakerCount(e.target.value)}
                />
              </label>
            </div>
          )}
          {runMut.error != null && (
            <div className="error-box" style={{ marginBottom: 10 }}>
              {(runMut.error as Error).message}
            </div>
          )}
          <button
            type="submit"
            className="btn primary"
            disabled={!paths[0].trim() || runMut.isPending}
          >
            {runMut.isPending ? tr("Starting…", "启动中…") : tr("Run", "运行")}
          </button>
        </form>
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

/** States whose natural next step is the speaker review page. */
const REVIEWABLE_STATES = new Set(["transcribed", "completed", "corrected"]);

export function ProjectsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showRun, setShowRun] = useState(false);
  const [exportsFor, setExportsFor] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [summarizeJob, setSummarizeJob] = useState<{ id: string; ref: string } | null>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: listProjects,
  });

  const summarizeMut = useMutation({
    mutationFn: (ref: string) => summarizeProject(ref),
    onSuccess: (r, ref) => {
      setNotice(
        r.existing
          ? tr("A summarize is already running; re-attached.", "纪要生成已在进行中，已重新挂接。")
          : null,
      );
      setSummarizeJob({ id: r.job_id, ref });
    },
    // Inline error (e.g. 409 pending capture) instead of the global toast.
    onError: (e) => setNotice(tr("Summarize failed: ", "生成纪要失败：") + (e as Error).message),
  });

  const openProject = (p: ProjectSummary) => {
    const stateKey = p.workflow?.state_key ?? "";
    if (stateKey === "broken") {
      const missing = p.workflow?.missing ?? [];
      setNotice(
        tr(
          `Project ${p.project_id} is broken: ${missing.join(", ") || "manifest unreadable"}`,
          `项目 ${p.project_id} 状态异常：${missing.join(", ") || "manifest 不可读"}`,
        ),
      );
      return;
    }
    if (stateKey && !REVIEWABLE_STATES.has(stateKey)) {
      setNotice(
        tr(
          "This project has not been transcribed yet — run the pipeline on its source media first.",
          "该项目还没有转写——请先对它的源媒体运行管线。",
        ),
      );
      return;
    }
    navigate(`/projects/${p.project_id}/speakers`);
  };

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
      {exportsFor && <ExportsModal projectRef={exportsFor} onClose={() => setExportsFor(null)} />}
      {notice && (
        <div className="notice-box" style={{ margin: "10px 0" }} onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}
      {summarizeJob && (
        <div style={{ margin: "10px 0" }}>
          <JobProgress
            jobId={summarizeJob.id}
            onDone={() => {
              queryClient.invalidateQueries({ queryKey: ["projects"] });
              queryClient.invalidateQueries({ queryKey: ["artifacts", summarizeJob.ref] });
              setNotice(tr("Summary generated.", "纪要已生成。"));
              setSummarizeJob(null);
            }}
            onError={(e) => {
              setNotice(tr("Summarize failed: ", "生成纪要失败：") + e);
              setSummarizeJob(null);
            }}
            onCancelled={() => {
              setNotice(tr("Summarize cancelled.", "纪要生成已取消。"));
              setSummarizeJob(null);
            }}
          />
        </div>
      )}
      {projects.length === 0 ? (
        <div className="placeholder">{tr("No projects yet.", "暂无项目。")}</div>
      ) : (
        <table className="projects">
          <thead>
            <tr>
              <th>{tr("ID", "ID")}</th>
              <th>{tr("Title", "标题")}</th>
              <th>{tr("State", "状态")}</th>
              <th>{tr("Next step", "下一步")}</th>
              <th>{tr("Meeting time", "会议时间")}</th>
              <th>{tr("Outputs", "产物")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => {
              const stateKey = p.workflow?.state_key ?? "";
              const outputs = p.workflow?.outputs ?? [];
              const canSummarize = REVIEWABLE_STATES.has(stateKey);
              const hasSummary = outputs.includes("summary");
              return (
                <tr
                  key={p.project_id}
                  className="clickable"
                  onClick={() => openProject(p)}
                  title={tr("Open speaker review", "打开 speaker review")}
                >
                  <td className="mono nowrap">{p.project_id}</td>
                  <td>{p.title || tr("(untitled)", "（无标题）")}</td>
                  <td>
                    <StateBadge project={p} />
                    {p.has_unresolved_matches && (
                      <span
                        className="badge state-broken"
                        style={{ marginLeft: 6 }}
                        title={tr(
                          "Some speakers still need manual review.",
                          "还有发言人匹配待人工复核。",
                        )}
                      >
                        {tr("needs review", "待复核")}
                      </span>
                    )}
                  </td>
                  <td className="subtle">{p.workflow?.next_action ?? "—"}</td>
                  <td className="mono nowrap">{formatTime(p.meeting_time)}</td>
                  <td className="mono">
                    {outputs.length
                      ? outputs.map((o, i) => (
                          // Break only between artifacts, never inside one (the
                          // hyphen in "corrected-srt" is a default break point).
                          <span key={o}>
                            {i > 0 ? ", " : ""}
                            <span className="nowrap">{o}</span>
                          </span>
                        ))
                      : "—"}
                  </td>
                  <td>
                    <span className="row gap" onClick={(e) => e.stopPropagation()}>
                      {outputs.length > 0 && (
                        <button
                          className="chip"
                          title={tr("View / download exports", "查看/下载产物")}
                          onClick={() => setExportsFor(p.project_id)}
                        >
                          {tr("Exports", "导出")}
                        </button>
                      )}
                      {canSummarize && (
                        <button
                          className="chip"
                          disabled={summarizeMut.isPending || summarizeJob != null}
                          title={tr(
                            "Generate meeting summary via LLM.",
                            "用 LLM 生成会议纪要。",
                          )}
                          onClick={() => summarizeMut.mutate(p.project_id)}
                        >
                          {hasSummary ? tr("Re-summarize", "重新纪要") : tr("Summarize", "生成纪要")}
                        </button>
                      )}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
