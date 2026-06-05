import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  capturePlan,
  captureAccept,
  captureRollback,
  captureRun,
  getJob,
  type CaptureResult,
  type ScoreChange,
} from "../api/client";
import { tr } from "../lib/i18n";
import { useClipAudio } from "../lib/useClipAudio";
import { Modal } from "../components/Modal";

function fmtMs(ms: number): string {
  const t = Math.round(ms / 1000);
  return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, "0")}`;
}

export function CapturePage() {
  const { ref = "" } = useParams();
  const navigate = useNavigate();
  const audio = useClipAudio();

  const { data, isLoading, error } = useQuery({
    queryKey: ["capture-plan", ref],
    queryFn: () => capturePlan(ref),
    staleTime: Infinity,
    retry: false,
  });

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<CaptureResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // Pre-select recommended clips once the plan loads.
  useEffect(() => {
    if (data) {
      const rec = new Set<string>();
      for (const sp of data.speakers)
        for (const c of sp.clips) if (c.recommended) rec.add(c.rel_path);
      setSelected(rec);
    }
  }, [data]);

  // Poll the capture job until it finishes.
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        if (!alive) return;
        if (job.status === "done") {
          setResult(job.result as CaptureResult);
          setRunning(false);
          setJobId(null);
        } else if (job.status === "error") {
          setRunError(job.error ?? "capture failed");
          setRunning(false);
          setJobId(null);
        } else {
          setTimeout(tick, 1000);
        }
      } catch (e) {
        if (alive) {
          setRunError((e as Error).message);
          setRunning(false);
          setJobId(null);
        }
      }
    };
    tick();
    return () => {
      alive = false;
    };
  }, [jobId]);

  const totalSelected = selected.size;

  const toggle = (relPath: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });

  const start = async () => {
    setRunError(null);
    setRunning(true);
    try {
      const { job_id } = await captureRun(ref, [...selected]);
      setJobId(job_id);
    } catch (e) {
      setRunError((e as Error).message);
      setRunning(false);
    }
  };

  if (isLoading)
    return <div className="placeholder">{tr("Planning capture (extracting clips)…", "正在规划采集（抽取片段）…")}</div>;
  if (error) return <div className="error-box">{(error as Error).message}</div>;
  if (!data) return null;

  return (
    <div>
      <div className="review-head" style={{ margin: "-18px -18px 14px", borderRadius: 0 }}>
        <div>
          <h1>{tr("Capture voiceprints", "采集声纹")}</h1>
          <div className="subtle mono">
            {ref} · {data.speakers.length} {tr("speakers", "发言人")} ·{" "}
            {totalSelected} {tr("clips selected", "已选片段")}
          </div>
        </div>
        <div className="row gap">
          <button className="btn ghost" onClick={() => navigate(`/projects/${ref}/speakers`)}>
            {tr("Back to review", "返回 review")}
          </button>
          <button className="btn primary" disabled={running || totalSelected === 0} onClick={start}>
            {running
              ? tr("Capturing + embedding…", "采集+嵌入中…")
              : tr(`Capture ${totalSelected}`, `采集 ${totalSelected} 条`)}
          </button>
        </div>
      </div>

      {runError && <div className="error-box" style={{ marginBottom: 12 }}>{runError}</div>}

      {data.speakers.map((sp) => (
        <div key={sp.speaker_id} className="capture-speaker">
          <div className="capture-speaker-head">
            <strong>{sp.name}</strong>
            <span className="subtle">
              {" "}
              · {sp.clips.filter((c) => selected.has(c.rel_path)).length}/{sp.clips.length}{" "}
              {tr("selected", "已选")}
            </span>
          </div>
          <div className="capture-clips">
            {sp.clips.map((c) => {
              const key = `cap:${c.rel_path}`;
              const playing = audio.playingKey === key;
              const on = selected.has(c.rel_path);
              return (
                <div key={c.rel_path} className={`capture-clip ${on ? "on" : ""}`}>
                  <input type="checkbox" checked={on} onChange={() => toggle(c.rel_path)} />
                  <button
                    className="play-btn"
                    onClick={() =>
                      audio.toggle(
                        key,
                        // Plan clips are extracted under the project; play via the
                        // project clip endpoint by time range.
                        `/api/projects/${encodeURIComponent(ref)}/clip?begin_ms=${c.begin_time_ms}&end_ms=${c.end_time_ms}`,
                      )
                    }
                  >
                    {playing ? "⏸" : "▶"}
                  </button>
                  <div className="segment-body">
                    <div className="segment-meta subtle mono">
                      {fmtMs(c.begin_time_ms)} · {c.duration_seconds.toFixed(1)}s ·{" "}
                      <span className="score-badge ok">sel {c.selection_score.toFixed(2)}</span>
                      {!c.recommended && <span className="badge">{tr("not recommended", "不推荐")}</span>}
                    </div>
                    <div className="segment-text">{c.text}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {result && (
        <CaptureResultModal
          result={result}
          onAccept={async () => {
            await captureAccept(result.transaction_id);
            setResult(null);
            navigate(`/projects/${ref}/speakers`);
          }}
          onRollback={async () => {
            await captureRollback(result.transaction_id);
            setResult(null);
          }}
        />
      )}
    </div>
  );
}

function fmtScore(s: number | null): string {
  return s == null ? "—" : s.toFixed(3);
}

function changeClass(c: ScoreChange): string {
  if (c.is_critical) return "low";
  if (c.is_warning) return "mid";
  if (c.status === "improved") return "ok";
  return "";
}

function ChangeRow({ c }: { c: ScoreChange }) {
  const arrow =
    c.delta == null ? "" : c.delta > 0 ? `▲${c.delta.toFixed(3)}` : `▼${Math.abs(c.delta).toFixed(3)}`;
  return (
    <div className="change-row">
      <span className="change-label">{c.label}</span>
      <span className="change-flow mono">
        {c.before_name ?? "—"} {fmtScore(c.before_score)} → {c.after_name ?? "—"}{" "}
        {fmtScore(c.after_score)}
      </span>
      <span className={`score-badge ${changeClass(c)}`}>
        {c.status} {arrow}
      </span>
    </div>
  );
}

function CaptureResultModal(props: {
  result: CaptureResult;
  onAccept: () => void;
  onRollback: () => void;
}) {
  const { result, onAccept, onRollback } = props;
  const risky = result.current_critical + result.historical_critical_count;
  const notableCurrent = result.current_changes.filter((c) => c.status !== "unchanged");
  return (
    <Modal
      title={tr("Capture result", "采集结果")}
      onClose={props.onRollback}
      footer={
        <div className="row gap">
          <button className="btn ghost" onClick={onRollback}>
            {tr("Rollback", "回滚")}
          </button>
          <button className="btn primary" onClick={onAccept}>
            {tr("Accept", "接受")}
          </button>
        </div>
      }
    >
      <div className="capture-result">
        <div>
          {tr("Captured", "已采集")} <strong>{result.captured_count}</strong> ·{" "}
          {tr("embedded", "已嵌入")} <strong>{result.embedded_count}</strong>
          {result.skipped_count > 0 && (
            <span className="subtle"> ({result.skipped_count} {tr("skipped", "跳过")})</span>
          )}
        </div>

        <div className="result-section">
          <div className="result-section-head">
            {tr("This project", "本项目")} ·{" "}
            <span className="score-badge ok">↑{result.current_improved}</span>{" "}
            <span className="score-badge mid">↓{result.current_declined}</span>{" "}
            <span className="subtle">⟳{result.current_changed_best}</span>
          </div>
          {notableCurrent.length === 0 ? (
            <div className="subtle">{tr("No score changes.", "分数无变化。")}</div>
          ) : (
            notableCurrent.map((c) => <ChangeRow key={c.speaker_id} c={c} />)
          )}
        </div>

        <div className="result-section">
          <div className="result-section-head">
            {tr("Historical regression", "历史回归")} · {result.historical_project_count}{" "}
            {tr("projects checked", "项目检查")}
            {result.historical_critical_count > 0 && (
              <span className="score-badge low"> {result.historical_critical_count} {tr("critical", "严重")}</span>
            )}
            {result.historical_warning_count > 0 && (
              <span className="score-badge mid"> {result.historical_warning_count} {tr("warning", "警告")}</span>
            )}
          </div>
          {result.historical_projects.length === 0 ? (
            <div className="subtle">{tr("No historical regressions.", "无历史回归。")}</div>
          ) : (
            result.historical_projects.map((p) => (
              <div key={p.project_id} className="hist-project">
                <div className="hist-project-head mono subtle">
                  {p.title || p.project_id}
                </div>
                {p.risky_changes.map((c) => (
                  <ChangeRow key={`${p.project_id}:${c.speaker_id}`} c={c} />
                ))}
              </div>
            ))
          )}
        </div>

        {risky > 0 && (
          <div className="subtle" style={{ marginTop: 10, color: "var(--yellow)" }}>
            {tr(
              "Regressions detected — review before accepting.",
              "检测到回归——接受前请复核。",
            )}
          </div>
        )}
      </div>
    </Modal>
  );
}
