import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ApiError,
  capturePlan,
  captureAccept,
  captureRollback,
  captureRollbackUrl,
  captureRun,
  clipUrl,
  type CaptureResult,
  type ScoreChange,
} from "../api/client";
import { tr } from "../lib/i18n";
import { useClipAudio } from "../lib/useClipAudio";
import { JobProgress } from "../components/JobProgress";
import { Modal } from "../components/Modal";

function fmtMs(ms: number): string {
  const t = Math.round(ms / 1000);
  return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, "0")}`;
}

export function CapturePage() {
  const { ref = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const audio = useClipAudio();

  const { data, isLoading, error, refetch } = useQuery({
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

  // A completed capture leaves a server-side transaction pending until the user accepts or
  // rolls it back. If they navigate away (or reload / close the tab) without deciding, that
  // transaction wedges every later store write with HTTP 409 until the 6h server sweep, with
  // no UI to recover it. Track the pending txn in a ref and roll it back on the way out (like
  // the TUI does on unmount). Cleared the moment accept/rollback resolves it explicitly.
  const pendingTxnRef = useRef<string | null>(null);
  useEffect(() => {
    pendingTxnRef.current = result ? result.transaction_id : null;
  }, [result]);
  useEffect(() => {
    // Ask before reload/close while a capture awaits a decision. The rollback beacon must
    // NOT fire here: beforeunload runs before the user answers the browser prompt, so a
    // beacon from it would destroy the capture even when they choose to stay.
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (pendingTxnRef.current) e.preventDefault();
    };
    // pagehide fires only when the page is actually going away -- roll back then.
    const onPageHide = () => {
      const txn = pendingTxnRef.current;
      if (txn) navigator.sendBeacon(captureRollbackUrl(txn));
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    window.addEventListener("pagehide", onPageHide);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      window.removeEventListener("pagehide", onPageHide);
      const txn = pendingTxnRef.current;
      // In-app navigation away from the page: best-effort rollback of an undecided capture.
      if (txn) captureRollback(txn).catch(() => {});
    };
  }, []);

  // Pre-select recommended clips once the plan loads.
  useEffect(() => {
    if (data) {
      const rec = new Set<string>();
      for (const sp of data.speakers)
        for (const c of sp.clips) if (c.recommended) rec.add(c.rel_path);
      setSelected(rec);
    }
  }, [data]);

  const totalSelected = selected.size;
  const allClipRefs = useMemo(
    () => (data?.speakers ?? []).flatMap((sp) => sp.clips.map((c) => c.rel_path)),
    [data],
  );
  const recommendedClipRefs = useMemo(
    () =>
      (data?.speakers ?? []).flatMap((sp) =>
        sp.clips.filter((c) => c.recommended).map((c) => c.rel_path),
      ),
    [data],
  );

  const toggle = (relPath: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });

  const selectOnlyRecommended = () => setSelected(new Set(recommendedClipRefs));
  const selectAll = () => setSelected(new Set(allClipRefs));
  const clearAll = () => setSelected(new Set());
  const setSpeakerSelected = (relPaths: string[], include: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      for (const rel of relPaths) {
        if (include) next.add(rel);
        else next.delete(rel);
      }
      return next;
    });

  const start = async () => {
    setRunError(null);
    setRunning(true);
    try {
      // Send each pick's stable (begin,end) AND identity (name + person) alongside its
      // index-based rel_path so the server can detect a plan that drifted since this page loaded
      // (project edited elsewhere) -- whether the audio window OR the speaker identity changed --
      // and refuse, instead of capturing the wrong clip or storing it under the wrong person.
      const byRel = new Map(
        (data?.speakers ?? []).flatMap((sp) =>
          sp.clips.map(
            (c) =>
              [
                c.rel_path,
                {
                  rel_path: c.rel_path,
                  begin_time_ms: c.begin_time_ms,
                  end_time_ms: c.end_time_ms,
                  name: sp.name,
                  person_public_id: sp.person_public_id,
                },
              ] as const,
          ),
        ),
      );
      const selectedClips = [...selected].flatMap((rel) => {
        const c = byRel.get(rel);
        return c ? [c] : [];
      });
      const { job_id } = await captureRun(ref, selectedClips);
      setJobId(job_id);
    } catch (e) {
      setRunError((e as Error).message);
      setRunning(false);
    }
  };

  if (isLoading)
    return <div className="placeholder">{tr("Planning capture (extracting clips)…", "正在规划采集（抽取片段）…")}</div>;
  if (error) {
    // 400 = "no named speaker yet" (user input, not a fault): show guidance.
    const noNamed = error instanceof ApiError && error.status === 400;
    return (
      <div>
        <div className="review-head" style={{ margin: "-18px -18px 14px", borderRadius: 0 }}>
          <div>
            <h1>{tr("Capture voiceprints", "采集声纹")}</h1>
            <div className="subtle mono">{ref}</div>
          </div>
          <div className="row gap">
            <button className="btn ghost" onClick={() => navigate(`/projects/${ref}/speakers`)}>
              {tr("Back to review", "返回 review")}
            </button>
            <button className="btn" onClick={() => refetch()}>
              {tr("Retry", "重试")}
            </button>
          </div>
        </div>
        {noNamed ? (
          <div className="placeholder">
            {tr(
              "No named speakers to capture from. Name (or accept a match for) at least one speaker in the review page first.",
              "还没有已命名的发言人可采集。请先在复核页给至少一位发言人命名或接受匹配。",
            )}
          </div>
        ) : (
          <div className="error-box">{(error as Error).message}</div>
        )}
      </div>
    );
  }
  if (!data) return null;

  return (
    <div>
      <div className="review-head" style={{ margin: "-18px -18px 14px", borderRadius: 0 }}>
        <div>
          <h1>{tr("Capture voiceprints", "采集声纹")}</h1>
          <div className="subtle mono">
            {ref} · {data.speakers.length} {tr("speakers", "发言人")} ·{" "}
            {totalSelected}/{data.sample_count} {tr("clips selected", "已选片段")} ·{" "}
            {tr("target", "目标")} {data.target_sample_count}
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

      {runError && (
        <div className="error-box" style={{ marginBottom: 12 }}>
          <div>{runError}</div>
          {/* Plan drift arrives as a job-error STRING (no status code): always offer a
              re-plan; the plan-reload effect resets the selection to recommended. */}
          <button
            className="btn ghost"
            style={{ marginTop: 8 }}
            onClick={() => {
              setRunError(null);
              queryClient.invalidateQueries({ queryKey: ["capture-plan", ref] });
            }}
          >
            {tr("Re-plan and re-select", "重新规划并重选")}
          </button>
        </div>
      )}

      {jobId && (
        <div style={{ marginBottom: 12 }}>
          <JobProgress
            jobId={jobId}
            onDone={(jobResult) => {
              setResult(jobResult as CaptureResult);
              setRunning(false);
              setJobId(null);
            }}
            onError={(e) => {
              setRunError(e);
              setRunning(false);
              setJobId(null);
            }}
            onCancelled={() => {
              // The workflow rolls its transaction back on the way out; nothing pending.
              setRunError(tr("Capture cancelled.", "采集已取消。"));
              setRunning(false);
              setJobId(null);
            }}
          />
        </div>
      )}

      <div className="capture-toolbar">
        <button className="chip" onClick={selectOnlyRecommended}>
          {tr("Recommended only", "只选推荐")}
        </button>
        <button className="chip" onClick={selectAll}>
          {tr("Select all", "全选")}
        </button>
        <button className="chip" onClick={clearAll}>
          {tr("Clear", "清空")}
        </button>
        <span className="subtle mono">
          {tr("Recommended", "推荐")} {recommendedClipRefs.length}/{allClipRefs.length}
        </span>
      </div>

      {data.speakers.map((sp) => {
        const speakerRefs = sp.clips.map((c) => c.rel_path);
        const speakerSelected = sp.clips.filter((c) => selected.has(c.rel_path)).length;
        const speakerAllSelected = speakerSelected === sp.clips.length && sp.clips.length > 0;
        return (
          <div key={sp.speaker_id} className="capture-speaker">
            <div className="capture-speaker-head">
              <div>
                <strong>{sp.name}</strong>
                <span className="subtle">
                  {" "}
                  · {speakerSelected}/{sp.clips.length} {tr("selected", "已选")}
                </span>
                {sp.person_public_id && (
                  <span className="subtle mono"> · {sp.person_public_id}</span>
                )}
              </div>
              <button
                className="chip"
                onClick={() => setSpeakerSelected(speakerRefs, !speakerAllSelected)}
              >
                {speakerAllSelected
                  ? tr("Exclude speaker", "排除该 speaker")
                  : tr("Include speaker", "选中该 speaker")}
              </button>
            </div>
            <div className="capture-clips">
              {sp.clips.map((c) => {
                const key = `cap:${c.rel_path}`;
                const playing = audio.playingKey === key;
                const on = selected.has(c.rel_path);
                const audioScore = c.audio_score == null ? null : c.audio_score.toFixed(2);
                return (
                  <div key={c.rel_path} className={`capture-clip ${on ? "on" : ""}`}>
                    <input type="checkbox" checked={on} onChange={() => toggle(c.rel_path)} />
                    <button
                      className="play-btn"
                      onClick={() =>
                        audio.toggle(
                          key,
                          // Plan clips are extracted under the project; play via the project
                          // clip endpoint by time range. clipUrl carries the auth token so
                          // playback works on token-protected binds too.
                          clipUrl(ref, c.begin_time_ms, c.end_time_ms),
                        )
                      }
                    >
                      {playing ? "⏸" : "▶"}
                    </button>
                    <div className="segment-body">
                      <div className="segment-meta subtle mono">
                        {fmtMs(c.begin_time_ms)}-{fmtMs(c.end_time_ms)} ·{" "}
                        {c.duration_seconds.toFixed(1)}s ·{" "}
                        <span className="score-badge ok" title={c.selection_reason}>
                          {tr("selection", "选择")} {c.selection_score.toFixed(2)}
                        </span>
                        {audioScore && (
                          <span className="score-badge mid" title={c.audio_reason}>
                            {tr("audio", "音频")} {audioScore}
                          </span>
                        )}
                        <span className={`badge ${c.recommended ? "status-pill active" : ""}`}>
                          {c.recommended ? tr("recommended", "推荐") : tr("candidate", "候选")}
                        </span>
                      </div>
                      <div className="segment-text">{c.text}</div>
                      <div className="subtle capture-reason">
                        {c.selection_reason}
                        {c.audio_reason && c.audio_reason !== "-" ? ` · ${c.audio_reason}` : ""}
                      </div>
                      {playing && (
                        <div
                          className="seg-progress seekable"
                          onClick={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            audio.seek((e.clientX - rect.left) / rect.width);
                          }}
                        >
                          <div
                            className="seg-progress-bar"
                            style={{ width: `${audio.progress * 100}%` }}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {result && (
        <CaptureResultModal
          result={result}
          onAccept={async () => {
            // Keep the txn ref armed until the accept actually succeeds: if it fails or the tab
            // closes mid-flight, the unload/unmount cleanup must still roll the pending txn back
            // (clearing first would strand it pending until the server sweep). Clear only after
            // success, so the navigate below does not redundantly roll back what we accepted.
            try {
              await captureAccept(result.transaction_id);
            } catch (e) {
              setRunError((e as Error).message);
              throw e;
            }
            pendingTxnRef.current = null;
            setResult(null);
            // Accepting changed the speaker matches; drop the cached review so navigating back
            // remounts SpeakerReviewPage with fresh data instead of the pre-capture snapshot.
            await queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
            // The just-captured speakers now have voiceprints, so the cached (staleTime: Infinity)
            // capture plan is stale too -- drop it so a later return here re-plans against the
            // new library state instead of re-offering already-captured clips.
            await queryClient.invalidateQueries({ queryKey: ["capture-plan", ref] });
            // Refresh the app-wide pending-capture banner immediately; on its own 5s poll it
            // could keep offering accept/rollback for this already-resolved transaction.
            await queryClient.invalidateQueries({ queryKey: ["pending-capture"] });
            navigate(`/projects/${ref}/speakers`);
          }}
          onRollback={async () => {
            // Same as accept: clear the ref only after the rollback succeeds, so a failed/aborted
            // rollback leaves the cleanup armed to retry rather than stranding the txn pending.
            try {
              await captureRollback(result.transaction_id);
            } catch (e) {
              setRunError((e as Error).message);
              throw e;
            }
            pendingTxnRef.current = null;
            setResult(null);
            // Same as accept: refresh the banner so it can't offer the resolved transaction.
            await queryClient.invalidateQueries({ queryKey: ["pending-capture"] });
          }}
        />
      )}
    </div>
  );
}

function fmtScore(s: number | null): string {
  return s == null ? "—" : s.toFixed(3);
}

function changeClass(c: ScoreChange, current: boolean): string {
  // For the CURRENT project a changed-best is the EXPECTED result of adding its own samples (a
  // better candidate won), so it reads as green success -- NOT the regression-risk red that
  // same status means for a historical reverse check. (See AGENTS.md Voiceprint Review Notes:
  // backend is_critical is tuned for historical checks; the current view must reinterpret it.)
  if (current && c.status === "changed-best") return "ok";
  if (c.is_critical) return "low";
  if (c.is_warning) return "mid";
  if (c.status === "improved") return "ok";
  return "";
}

function ChangeRow({ c, current = false }: { c: ScoreChange; current?: boolean }) {
  const arrow =
    c.delta == null ? "" : c.delta > 0 ? `▲${c.delta.toFixed(3)}` : `▼${Math.abs(c.delta).toFixed(3)}`;
  return (
    <div className="change-row">
      <span className="change-label">{c.label}</span>
      <span className="change-flow mono">
        {c.before_name ?? "—"} {fmtScore(c.before_score)} → {c.after_name ?? "—"}{" "}
        {fmtScore(c.after_score)}
      </span>
      <span className={`score-badge ${changeClass(c, current)}`}>
        {c.status} {arrow}
      </span>
    </div>
  );
}

function CaptureResultModal(props: {
  result: CaptureResult;
  onAccept: () => Promise<void>;
  onRollback: () => Promise<void>;
}) {
  const { result, onAccept, onRollback } = props;
  const [resolving, setResolving] = useState<"accept" | "rollback" | null>(null);
  // current changed-best is expected success, not a regression -- exclude it from the warning.
  // changed-best is disjoint from the other current criticals (below-threshold / lost-candidate),
  // so subtracting its count leaves exactly the genuinely-risky current changes.
  const currentRisky = result.current_critical - result.current_changed_best;
  const risky = currentRisky + result.historical_critical_count;
  const notableCurrent = result.current_changes.filter((c) => c.status !== "unchanged");
  const resolve = async (action: "accept" | "rollback", run: () => Promise<void>) => {
    if (resolving) return;
    setResolving(action);
    try {
      await run();
    } catch {
      setResolving(null);
    }
  };
  return (
    <Modal
      title={tr("Capture result", "采集结果")}
      // No passive close: Esc / backdrop / ✕ used to silently roll back the whole
      // capture+embed run. Force an explicit Accept-or-Rollback choice instead.
      onClose={() => {}}
      closeDisabled
      footer={
        <div className="row gap">
          <button
            className="btn ghost"
            disabled={resolving !== null}
            onClick={() => void resolve("rollback", onRollback)}
          >
            {resolving === "rollback"
              ? tr("Rolling back…", "回滚中…")
              : tr("Rollback", "回滚")}
          </button>
          <button
            className="btn primary"
            disabled={resolving !== null}
            onClick={() => void resolve("accept", onAccept)}
          >
            {resolving === "accept"
              ? tr("Accepting…", "接受中…")
              : tr("Accept", "接受")}
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
          {result.quality_gate_excluded_count > 0 && (
            <span className="subtle">
              {" "}
              · {tr("quality gate excluded", "质量闸门已排除")}{" "}
              <strong>{result.quality_gate_excluded_count}</strong>
            </span>
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
            notableCurrent.map((c) => <ChangeRow key={c.speaker_id} c={c} current />)
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
