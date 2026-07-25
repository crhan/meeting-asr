import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
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
} from "../api/client";
import { tr } from "../lib/i18n";
import { useClipAudio } from "../lib/useClipAudio";
import { JobProgress } from "../components/JobProgress";
import { SeekBar } from "../components/SeekBar";
import { CaptureResultModal } from "../components/CaptureResultModal";

function fmtMs(ms: number): string {
  const t = Math.round(ms / 1000);
  return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, "0")}`;
}

export function CapturePage() {
  const { ref = "" } = useParams();
  const [searchParams] = useSearchParams();
  // Arriving from the quality page's sourcing panel names one speaker: that
  // request is "top up this person", not "capture this meeting". Honouring it
  // is what makes the deep link land aimed instead of on a full plan the
  // operator has to re-narrow by hand.
  const focusSpeakerParam = searchParams.get("speaker");
  const focusSpeakerId =
    focusSpeakerParam !== null && /^\d+$/.test(focusSpeakerParam)
      ? Number(focusSpeakerParam)
      : null;
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
  const [runNotice, setRunNotice] = useState<string | null>(null);

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

  // Pre-select recommended clips once the plan loads, narrowed to the focused
  // speaker when one was requested.
  useEffect(() => {
    if (data) {
      const rec = new Set<string>();
      for (const sp of data.speakers) {
        if (focusSpeakerId !== null && sp.speaker_id !== focusSpeakerId) continue;
        for (const c of sp.clips) if (c.recommended) rec.add(c.rel_path);
      }
      setSelected(rec);
    }
  }, [data, focusSpeakerId]);

  // A focused speaker the plan does not contain means the project changed
  // since the recommendation was computed; say so rather than silently
  // showing an unfiltered plan the operator believes is filtered.
  const focusMissing =
    focusSpeakerId !== null &&
    !!data &&
    !data.speakers.some((sp) => sp.speaker_id === focusSpeakerId);

  // Focus narrows what is shown and what the toolbar counts, but never what
  // can be captured: the "show all" escape hatch restores the full plan.
  const visibleSpeakers = useMemo(() => {
    const speakers = data?.speakers ?? [];
    if (focusSpeakerId === null || focusMissing) return speakers;
    return speakers.filter((sp) => sp.speaker_id === focusSpeakerId);
  }, [data, focusSpeakerId, focusMissing]);
  const focusedName = visibleSpeakers.length === 1 ? visibleSpeakers[0].name : null;
  const hiddenSpeakerCount = (data?.speakers.length ?? 0) - visibleSpeakers.length;

  const totalSelected = selected.size;
  const allClipRefs = useMemo(
    () => visibleSpeakers.flatMap((sp) => sp.clips.map((c) => c.rel_path)),
    [visibleSpeakers],
  );
  const recommendedClipRefs = useMemo(
    () =>
      visibleSpeakers.flatMap((sp) =>
        sp.clips.filter((c) => c.recommended).map((c) => c.rel_path),
      ),
    [visibleSpeakers],
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
      const { job_id, existing } = await captureRun(ref, selectedClips);
      // The server deduplicates onto an in-flight capture for this project; THIS
      // submit's clip selection did not run. Attaching silently would present the
      // other selection's result as if it were this one.
      setRunNotice(
        existing
          ? tr(
              "A capture for this project is already running — attached to its progress. This submission's clip selection was NOT used; wait for (or cancel) the running capture, then re-select.",
              "该项目已有一次采集在进行中——已挂接其进度。本次提交的片段选择并未生效；请等它完成（或取消）后重新选择提交。",
            )
          : null,
      );
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
          {/* Counts describe what is on screen. Reporting the whole plan's 5
              speakers and 60 clips while showing one speaker's 12 reads as a
              filter that did not take. */}
          <div className="subtle mono">
            {ref} · {visibleSpeakers.length} {tr("speakers", "发言人")} ·{" "}
            {totalSelected}/{allClipRefs.length} {tr("clips selected", "已选片段")} ·{" "}
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

      {runNotice && (
        <div className="notice-box" style={{ marginBottom: 12 }}>
          {runNotice}
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

      {focusMissing && (
        <div className="notice-box" style={{ margin: "10px 0" }}>
          {tr(
            `Speaker ${focusSpeakerParam} is no longer in this project's plan — the project changed since it was recommended. Showing every speaker.`,
            `本项目的采集计划里已经没有 speaker ${focusSpeakerParam} 了——推荐生成之后项目被改过。下面显示全部说话人。`,
          )}
        </div>
      )}

      {focusedName && hiddenSpeakerCount > 0 && (
        <div className="notice-box" style={{ margin: "10px 0" }}>
          {tr(
            `Showing ${focusedName} only, with their recommended clips pre-selected.`,
            `只显示 ${focusedName}，并已预选其推荐片段。`,
          )}{" "}
          <button
            className="btn ghost"
            style={{ marginLeft: 8 }}
            onClick={() => navigate(`/projects/${ref}/capture`)}
          >
            {tr(
              `Show all ${hiddenSpeakerCount + 1} speakers`,
              `显示全部 ${hiddenSpeakerCount + 1} 位说话人`,
            )}
          </button>
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

      {visibleSpeakers.map((sp) => {
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
                      {playing && <SeekBar progress={audio.progress} onSeek={audio.seek} />}
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
