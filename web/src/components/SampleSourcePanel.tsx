import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  captureAccept,
  captureRollback,
  captureRollbackUrl,
  captureRun,
  clipUrl,
  getSampleSources,
  type CaptureResult,
  type SampleSource,
} from "../api/client";
import { tr } from "../lib/i18n";
import { fmtClock, fmtSeconds } from "../lib/format";
import { useClipAudio } from "../lib/useClipAudio";
import { useJobStream } from "../lib/useJobStream";
import { CaptureResultModal } from "./CaptureResultModal";
import { JobProgress } from "./JobProgress";
import { PanelHead } from "./PanelHead";

/** Restate a person's unmet health bar in the active locale. */
function deficitLabel(kind: string): string {
  if (kind === "unusable") return tr("not matchable", "无法参与匹配");
  if (kind === "fragile-cluster") return tr("too few samples", "样本太少");
  if (kind === "short-audio") return tr("too little audio", "音频太短");
  if (kind === "single-source") return tr("one recording only", "只有一场录音");
  return kind;
}

/**
 * Restate why a project earned its rank.
 *
 * Same contract as the issue queue: the backend ships reason *kinds*, the
 * locale lives here. A reason is only worth showing if it changes what the
 * operator would do, so each one names the consequence, not the metric.
 */
function reasonLabel(kind: string): { text: string; tone: string } | null {
  if (kind === "new-project")
    return { text: tr("new recording", "新的录音场次"), tone: "good" };
  if (kind === "retry-quarantined")
    return {
      text: tr("last attempt was quarantined", "上次采的样本被隔离了"),
      tone: "warn",
    };
  if (kind === "large-supply")
    return { text: tr("lots to take", "素材充足"), tone: "good" };
  if (kind === "thin-supply")
    return { text: tr("barely any left", "所剩无几"), tone: "warn" };
  if (kind === "name-only")
    return { text: tr("matched by name only", "仅按名字匹配"), tone: "warn" };
  if (kind === "already-harvested")
    return { text: tr("already sampled here", "这场已采过"), tone: "" };
  if (kind === "crowded-only")
    return {
      text: tr("every clip has another voice", "每条都录到他人"),
      tone: "warn",
    };
  return null;
}

/**
 * Ranked places to harvest more samples for one person, capturable in place.
 *
 * This panel exists because "capture more samples" is only half an
 * instruction. The half that costs the operator an afternoon is *where* --
 * which of a dozen meetings this person actually speaks in, and whether
 * anything is left there worth taking.
 *
 * Having answered that, sending them to another page to re-pick clips would
 * waste the answer: the clips are already chosen, already listenable, and
 * already carry the `rel_path` a capture run accepts. So the rows are
 * checkboxes and capture happens here. The per-project picker stays one click
 * away for the cases this view deliberately does not cover -- audio-quality
 * scores, seek bars, and the other speakers in that meeting.
 *
 * Shared by the quality page and the library page rather than reimplemented in
 * each: a capture leaves a server-side transaction that wedges every later
 * store write until it is accepted or rolled back, so a second copy of this
 * flow is a second chance to forget the rollback path.
 */
export function SampleSourcePanel(props: {
  personId: string;
  personName: string;
  onClose: () => void;
  /** Refresh whatever the host page shows about this person after a capture
   *  is accepted or rolled back. */
  onSettled?: () => void;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const audio = useClipAudio();
  const query = useQuery({
    queryKey: ["vp-sources", props.personId],
    queryFn: () => getSampleSources(props.personId),
  });
  const data = query.data;

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeProject, setActiveProject] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<CaptureResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJobStream(jobId);

  // Pre-select each source's recommended clips once the plan arrives.
  useEffect(() => {
    if (!data) return;
    const picks = new Set<string>();
    for (const source of data.sources)
      for (const clip of source.clips) if (clip.recommended) picks.add(clip.rel_path);
    setSelected(picks);
  }, [data]);

  // A completed capture leaves a server-side transaction open until it is
  // accepted or rolled back, and an unresolved one wedges every later store
  // write with HTTP 409. Roll back on the way out if the user never decided.
  const pendingTxnRef = useRef<string | null>(null);
  useEffect(() => {
    pendingTxnRef.current = result ? result.transaction_id : null;
  }, [result]);
  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      if (pendingTxnRef.current) event.preventDefault();
    };
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
      if (txn) captureRollback(txn).catch(() => {});
    };
  }, []);

  const reportedJobRef = useRef<string | null>(null);
  useEffect(() => {
    if (!jobId || !job.done) return;
    if (reportedJobRef.current === jobId) return;
    reportedJobRef.current = jobId;
    setJobId(null);
    setActiveProject(null);
    if (job.error) {
      setError(job.error);
      return;
    }
    setResult(job.result as CaptureResult);
  }, [jobId, job.done, job.error, job.result]);

  const toggle = (relPath: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });

  const capture = async (source: SampleSource) => {
    const picks = source.clips.filter((clip) => selected.has(clip.rel_path));
    if (!picks.length) return;
    setError(null);
    setActiveProject(source.project_id);
    try {
      // Echo each pick's stable (begin,end) and the plan's identity so the
      // server can refuse a plan that drifted since this panel loaded, rather
      // than storing the wrong audio under this person.
      const { job_id, existing } = await captureRun(
        source.project_id,
        picks.map((clip) => ({
          rel_path: clip.rel_path,
          begin_time_ms: clip.begin_time_ms,
          end_time_ms: clip.end_time_ms,
          name: source.speaker_name,
          person_public_id: source.person_public_id,
        })),
      );
      if (existing) {
        // The server attached to an in-flight capture; THIS selection did not
        // run. Presenting its result as ours would be a lie.
        setActiveProject(null);
        setError(
          tr(
            "A capture for this project is already running; wait for it to finish, then retry.",
            "该项目已有采集在运行,等它结束后再重试。",
          ),
        );
        return;
      }
      setJobId(job_id);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    } catch (err) {
      setActiveProject(null);
      setError(String(err));
    }
  };

  const finish = () => {
    setResult(null);
    pendingTxnRef.current = null;
    queryClient.invalidateQueries({ queryKey: ["vp-health"] });
    queryClient.invalidateQueries({ queryKey: ["vp-sources", props.personId] });
    props.onSettled?.();
  };

  return (
    <section className="vq-panel vq-panel-plan" id="source-plan">
      <PanelHead
        eyebrow={tr("Sourcing", "补采")}
        title={tr(`Where to top up ${props.personName}`, `给 ${props.personName} 补采样本`)}
      >
        <span className="spacer" />
        <button className="btn ghost" onClick={props.onClose}>
          {tr("Close", "关闭")}
        </button>
      </PanelHead>

      {query.isLoading && (
        <div className="vq-empty">
          {tr("Reading every project transcript…", "正在读取全部项目的转写…")}
        </div>
      )}
      {query.error && <div className="error-box">{String(query.error)}</div>}
      {error && <div className="error-box">{error}</div>}

      {jobId && (
        <JobProgress
          jobId={jobId}
          onDone={() => {}}
          onError={(message) => {
            setError(message);
            setJobId(null);
            setActiveProject(null);
          }}
          onCancelled={() => {
            setError(tr("Capture cancelled.", "采集已取消。"));
            setJobId(null);
            setActiveProject(null);
          }}
        />
      )}

      {data && (
        <>
          <div className="vq-plan-state">
            <span className="mono">
              {tr(
                `Has ${data.matching_sample_count} sample(s) · ${fmtSeconds(data.matching_seconds)} · ${data.project_count} recording(s)`,
                `现有 ${data.matching_sample_count} 条样本 · ${fmtSeconds(data.matching_seconds)} · ${data.project_count} 场录音`,
              )}
            </span>
            {data.deficits.map((kind) => (
              <span className="vq-chip warn" key={kind}>
                {deficitLabel(kind)}
              </span>
            ))}
            <span className="spacer" />
            <span className="subtle mono">
              {tr(
                `${data.scanned_project_count} project(s) scanned`,
                `已扫描 ${data.scanned_project_count} 个项目`,
              )}
            </span>
          </div>

          {data.sources.length === 0 && (
            <div className="vq-empty">
              <b>{tr("Nothing to harvest", "没有可采的素材")}</b>
              {tr(
                "This person is not named in any project that still has unused speech. Name them during a project review first, then come back.",
                "没有任何项目里既标着这个人、又还剩可用发言。先在项目 review 里给对应说话人命名,再回来。",
              )}
            </div>
          )}

          <ol className="vq-sources">
            {data.sources.map((source, index) => (
              <SourceRow
                key={`${source.project_id}:${source.speaker_id}`}
                rank={index + 1}
                source={source}
                audio={audio}
                selected={selected}
                onToggle={toggle}
                busy={jobId !== null}
                capturing={activeProject === source.project_id}
                onCapture={() => void capture(source)}
                onOpenPicker={() =>
                  navigate(
                    `/projects/${encodeURIComponent(source.project_id)}/capture?speaker=${source.speaker_id}`,
                  )
                }
              />
            ))}
          </ol>

          {data.sources.length > 0 &&
            data.new_project_count === 0 &&
            data.deficits.includes("single-source") && (
              <p className="vq-plan-note">
                {tr(
                  "Every source above is a meeting that already contributes samples. More clips from the same recording cannot fix a single-source voiceprint — that needs this person in a different meeting.",
                  "上面每条来源都是已经在贡献样本的那场会。同一场录音里再多采几条,治不了「只有一场录音」——那需要这个人出现在另一场会里。",
                )}
              </p>
            )}

          {data.skipped.length > 0 && (
            <p className="vq-plan-note subtle">
              {tr(
                `Skipped ${data.skipped.length} unreadable project(s): `,
                `跳过了 ${data.skipped.length} 个读不了的项目:`,
              )}
              <span className="mono">
                {data.skipped.map((item) => item.project_id).join(", ")}
              </span>
            </p>
          )}
        </>
      )}

      {result && (
        <CaptureResultModal
          result={result}
          onAccept={async () => {
            await captureAccept(result.transaction_id);
            finish();
          }}
          onRollback={async () => {
            await captureRollback(result.transaction_id);
            finish();
          }}
        />
      )}
    </section>
  );
}

function SourceRow(props: {
  rank: number;
  source: SampleSource;
  audio: ReturnType<typeof useClipAudio>;
  selected: Set<string>;
  onToggle: (relPath: string) => void;
  busy: boolean;
  capturing: boolean;
  onCapture: () => void;
  onOpenPicker: () => void;
}) {
  const { source, audio, selected } = props;
  const [expanded, setExpanded] = useState(false);
  const chips = source.reasons.map(reasonLabel).filter(Boolean) as {
    text: string;
    tone: string;
  }[];
  // Default to the pre-selected clips only. The rest are one click away, but
  // showing a dozen rows per source turns a ranked shortlist back into the
  // transcript-scrolling this panel exists to replace.
  const shortlist = source.clips.filter((clip) => clip.recommended);
  const visible = expanded ? source.clips : shortlist;
  const hidden = source.clips.length - visible.length;
  const pickedCount = source.clips.filter((clip) =>
    selected.has(clip.rel_path),
  ).length;

  return (
    <li className="vq-source">
      <span className="vq-source-rank mono">
        {String(props.rank).padStart(2, "0")}
      </span>
      <div className="vq-source-body">
        <div className="vq-source-head">
          <b>{source.title}</b>
          <span className="vq-chip">
            {tr(`Speaker ${source.speaker_id}`, `说话人 ${source.speaker_id}`)}
            {source.speaker_name ? ` · ${source.speaker_name}` : ""}
          </span>
          <span className={`vq-chip ${source.evidence === "person-map" ? "good" : "warn"}`}>
            {source.evidence === "person-map"
              ? tr("linked", "已关联")
              : tr("by name", "按名字")}
          </span>
        </div>

        <div className="vq-source-supply">
          <span className="vq-source-figure mono">
            {fmtSeconds(source.candidate_seconds)}
          </span>
          <span className="subtle">
            {tr(
              `across ${source.candidate_count} usable utterance(s)`,
              `分布在 ${source.candidate_count} 条可用发言里`,
            )}
          </span>
          {source.matching_sample_count > 0 && (
            <span className="subtle mono">
              {tr(
                `· ${source.matching_sample_count} already taken`,
                `· 已采 ${source.matching_sample_count} 条`,
              )}
            </span>
          )}
          {source.quarantined_sample_count > 0 && (
            <span className="subtle mono warn">
              {tr(
                `· ${source.quarantined_sample_count} quarantined`,
                `· ${source.quarantined_sample_count} 条被隔离`,
              )}
            </span>
          )}
        </div>

        {chips.length > 0 && (
          <div className="vq-source-reasons">
            {chips.map((chip) => (
              <span className={`vq-chip ${chip.tone}`} key={chip.text}>
                {chip.text}
              </span>
            ))}
          </div>
        )}

        {source.clips.length > 0 && pickedCount === 0 && (
          <p className="vq-plan-note">
            {tr(
              "Nothing pre-selected: every clip here has another speaker within half a second, so capturing one would store a mixture of voices. Prefer a source above with clean audio.",
              "一条都没预选:这里每条 clip 前后半秒内都有另一个人在说话,采进去存的会是混合音。优先用上面音频干净的来源。",
            )}
          </p>
        )}

        {source.clips.length === 0 ? (
          <p className="vq-plan-note subtle">
            {tr(
              "No capture plan for this project — open the picker to see why.",
              "这个项目建不出采集计划,打开挑选页看原因。",
            )}
          </p>
        ) : (
          <div className="vq-source-clips">
            {visible.map((clip) => {
              const key = `src:${source.project_id}:${clip.begin_time_ms}`;
              const playing = audio.playingKey === key;
              const on = selected.has(clip.rel_path);
              return (
                <label className={`vq-clip ${on ? "on" : ""}`} key={clip.rel_path}>
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={props.busy}
                    onChange={() => props.onToggle(clip.rel_path)}
                  />
                  <button
                    type="button"
                    className="play-btn"
                    aria-label={tr("Play sample", "试听")}
                    onClick={(event) => {
                      // The button sits inside the label; without this the
                      // click would also toggle the checkbox.
                      event.preventDefault();
                      audio.toggle(
                        key,
                        clipUrl(source.project_id, clip.begin_time_ms, clip.end_time_ms),
                      );
                    }}
                  >
                    {playing ? "⏸" : "▶"}
                  </button>
                  <span className="mono subtle">
                    {fmtClock(clip.begin_time_ms)} · {clip.duration_seconds.toFixed(1)}s
                  </span>
                  {clip.overlap_risk && (
                    <span
                      className="vq-chip warn"
                      title={tr(
                        "Another speaker is within half a second of this clip; the reference audio would be a mixture.",
                        "这段前后半秒内有另一个人在说话,采进去的参考音频会是混合体。",
                      )}
                    >
                      {tr("overlap", "有他人")}
                    </span>
                  )}
                  <span className="vq-clip-text">{clip.text}</span>
                </label>
              );
            })}

            <div className="vq-source-actions">
              {hidden > 0 && (
                <button className="btn ghost" onClick={() => setExpanded(true)}>
                  {tr(`+${hidden} more clips`, `再看 ${hidden} 条候选`)}
                </button>
              )}
              {expanded && (
                <button className="btn ghost" onClick={() => setExpanded(false)}>
                  {tr("Collapse", "收起")}
                </button>
              )}
              <span className="spacer" />
              <button className="btn ghost" onClick={props.onOpenPicker}>
                {tr("Full picker", "完整挑选页")}
              </button>
              <button
                className="btn primary"
                disabled={props.busy || pickedCount === 0}
                onClick={props.onCapture}
              >
                {props.capturing
                  ? tr("Capturing…", "采集中…")
                  : tr(`Capture ${pickedCount}`, `采集 ${pickedCount} 条`)}
              </button>
            </div>
          </div>
        )}
      </div>
    </li>
  );
}
