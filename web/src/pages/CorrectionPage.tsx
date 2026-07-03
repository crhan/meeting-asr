import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  acceptCorrection,
  discardProposal,
  getJob,
  getProposal,
  polishProject,
  ApiError,
  clipUrl,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { reportGlobalError, reportGlobalNotice } from "../lib/globalError";
import { diffPair } from "../lib/textDiff";
import { JobProgress } from "../components/JobProgress";
import { useClipAudio } from "../lib/useClipAudio";

function fmtMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function projectFromSentenceRef(sentenceRef: string): string {
  return sentenceRef.split("#", 1)[0];
}

const polishJobKey = (ref: string) => `masr-polish-job:${ref}`;
// proposal_id is a content hash, so a regenerated proposal never inherits a stale selection.
const selectionKey = (ref: string, proposalId: string) =>
  `masr-correction-sel:${ref}:${proposalId}`;

/** First tag of a possibly multi-tagged change_type ('dup|filler' -> 'dup'), '' -> other. */
function primaryChangeType(changeType: string): string {
  const first = changeType.split("|", 1)[0].trim();
  return first || "other";
}

export function CorrectionPage() {
  const { ref = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const audio = useClipAudio();
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [jobNotice, setJobNotice] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const proposalQuery = useQuery({
    queryKey: ["proposal", ref],
    queryFn: () => getProposal(ref),
    retry: false,
  });

  useEffect(() => {
    if (!proposalQuery.data) return;
    const { changes, proposal_id } = proposalQuery.data;
    const valid = new Set(changes.map((c) => c.index));
    // Restore a persisted selection (survives the sentence-locator jump / reload);
    // fall back to select-all for a freshly reviewed proposal.
    try {
      const stored = sessionStorage.getItem(selectionKey(ref, proposal_id));
      if (stored) {
        const indices = (JSON.parse(stored) as number[]).filter((i) => valid.has(i));
        setSelected(new Set(indices));
        return;
      }
    } catch {
      // corrupt storage entry; fall through to select-all
    }
    setSelected(valid);
  }, [proposalQuery.data, ref]);

  // Persist the working selection per (project, proposal) so it survives navigation.
  useEffect(() => {
    if (!proposalQuery.data) return;
    sessionStorage.setItem(
      selectionKey(ref, proposalQuery.data.proposal_id),
      JSON.stringify([...selected]),
    );
  }, [selected, proposalQuery.data, ref]);

  // The polish job survives this component (it runs server-side); persist its id so a
  // reload / tab switch re-attaches to the live progress instead of going blind and
  // letting the button re-trigger a paid LLM run.
  useEffect(() => {
    const stored = sessionStorage.getItem(polishJobKey(ref));
    if (!stored) return;
    getJob(stored)
      .then((job) => {
        if (job.status === "queued" || job.status === "running") setJobId(stored);
        else sessionStorage.removeItem(polishJobKey(ref));
      })
      // 404: jobs live in server memory; a restart forgot it.
      .catch(() => sessionStorage.removeItem(polishJobKey(ref)));
  }, [ref]);

  const polishMut = useMutation({
    mutationFn: () => polishProject(ref),
    onSuccess: (r) => {
      setJobError(null);
      setJobNotice(
        // The backend deduplicates onto an in-flight polish for this project; say so
        // instead of looking like a fresh (double-billed) run started.
        r.existing
          ? tr("A polish is already running; re-attached to it.", "润色已在进行中，已重新挂接进度。")
          : null,
      );
      sessionStorage.setItem(polishJobKey(ref), r.job_id);
      setJobId(r.job_id);
    },
  });

  const acceptMut = useMutation({
    // Pass the reviewed proposal's id so the server refuses (409) if it was regenerated since,
    // rather than applying these indices to a different proposal. The Accept button only renders
    // when proposalQuery.data exists, so proposal_id is present here.
    mutationFn: () =>
      acceptCorrection(ref, [...selected], proposalQuery.data!.proposal_id),
    onSuccess: async (res) => {
      if (proposalQuery.data)
        sessionStorage.removeItem(selectionKey(ref, proposalQuery.data.proposal_id));
      // The lexicon learning side effect is otherwise invisible (we navigate away).
      if (res.learned_count > 0) {
        reportGlobalNotice(
          tr(
            `Applied ${res.change_count} change(s); learned ${res.learned_count} context(s) into the lexicon.`,
            `已应用 ${res.change_count} 条修改；已学习 ${res.learned_count} 条语境进纠错词库。`,
          ),
        );
      }
      // Accepting rewrites the transcript; drop the cached review so navigating back shows
      // the corrected sentences instead of the still-fresh pre-correction text. The server
      // also archived the proposal, so drop the cached one too.
      await queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
      await queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
      navigate(`/projects/${ref}/speakers`);
    },
    onError: (e) => {
      // 409 = the proposal was regenerated since review; reload it and ask to re-select.
      if (e instanceof ApiError && e.status === 409) {
        setJobError(
          tr(
            "The proposal changed since you reviewed it — it was reloaded; please re-select and accept again.",
            "提案在审阅后被更新——已重新加载，请重新勾选后再应用。",
          ),
        );
        void queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
        return;
      }
      // A custom onError replaces the global default; report explicitly.
      reportGlobalError(tr("Operation failed: ", "操作失败：") + (e as Error).message);
    },
  });

  const discardMut = useMutation({
    mutationFn: () => discardProposal(ref, proposalQuery.data!.proposal_id),
    onSuccess: async () => {
      setJobNotice(
        tr("Proposal discarded; no changes were applied.", "已放弃提案，未应用任何修改。"),
      );
      await queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
    },
  });

  const noProposal =
    proposalQuery.error instanceof ApiError && proposalQuery.error.status === 404;

  const toggle = (i: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  // Indices grouped by primary change type, for batch check/uncheck chips.
  const typeGroups = useMemo(() => {
    const groups = new Map<string, number[]>();
    for (const change of proposalQuery.data?.changes ?? []) {
      const key = primaryChangeType(change.change_type);
      const bucket = groups.get(key);
      if (bucket) bucket.push(change.index);
      else groups.set(key, [change.index]);
    }
    return groups;
  }, [proposalQuery.data]);

  const toggleGroup = (indices: number[]) =>
    setSelected((prev) => {
      const next = new Set(prev);
      const allOn = indices.every((i) => next.has(i));
      for (const i of indices) {
        if (allOn) next.delete(i);
        else next.add(i);
      }
      return next;
    });

  // Character-level diffs, one pass per proposal load (not per re-render).
  const diffs = useMemo(
    () =>
      new Map(
        (proposalQuery.data?.changes ?? []).map((change) => [
          change.index,
          diffPair(change.original_text, change.corrected_text),
        ]),
      ),
    [proposalQuery.data],
  );

  return (
    <div>
      <div className="review-head" style={{ margin: "-18px -18px 14px", borderRadius: 0 }}>
        <div>
          <h1>{tr("Transcript correction", "文字纠错")}</h1>
          <div className="subtle mono">{ref}</div>
        </div>
        <div className="row gap">
          <button className="btn ghost" onClick={() => navigate(`/projects/${ref}/speakers`)}>
            {tr("Back", "返回")}
          </button>
          <button
            className="btn"
            onClick={async () => {
              // Regenerating overwrites the pending proposal (and the staged selection).
              if (
                proposalQuery.data &&
                !(await confirmDialog({
                  message: tr(
                    "A pending proposal already exists; regenerating replaces it and resets your selection.",
                    "已有待处理的提案；重新生成会覆盖它并重置当前勾选。",
                  ),
                  confirmLabel: tr("Regenerate", "重新生成"),
                  danger: true,
                }))
              )
                return;
              polishMut.mutate();
            }}
            disabled={polishMut.isPending || !!jobId}
          >
            {proposalQuery.data ? tr("Regenerate polish", "重新生成润色") : tr("Generate polish", "生成润色")}
          </button>
          {proposalQuery.data && (
            <button
              className="btn ghost"
              disabled={discardMut.isPending || !!jobId || proposalQuery.isFetching}
              onClick={async () => {
                if (
                  await confirmDialog({
                    message: tr(
                      "Discard this proposal? No changes will be applied; you can regenerate at any time.",
                      "放弃这份提案？不会应用任何修改，之后可以随时重新生成。",
                    ),
                    confirmLabel: tr("Discard", "放弃"),
                  })
                )
                  discardMut.mutate();
              }}
            >
              {discardMut.isPending ? tr("Discarding…", "放弃中…") : tr("Discard", "放弃提案")}
            </button>
          )}
          {proposalQuery.data && (
            <button
              className="btn primary"
              onClick={() => acceptMut.mutate()}
              // Disable while a polish job runs OR while the proposal is (re)fetching: a
              // regenerate rewrites this very proposal file, so accepting against the still-
              // displayed stale proposal would send old selection indices that the backend
              // applies to the freshly written proposal -- the wrong subset of changes.
              disabled={
                acceptMut.isPending ||
                selected.size === 0 ||
                !!jobId ||
                proposalQuery.isFetching
              }
            >
              {acceptMut.isPending
                ? tr("Accepting…", "应用中…")
                : tr(`Accept ${selected.size}`, `应用 ${selected.size} 条`)}
            </button>
          )}
        </div>
      </div>

      {jobId && (
        <div style={{ marginBottom: 14 }}>
          <JobProgress
            jobId={jobId}
            onDone={async (result) => {
              // The polish backend catches model failures (missing key, network, LLM error)
              // into `model_error` and still ends the job "done" -- the CLI prints it as
              // "Model fallback: ...". Without surfacing it here, an LLM failure looks like
              // "No pending proposal" with zero explanation.
              const summary = (result ?? {}) as {
                proposed_change_count?: number;
                model?: string;
                model_error?: string | null;
              };
              if (summary.model_error) {
                setJobError(
                  tr("Polish model failed: ", "润色模型失败：") + summary.model_error,
                );
              } else if (summary.proposed_change_count === 0) {
                // Zero proposed changes is a legitimate outcome, not a failure -- without
                // this the panel just vanishes and the page shows the empty state with
                // no explanation of what the LLM concluded.
                setJobNotice(
                  tr(
                    "Polish proposed no changes — the transcript looks clean.",
                    "润色未提出任何修改——转写看起来已经很干净。",
                  ),
                );
              }
              // Refetch the regenerated proposal BEFORE clearing jobId. Clearing first would
              // re-enable Accept against the stale proposal during the refetch window; keeping
              // jobId set holds the job panel (and the disabled Accept) until fresh data lands.
              await queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
              sessionStorage.removeItem(polishJobKey(ref));
              setJobId(null);
            }}
            // Keep the polish failure visible after the job panel unmounts.
            onError={(e) => {
              setJobError(e);
              sessionStorage.removeItem(polishJobKey(ref));
              setJobId(null);
            }}
            onCancelled={() => {
              setJobNotice(tr("Polish cancelled.", "润色已取消。"));
              sessionStorage.removeItem(polishJobKey(ref));
              setJobId(null);
            }}
          />
        </div>
      )}

      {jobError && !jobId && (
        <div className="error-box" style={{ marginBottom: 14 }}>
          {jobError}
        </div>
      )}

      {jobNotice && !jobId && !jobError && (
        <div className="notice-box" style={{ marginBottom: 14 }}>
          {jobNotice}
        </div>
      )}

      {proposalQuery.isLoading && <div className="placeholder">{tr("Loading…", "加载中…")}</div>}

      {proposalQuery.isError && !noProposal && (
        // Non-404 load failure (e.g. a malformed proposal file deliberately 500s):
        // without this branch the page is just blank.
        <div className="error-box" style={{ marginBottom: 14 }}>
          {(proposalQuery.error as Error).message}
          <div style={{ marginTop: 8 }}>
            <button className="btn ghost" onClick={() => proposalQuery.refetch()}>
              {tr("Retry", "重试")}
            </button>
          </div>
        </div>
      )}

      {noProposal && !jobId && (
        <div className="placeholder">
          {tr(
            "No pending proposal. Click “Generate polish” to create one.",
            "没有待处理的提案。点「生成润色」创建。",
          )}
        </div>
      )}

      {proposalQuery.data && (
        <div>
          <div className="subtle mono" style={{ marginBottom: 10 }}>
            {proposalQuery.data.model} · {proposalQuery.data.change_count}{" "}
            {tr("proposed changes", "条建议")}
          </div>
          <div className="capture-toolbar">
            <button
              className="chip"
              onClick={() =>
                setSelected(new Set(proposalQuery.data!.changes.map((c) => c.index)))
              }
            >
              {tr("Select all", "全选")}
            </button>
            <button className="chip" onClick={() => setSelected(new Set())}>
              {tr("Clear", "清空")}
            </button>
            {[...typeGroups.entries()].map(([type, indices]) => {
              const onCount = indices.filter((i) => selected.has(i)).length;
              return (
                <button
                  key={type}
                  className={`chip ${onCount === indices.length ? "on" : ""}`}
                  title={tr(
                    "Toggle all changes of this type.",
                    "整组勾选/取消该类型的修改。",
                  )}
                  onClick={() => toggleGroup(indices)}
                >
                  {type} {onCount}/{indices.length}
                </button>
              );
            })}
          </div>
          <div className="changes">
            {proposalQuery.data.changes.map((c) => {
              const hasAudio = c.begin_time_ms != null && c.end_time_ms != null;
              const audioKey = hasAudio
                ? `correction:${c.index}:${c.begin_time_ms}:${c.end_time_ms}`
                : "";
              const playing = audio.playingKey === audioKey;
              return (
                <div key={c.index} className={`change-card ${selected.has(c.index) ? "on" : ""}`}>
                  <input
                    type="checkbox"
                    checked={selected.has(c.index)}
                    onChange={() => toggle(c.index)}
                  />
                  {hasAudio && (
                    <button
                      className="play-btn"
                      onClick={() =>
                        audio.toggle(
                          audioKey,
                          clipUrl(ref, c.begin_time_ms!, c.end_time_ms!),
                        )
                      }
                      title={tr("Play original audio", "播放原音频")}
                      aria-label={tr("Play original audio", "播放原音频")}
                    >
                      {playing ? "⏸" : "▶"}
                    </button>
                  )}
                  <div className="change-card-body">
                    <div className="change-card-meta subtle mono">
                      {c.sentence_ref && (
                        // New tab: the point is to peek at surrounding context; an in-tab
                        // navigation would throw away this page's checkbox review state.
                        <a
                          className="sentence-id sentence-id-button"
                          target="_blank"
                          rel="noreferrer"
                          href={`/projects/${encodeURIComponent(projectFromSentenceRef(c.sentence_ref))}/speakers?sentence=${encodeURIComponent(c.sentence_ref)}`}
                          title={tr(
                            "Locate in speaker review (new tab)",
                            "在 speaker review 中定位（新标签页）",
                          )}
                        >
                          {c.sentence_ref}
                        </a>
                      )}
                      {hasAudio && <span>{fmtMs(c.begin_time_ms!)}</span>}
                      {c.speaker_name}
                      {c.change_type && <span className="badge">{c.change_type}</span>}
                    </div>
                    <div className="diff-before">
                      {(diffs.get(c.index)?.before ?? [{ text: c.original_text, changed: false }]).map(
                        (segment, i) =>
                          segment.changed ? (
                            <del key={i}>{segment.text}</del>
                          ) : (
                            <span key={i}>{segment.text}</span>
                          ),
                      )}
                    </div>
                    <div className="diff-after">
                      {(diffs.get(c.index)?.after ?? [{ text: c.corrected_text, changed: false }]).map(
                        (segment, i) =>
                          segment.changed ? (
                            <ins key={i}>{segment.text}</ins>
                          ) : (
                            <span key={i}>{segment.text}</span>
                          ),
                      )}
                    </div>
                    {c.reason && (
                      <div className="subtle" style={{ fontSize: 11.5 }}>
                        {c.reason}
                      </div>
                    )}
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
      )}
    </div>
  );
}
