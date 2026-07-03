import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  acceptCorrection,
  discardProposal,
  getProposal,
  polishProject,
  ApiError,
  clipUrl,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
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
    if (proposalQuery.data) {
      setSelected(new Set(proposalQuery.data.changes.map((c) => c.index)));
    }
  }, [proposalQuery.data]);

  const polishMut = useMutation({
    mutationFn: () => polishProject(ref),
    onSuccess: (r) => {
      setJobError(null);
      setJobNotice(null);
      setJobId(r.job_id);
    },
  });

  const acceptMut = useMutation({
    // Pass the reviewed proposal's id so the server refuses (409) if it was regenerated since,
    // rather than applying these indices to a different proposal. The Accept button only renders
    // when proposalQuery.data exists, so proposal_id is present here.
    mutationFn: () =>
      acceptCorrection(ref, [...selected], proposalQuery.data!.proposal_id),
    onSuccess: async () => {
      // Accepting rewrites the transcript; drop the cached review so navigating back shows
      // the corrected sentences instead of the still-fresh pre-correction text. The server
      // also archived the proposal, so drop the cached one too.
      await queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
      await queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
      navigate(`/projects/${ref}/speakers`);
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
            onClick={() => polishMut.mutate()}
            disabled={polishMut.isPending || !!jobId}
          >
            {tr("Generate polish", "生成润色")}
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
              setJobId(null);
            }}
            // Keep the polish failure visible after the job panel unmounts.
            onError={(e) => {
              setJobError(e);
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
                        <button
                          className="sentence-id sentence-id-button"
                          onClick={() =>
                            navigate(
                              `/projects/${encodeURIComponent(projectFromSentenceRef(c.sentence_ref!))}/speakers?sentence=${encodeURIComponent(c.sentence_ref!)}`,
                            )
                          }
                          title={tr("Locate in speaker review", "在 speaker review 中定位")}
                        >
                          {c.sentence_ref}
                        </button>
                      )}
                      {hasAudio && <span>{fmtMs(c.begin_time_ms!)}</span>}
                      {c.speaker_name}
                      {c.change_type && <span className="badge">{c.change_type}</span>}
                    </div>
                    <div className="diff-before">{c.original_text}</div>
                    <div className="diff-after">{c.corrected_text}</div>
                    {c.reason && (
                      <div className="subtle" style={{ fontSize: 11.5 }}>
                        {c.reason}
                      </div>
                    )}
                    {playing && (
                      <div className="seg-progress">
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
