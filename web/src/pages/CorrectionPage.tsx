import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { acceptCorrection, getProposal, polishProject, ApiError } from "../api/client";
import { tr } from "../lib/i18n";
import { JobProgress } from "../components/JobProgress";

export function CorrectionPage() {
  const { ref = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
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
    onSuccess: (r) => setJobId(r.job_id),
  });

  const acceptMut = useMutation({
    mutationFn: () => acceptCorrection(ref, [...selected]),
    onSuccess: () => navigate(`/projects/${ref}/speakers`),
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
              className="btn primary"
              onClick={() => acceptMut.mutate()}
              // Disable while a polish job runs: it is regenerating this very proposal, so
              // accepting the stale one would apply pre-regeneration text and leave a
              // mismatched pending proposal behind.
              disabled={acceptMut.isPending || selected.size === 0 || !!jobId}
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
            onDone={() => {
              setJobId(null);
              queryClient.invalidateQueries({ queryKey: ["proposal", ref] });
            }}
            onError={() => setJobId(null)}
          />
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
            {proposalQuery.data.changes.map((c) => (
              <div key={c.index} className={`change-card ${selected.has(c.index) ? "on" : ""}`}>
                <input
                  type="checkbox"
                  checked={selected.has(c.index)}
                  onChange={() => toggle(c.index)}
                />
                <div className="change-card-body">
                  <div className="change-card-meta subtle mono">
                    {c.speaker_name}
                    {c.change_type && <span className="badge">{c.change_type}</span>}
                  </div>
                  <div className="diff-before">{c.original_text}</div>
                  <div className="diff-after">{c.corrected_text}</div>
                  {c.reason && <div className="subtle" style={{ fontSize: 11.5 }}>{c.reason}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
