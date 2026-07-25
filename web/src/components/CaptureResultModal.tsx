import { useState } from "react";
import { tr } from "../lib/i18n";
import { Modal } from "./Modal";
import type { CaptureResult, ScoreChange } from "../api/client";

/**
 * The accept-or-rollback decision a completed capture demands.
 *
 * A capture run leaves a server-side transaction open until it is resolved,
 * and an unresolved one wedges every later store write with HTTP 409. That
 * makes this modal a required step rather than a report, so it lives here
 * where every surface that can start a capture -- the per-project picker and
 * the quality page's sourcing panel -- shows the identical decision.
 */

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

export function CaptureResultModal(props: {
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
