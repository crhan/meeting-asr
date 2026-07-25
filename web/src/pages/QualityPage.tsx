import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getEmbedBacklog,
  getLibraryHealth,
  getMatchThreshold,
  runEmbedBackfill,
  setMatchThreshold,
  type Calibration,
  type LibraryHealth,
  type LibraryIssue,
  type PersonHealth,
  type ThresholdCost,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";

const AVAILABILITY_ORDER = ["unusable", "fragile", "ok"] as const;

function availabilityLabel(value: string): string {
  if (value === "unusable") return tr("Unusable", "不可用");
  if (value === "fragile") return tr("Fragile", "脆弱");
  return tr("Healthy", "健康");
}

function severityLabel(value: string): string {
  if (value === "critical") return tr("Critical", "严重");
  if (value === "warning") return tr("Warning", "警告");
  return tr("Info", "提示");
}

function fmtSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function num(context: Record<string, number | string>, key: string): number {
  const value = context[key];
  return typeof value === "number" ? value : 0;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Restate an issue in the active locale.
 *
 * The backend ships English prose plus the raw numbers behind it; translating
 * whole sentences client-side would duplicate the numbers, so we re-render
 * from `kind` + `context` and fall back to the server text for any kind this
 * build does not know about yet.
 */
function issueText(issue: LibraryIssue): { title: string; detail: string } {
  const c = issue.context;
  const name = String(c.name ?? issue.person_name ?? "");
  switch (issue.kind) {
    case "no-enabled-samples":
      return {
        title: tr(issue.title, `${name} 没有任何样本参与匹配`),
        detail: tr(
          issue.detail,
          `全部 ${num(c, "total_sample_count")} 个样本都处于隔离/拒绝/失效状态,这个人永远不会被匹配到。样本一致性检查显示「无异常」只是因为已经没有样本可供挑错。`,
        ),
      };
    case "missing-embeddings":
      return {
        title: tr(issue.title, `${name} 在当前模型下没有任何向量`),
        detail: tr(
          issue.detail,
          `有 ${num(c, "missing_embedding_count")} 个可用样本,但没有一个生成了当前模型的向量,因此这个人完全不在匹配池里。通常是切换 provider 后没有重新嵌入。`,
        ),
      };
    case "partial-embeddings":
      return {
        title: tr(issue.title, `${name} 有部分样本未嵌入`),
        detail: tr(
          issue.detail,
          `${num(c, "enabled_sample_count")} 个可用样本中有 ${num(c, "missing_embedding_count")} 个没有当前模型的向量,不参与匹配。`,
        ),
      };
    case "fragile-cluster":
      return {
        title: tr(
          issue.title,
          `${name} 只有 ${num(c, "matching_sample_count")} 个匹配样本`,
        ),
        detail: tr(
          issue.detail,
          `低于 ${num(c, "min_cluster_size")} 个样本时,质心刻画的是某一次录音的特点而不是这个人的声音;而且一致性检查也无法发现离群样本 —— 因为每个样本都在定义它自己被比较的那个质心。`,
        ),
      };
    case "short-audio":
      return {
        title: tr(
          issue.title,
          `${name} 只有 ${num(c, "matching_seconds").toFixed(0)}s 有效音频`,
        ),
        detail: tr(
          issue.detail,
          `决定嵌入稳定性的是有效发声总时长,不是样本条数。建议在多句话中累计到 ${num(c, "min_healthy_seconds").toFixed(0)}s 以上。`,
        ),
      };
    case "single-source":
      return {
        title: tr(issue.title, `${name} 只来自一次录音`),
        detail: tr(
          issue.detail,
          "全部匹配样本来自同一个项目,声纹里同时编码了那个房间、那只麦克风和当时的状态。从另一场会议补几条样本可以提升泛化能力。",
        ),
      };
    case "threshold-too-low":
      return {
        title: tr(
          issue.title,
          `阈值 ${num(c, "current_threshold").toFixed(2)} 会接受 ${num(c, "current_false_accept_count")} 个错人分数`,
        ),
        detail: tr(
          issue.detail,
          `${pct(num(c, "current_false_accept_rate"))} 的「最像的其他人」分数达到了当前阈值,管线可能自动挂上错误的名字。建议阈值 ${num(c, "suggested_threshold").toFixed(3)}。`,
        ),
      };
    case "threshold-too-high":
      return {
        title: tr(
          issue.title,
          `阈值 ${num(c, "current_threshold").toFixed(2)} 会拒掉 ${pct(num(c, "current_false_reject_rate"))} 的正确匹配`,
        ),
        detail: tr(
          issue.detail,
          `${num(c, "genuine_count")} 个同人分数里有 ${num(c, "current_false_reject_count")} 个低于当前阈值,这些都要靠人工命名。改到 ${num(c, "suggested_threshold").toFixed(3)} 可以救回 ${num(c, "current_false_reject_count") - num(c, "suggested_false_reject_count")} 个,同时误纳仍为 ${num(c, "suggested_false_accept_count")} 个。`,
        ),
      };
    default:
      return { title: issue.title, detail: issue.detail };
  }
}

function suggestionReason(calibration: Calibration): string {
  const impostorMax = calibration.impostor?.max ?? 0;
  const genuineP5 = calibration.genuine?.p5 ?? 0;
  switch (calibration.suggested_kind) {
    case "gap":
      return tr(
        calibration.suggested_reason,
        `取「最差异人分数 ${impostorMax.toFixed(3)}」与「同人分数 5 分位 ${genuineP5.toFixed(3)}」之间空档的中点,两侧都留出余量。`,
      );
    case "single-person":
      return tr(
        calibration.suggested_reason,
        "库里只有一个人有向量,没有异人证据,这个建议只能防止拒掉他本人。",
      );
    case "overlap":
      return tr(
        calibration.suggested_reason,
        "同人与异人分数已经重叠,没有任何阈值能干净地把两者分开;这是等错误率下的折中。此时改善样本质量比调阈值更有效。",
      );
    default:
      return tr(calibration.suggested_reason, "证据不足,无法给出建议阈值。");
  }
}

function couplingWarning(kind: string, fallback: string): string {
  if (kind === "strong-margin-dead")
    return tr(
      fallback,
      "该阈值已降到 strong-margin 救援分(0.65)及以下:这条「低于阈值但明显领先仍接受」的规则会变成死代码,因为它本来要救的都已经被直接接受了。",
    );
  if (kind === "below-crosstalk-floor")
    return tr(
      fallback,
      "该阈值已降到 crosstalk 分数下限(0.5)及以下:接受层现在能接受的说话人,同时也符合串场标记的条件。",
    );
  return fallback;
}

/**
 * Price a candidate threshold locally.
 *
 * The raw score populations ship with the calibration payload precisely so the
 * slider can answer "what does this cost" on every pixel of travel without a
 * round trip -- a threshold the user cannot feel out is a threshold they will
 * not touch.
 */
function costAt(calibration: Calibration, threshold: number): ThresholdCost {
  const rejected = calibration.genuine_scores.filter((s) => s < threshold).length;
  const accepted = calibration.impostor_scores.filter((s) => s >= threshold).length;
  return {
    threshold,
    false_reject_count: rejected,
    false_reject_rate: calibration.genuine_scores.length
      ? rejected / calibration.genuine_scores.length
      : 0,
    false_accept_count: accepted,
    false_accept_rate: calibration.impostor_scores.length
      ? accepted / calibration.impostor_scores.length
      : 0,
  };
}

export function QualityPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const healthQuery = useQuery({ queryKey: ["vp-health"], queryFn: getLibraryHealth });
  const thresholdQuery = useQuery({
    queryKey: ["vp-threshold"],
    queryFn: getMatchThreshold,
  });
  const backlogQuery = useQuery({
    queryKey: ["vp-embed-backlog"],
    queryFn: getEmbedBacklog,
  });
  const [toast, setToast] = useState<string | null>(null);

  const embedMut = useMutation({
    mutationFn: runEmbedBackfill,
    onSuccess: () => {
      setToast(
        tr(
          "Embedding started — follow it in the jobs panel.",
          "已开始补齐嵌入 — 可在任务面板查看进度。",
        ),
      );
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const thresholdMut = useMutation({
    mutationFn: (value: number | null) => setMatchThreshold(value),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["vp-threshold"] });
      queryClient.invalidateQueries({ queryKey: ["vp-health"] });
      setToast(
        result.configured === null
          ? tr("Threshold reset to the built-in default.", "阈值已恢复为内置默认值。")
          : tr(
              `Threshold set to ${result.configured.toFixed(3)}.`,
              `阈值已设为 ${result.configured.toFixed(3)}。`,
            ),
      );
    },
  });

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 6000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const health = healthQuery.data;
  const isLoading = healthQuery.isLoading || thresholdQuery.isLoading;
  const error = healthQuery.error || thresholdQuery.error;

  const runAction = async (issue: LibraryIssue) => {
    if (issue.action === "embed") {
      embedMut.mutate();
      return;
    }
    if (issue.action === "review-samples" && issue.person_public_id) {
      navigate(`/voiceprints?person=${encodeURIComponent(issue.person_public_id)}`);
      return;
    }
    if (issue.action === "capture") {
      navigate("/projects");
      return;
    }
    if (issue.action === "set-threshold") {
      document
        .getElementById("threshold-card")
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <div className="pad quality-page">
      <div className="vp-head">
        <h1 style={{ marginRight: 8 }}>{tr("Library quality", "声纹库质量")}</h1>
        <span className="spacer" />
        <Link className="btn ghost" to="/voiceprints">
          {tr("Manage library", "管理声纹库")}
        </Link>
      </div>

      {toast && <div className="notice-box">{toast}</div>}
      {isLoading && <div className="placeholder">{tr("Loading…", "加载中…")}</div>}
      {error && <div className="error-box">{String(error)}</div>}

      {health && (
        <>
          <LibraryVitals health={health} backlog={backlogQuery.data} />
          {health.calibration && (
            <ThresholdCard
              calibration={health.calibration}
              configured={thresholdQuery.data?.configured ?? null}
              builtinDefault={thresholdQuery.data?.default ?? 0.75}
              warnings={thresholdQuery.data?.warnings ?? []}
              warningKinds={thresholdQuery.data?.warning_kinds ?? []}
              pending={thresholdMut.isPending}
              onApply={async (value) => {
                const ok = await confirmDialog({
                  title: tr("Change match threshold?", "修改匹配阈值?"),
                  message:
                    value === null
                      ? tr(
                          "Restore the built-in default threshold?",
                          "恢复内置默认阈值?",
                        )
                      : tr(
                          `Set the acceptance threshold to ${value.toFixed(3)}? This affects every future match and rematch, including automatic naming during project runs. You can reset it at any time.`,
                          `将接受阈值设为 ${value.toFixed(3)}?此设置影响之后所有匹配与重匹配,包括 project run 时的自动命名。随时可以恢复默认。`,
                        ),
                });
                if (ok) thresholdMut.mutate(value);
              }}
            />
          )}
          <IssueQueue
            issues={health.issues}
            onAction={runAction}
            busy={embedMut.isPending}
          />
          <PeopleMatrix people={health.people} />
        </>
      )}
    </div>
  );
}

function LibraryVitals(props: {
  health: LibraryHealth;
  backlog?: { missing_sample_count: number; person_count: number };
}) {
  const { health, backlog } = props;
  const buckets = AVAILABILITY_ORDER.map((key) => ({
    key,
    count: health.people.filter((person) => person.availability === key).length,
  }));
  const total = health.people.length || 1;
  return (
    <section className="quality-card">
      <div className="quality-card-head">
        <h2>{tr("Library vitals", "库体检")}</h2>
        <span className="subtle mono">{health.model}</span>
      </div>
      <div className="quality-stats">
        <Stat
          value={`${health.usable_person_count}/${health.person_count}`}
          label={tr("people matchable", "人可参与匹配")}
          tone={health.usable_person_count < health.person_count ? "bad" : "good"}
        />
        <Stat
          value={String(health.matching_sample_count)}
          label={tr("matching samples", "匹配样本")}
        />
        <Stat
          value={fmtSeconds(health.matching_seconds)}
          label={tr("matching audio", "匹配音频")}
        />
        <Stat
          value={String(health.critical_count)}
          label={tr("critical issues", "严重问题")}
          tone={health.critical_count > 0 ? "bad" : "good"}
        />
      </div>
      <div className="quality-healthbar" role="img">
        {buckets.map(
          (bucket) =>
            bucket.count > 0 && (
              <span
                key={bucket.key}
                className={`quality-healthbar-seg av-${bucket.key}`}
                style={{ width: `${(bucket.count / total) * 100}%` }}
                title={`${availabilityLabel(bucket.key)}: ${bucket.count}`}
              />
            ),
        )}
      </div>
      <div className="row gap quality-legend">
        {buckets.map((bucket) => (
          <span key={bucket.key} className="subtle">
            <i className={`quality-dot av-${bucket.key}`} />
            {availabilityLabel(bucket.key)} {bucket.count}
          </span>
        ))}
        {backlog && backlog.missing_sample_count > 0 && (
          <span className="warn">
            {tr(
              `${backlog.missing_sample_count} sample(s) still need embedding`,
              `${backlog.missing_sample_count} 个样本待嵌入`,
            )}
          </span>
        )}
      </div>
    </section>
  );
}

function Stat(props: { value: string; label: string; tone?: "good" | "bad" }) {
  return (
    <div className="quality-stat">
      <div className={`quality-stat-value ${props.tone ?? ""}`}>{props.value}</div>
      <div className="quality-stat-label">{props.label}</div>
    </div>
  );
}

function ThresholdCard(props: {
  calibration: Calibration;
  configured: number | null;
  builtinDefault: number;
  warnings: string[];
  warningKinds: string[];
  pending: boolean;
  onApply: (value: number | null) => void;
}) {
  const { calibration, configured, builtinDefault, warnings, warningKinds, pending } =
    props;
  const [candidate, setCandidate] = useState(calibration.current_threshold);

  useEffect(() => {
    setCandidate(calibration.current_threshold);
  }, [calibration.current_threshold]);

  const cost = useMemo(() => costAt(calibration, candidate), [calibration, candidate]);
  const currentCost = calibration.current_cost;
  const suggested = calibration.suggested_threshold;
  const dirty = Math.abs(candidate - calibration.current_threshold) > 1e-6;

  return (
    <section className="quality-card" id="threshold-card">
      <div className="quality-card-head">
        <h2>{tr("Acceptance threshold", "接受阈值")}</h2>
        <span className="subtle">
          {tr(
            `${calibration.genuine_scores.length} same-person vs ${calibration.impostor_scores.length} wrong-person scores`,
            `${calibration.genuine_scores.length} 个同人分数 vs ${calibration.impostor_scores.length} 个异人分数`,
          )}
        </span>
      </div>

      <ScoreHistogram
        calibration={calibration}
        candidate={candidate}
        suggested={suggested}
      />

      <input
        className="quality-slider"
        type="range"
        min={0.3}
        max={0.95}
        step={0.005}
        value={candidate}
        onChange={(event) => setCandidate(Number(event.target.value))}
        aria-label={tr("Candidate threshold", "候选阈值")}
      />

      <div className="quality-cost-row">
        <CostReadout
          title={tr("At this threshold", "此阈值下")}
          threshold={candidate}
          cost={cost}
          highlight
        />
        {currentCost && (
          <CostReadout
            title={tr("Active now", "当前生效")}
            threshold={currentCost.threshold}
            cost={currentCost}
          />
        )}
        {suggested !== null && (
          <CostReadout
            title={tr("Suggested", "建议值")}
            threshold={suggested}
            cost={costAt(calibration, suggested)}
          />
        )}
      </div>

      {suggested !== null && (
        <p className="subtle quality-reason">
          {tr("Why this suggestion: ", "建议理由:")}
          {suggestionReason(calibration)}
        </p>
      )}
      {calibration.low_confidence && (
        <p className="warn">
          {tr(
            "Small library — treat the suggestion as a direction, not a precise operating point. Adding samples will move it.",
            "库样本量偏小 —— 建议值只指方向,不是精确工作点。补样本后会变化。",
          )}
        </p>
      )}
      {warnings.map((warning, index) => (
        <p className="warn" key={warning}>
          {couplingWarning(warningKinds[index] ?? "", warning)}
        </p>
      ))}

      <div className="row gap">
        <button
          className="btn primary"
          disabled={!dirty || pending}
          onClick={() => props.onApply(Number(candidate.toFixed(3)))}
        >
          {tr(
            `Apply ${candidate.toFixed(3)}`,
            `应用 ${candidate.toFixed(3)}`,
          )}
        </button>
        {suggested !== null && (
          <button
            className="btn"
            disabled={pending}
            onClick={() => setCandidate(suggested)}
          >
            {tr("Preview suggested", "预览建议值")}
          </button>
        )}
        <button
          className="btn ghost"
          disabled={pending || configured === null}
          onClick={() => props.onApply(null)}
          title={tr(
            `Built-in default is ${builtinDefault}`,
            `内置默认值为 ${builtinDefault}`,
          )}
        >
          {tr("Reset to default", "恢复默认")}
        </button>
        <span className="spacer" />
        <span className="subtle mono">
          {configured === null
            ? tr(`unset (default ${builtinDefault})`, `未配置(默认 ${builtinDefault})`)
            : tr(`configured ${configured}`, `已配置 ${configured}`)}
        </span>
      </div>
    </section>
  );
}

function CostReadout(props: {
  title: string;
  threshold: number;
  cost: ThresholdCost;
  highlight?: boolean;
}) {
  const { cost } = props;
  return (
    <div className={`quality-cost ${props.highlight ? "on" : ""}`}>
      <div className="quality-cost-title">{props.title}</div>
      <div className="quality-cost-threshold mono">{props.threshold.toFixed(3)}</div>
      <div className="quality-cost-line">
        <span className={cost.false_reject_count > 0 ? "warn" : "subtle"}>
          {tr("misses", "误拒")} {cost.false_reject_count} (
          {(cost.false_reject_rate * 100).toFixed(0)}%)
        </span>
      </div>
      <div className="quality-cost-line">
        <span className={cost.false_accept_count > 0 ? "danger-text" : "subtle"}>
          {tr("wrong names", "误纳")} {cost.false_accept_count} (
          {(cost.false_accept_rate * 100).toFixed(0)}%)
        </span>
      </div>
    </div>
  );
}

/**
 * Two overlaid score histograms with threshold markers.
 *
 * The gap between the impostor mass and the genuine mass is the whole story:
 * a threshold sitting inside that gap is safe, one sitting inside the genuine
 * mass is silently rejecting correct matches. Drawing both populations on one
 * axis makes that visible in a way any single summary number cannot.
 */
function ScoreHistogram(props: {
  calibration: Calibration;
  candidate: number;
  suggested: number | null;
}) {
  const { calibration, candidate, suggested } = props;
  const min = 0.3;
  const max = 0.95;
  const bins = 26;
  const width = 100;
  const height = 34;

  const histogram = (scores: number[]) => {
    const counts = new Array(bins).fill(0);
    for (const score of scores) {
      const clamped = Math.min(Math.max(score, min), max - 1e-9);
      const index = Math.floor(((clamped - min) / (max - min)) * bins);
      counts[index] += 1;
    }
    return counts;
  };

  const genuine = histogram(calibration.genuine_scores);
  const impostor = histogram(calibration.impostor_scores);
  const peak = Math.max(1, ...genuine, ...impostor);
  const x = (value: number) => ((value - min) / (max - min)) * width;

  const bars = (counts: number[], className: string) =>
    counts.map((count, index) =>
      count === 0 ? null : (
        <rect
          key={`${className}-${index}`}
          className={className}
          x={(index / bins) * width}
          y={height - (count / peak) * height}
          width={width / bins - 0.35}
          height={(count / peak) * height}
        />
      ),
    );

  return (
    <div className="quality-histogram">
      <svg viewBox={`0 0 ${width} ${height + 8}`} preserveAspectRatio="none">
        {bars(impostor, "hist-impostor")}
        {bars(genuine, "hist-genuine")}
        {suggested !== null && (
          <line
            className="hist-mark suggested"
            x1={x(suggested)}
            x2={x(suggested)}
            y1={0}
            y2={height}
          />
        )}
        <line
          className="hist-mark current"
          x1={x(calibration.current_threshold)}
          x2={x(calibration.current_threshold)}
          y1={0}
          y2={height}
        />
        <line
          className="hist-mark candidate"
          x1={x(candidate)}
          x2={x(candidate)}
          y1={0}
          y2={height}
        />
      </svg>
      <div className="quality-histogram-legend subtle">
        <span>
          <i className="quality-dot hist-legend-impostor" />
          {tr("wrong person", "异人")}
        </span>
        <span>
          <i className="quality-dot hist-legend-genuine" />
          {tr("same person", "同人")}
        </span>
        <span>
          <i className="quality-dot hist-legend-candidate" />
          {tr("candidate", "候选阈值")}
        </span>
        <span className="spacer" />
        <span className="mono">
          {min.toFixed(2)} — {max.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function IssueQueue(props: {
  issues: LibraryIssue[];
  onAction: (issue: LibraryIssue) => void;
  busy: boolean;
}) {
  const { issues } = props;
  const [showInfo, setShowInfo] = useState(false);
  const visible = showInfo ? issues : issues.filter((item) => item.severity !== "info");
  const infoCount = issues.filter((item) => item.severity === "info").length;

  const actionLabel = (action: string) => {
    if (action === "embed") return tr("Backfill embeddings", "补齐嵌入");
    if (action === "review-samples") return tr("Review samples", "查看样本");
    if (action === "capture") return tr("Capture more", "补采样本");
    if (action === "set-threshold") return tr("Tune threshold", "调整阈值");
    return tr("Open", "打开");
  };

  return (
    <section className="quality-card">
      <div className="quality-card-head">
        <h2>{tr("What to fix", "待办队列")}</h2>
        <span className="spacer" />
        {infoCount > 0 && (
          <button
            className={`btn ghost ${showInfo ? "on" : ""}`}
            onClick={() => setShowInfo((value) => !value)}
          >
            {tr(`Show ${infoCount} suggestion(s)`, `显示 ${infoCount} 条提示`)}
          </button>
        )}
      </div>
      {visible.length === 0 && (
        <div className="placeholder">
          {tr("Nothing blocking — the library is healthy.", "没有阻塞问题 —— 声纹库健康。")}
        </div>
      )}
      <div className="quality-issues">
        {visible.map((issue, index) => {
          const text = issueText(issue);
          return (
          <div
            key={`${issue.kind}-${issue.person_public_id ?? "library"}-${index}`}
            className={`quality-issue sev-${issue.severity}`}
          >
            <div className="quality-issue-main">
              <div className="quality-issue-title">
                <span className={`badge sev-${issue.severity}`}>
                  {severityLabel(issue.severity)}
                </span>
                {text.title}
              </div>
              <div className="quality-issue-detail subtle">{text.detail}</div>
            </div>
            <button
              className="btn"
              disabled={props.busy && issue.action === "embed"}
              onClick={() => props.onAction(issue)}
            >
              {actionLabel(issue.action)}
            </button>
          </div>
          );
        })}
      </div>
    </section>
  );
}

function PeopleMatrix(props: { people: PersonHealth[] }) {
  return (
    <section className="quality-card">
      <div className="quality-card-head">
        <h2>{tr("People", "人员矩阵")}</h2>
      </div>
      <table className="quality-table">
        <thead>
          <tr>
            <th>{tr("Name", "姓名")}</th>
            <th>{tr("Availability", "可用性")}</th>
            <th>{tr("Matching", "匹配样本")}</th>
            <th>{tr("Audio", "有效时长")}</th>
            <th>{tr("Sources", "来源")}</th>
          </tr>
        </thead>
        <tbody>
          {props.people.map((person) => (
            <tr key={person.public_id}>
              <td>
                <Link
                  className="vp-project-link"
                  to={`/voiceprints?person=${encodeURIComponent(person.public_id)}`}
                >
                  {person.name}
                </Link>
              </td>
              <td>
                <span className={`badge av-${person.availability}`}>
                  {availabilityLabel(person.availability)}
                </span>
              </td>
              <td className="mono">
                {person.matching_sample_count}/{person.enabled_sample_count}
                {person.missing_embedding_count > 0 && (
                  <span className="warn">
                    {" "}
                    (−{person.missing_embedding_count})
                  </span>
                )}
              </td>
              <td className="mono">{fmtSeconds(person.matching_seconds)}</td>
              <td className="mono">{person.project_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
