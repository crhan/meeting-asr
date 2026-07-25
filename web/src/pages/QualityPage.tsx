import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  captureAccept,
  captureRollback,
  captureRollbackUrl,
  captureRun,
  clipUrl,
  getEmbedBacklog,
  getLibraryHealth,
  getMatchThreshold,
  getSampleSources,
  runEmbedBackfill,
  setMatchThreshold,
  type Calibration,
  type CaptureResult,
  type LibraryHealth,
  type LibraryIssue,
  type PersonHealth,
  type SampleSource,
  type ThresholdCost,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { useClipAudio } from "../lib/useClipAudio";
import { useJobStream } from "../lib/useJobStream";
import { CaptureResultModal } from "../components/CaptureResultModal";
import { JobProgress } from "../components/JobProgress";

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
          `有 ${num(c, "embeddable_count")} 个样本音频可读但没有当前模型的向量,因此这个人完全不在匹配池里。通常是切换 provider 后没有重新嵌入。`,
        ),
      };
    case "missing-clips":
      return {
        title: tr(
          issue.title,
          `${name} 有 ${num(c, "missing_clip_count")} 个样本的音频已丢失`,
        ),
        detail: tr(
          issue.detail,
          "库里还有这些样本的记录,但在当前 store 下读不到它们的 clip 文件,因此永远无法嵌入。只能删除这些样本,或者从某个项目重新采集这个人。补齐嵌入解决不了。",
        ),
      };
    case "partial-embeddings":
      return {
        title: tr(issue.title, `${name} 有部分样本未嵌入`),
        detail: tr(
          issue.detail,
          `${num(c, "enabled_sample_count")} 个可用样本中有 ${num(c, "embeddable_count")} 个没有当前模型的向量,不参与匹配。`,
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
    case "overlapped-samples":
      return {
        title: tr(
          issue.title,
          `${name} 有 ${num(c, "overlapped_count")} 条样本是在别人说话时录的`,
        ),
        detail: tr(
          issue.detail,
          `这些样本前后半秒内就有另一个人在说话,参考音频是混合体而不是单一声音。它会把这个人的质心往「跟他对话的那个人」拉——而那恰恰是最难区分的一对。换成他独自连续说话的片段。`,
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

function fmtClock(ms: number): string {
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}:${(total % 60).toString().padStart(2, "0")}`;
}

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
  return null;
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
  // "Capture more" used to navigate to the project list, which is where the
  // question starts rather than where it is answered. Hold the person here and
  // resolve it in place: which meetings hold their speech, and how much.
  const [sourcingFor, setSourcingFor] = useState<{ id: string; name: string } | null>(
    null,
  );
  // Submitting only queues the job; the health numbers cannot change until it
  // finishes. Track it to completion, then refresh and report what it did --
  // otherwise the queue sits there unchanged and reads as a broken button.
  const [embedJobId, setEmbedJobId] = useState<string | null>(null);
  const embedJob = useJobStream(embedJobId);
  const reportedJobRef = useRef<string | null>(null);

  const embedMut = useMutation({
    mutationFn: runEmbedBackfill,
    onSuccess: (job) => {
      setEmbedJobId(job.job_id);
      setToast(tr("Embedding started…", "正在补齐嵌入…"));
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  useEffect(() => {
    if (!embedJobId || !embedJob.done) return;
    if (reportedJobRef.current === embedJobId) return;
    reportedJobRef.current = embedJobId;
    queryClient.invalidateQueries({ queryKey: ["vp-health"] });
    queryClient.invalidateQueries({ queryKey: ["vp-embed-backlog"] });
    if (embedJob.error) {
      setToast(tr(`Embedding failed: ${embedJob.error}`, `补齐嵌入失败:${embedJob.error}`));
      return;
    }
    const result = embedJob.result as
      | { embedded_count?: number; skipped_count?: number }
      | null;
    const embedded = result?.embedded_count ?? 0;
    if (embedded === 0) {
      setToast(
        tr(
          "Nothing was embedded — the remaining samples have no readable clip audio.",
          "没有新增任何向量 —— 剩下的样本读不到 clip 音频,补嵌入无法解决。",
        ),
      );
      return;
    }
    setToast(
      tr(
        `Embedded ${embedded} sample(s).`,
        `已补齐 ${embedded} 个样本的向量。`,
      ),
    );
  }, [embedJobId, embedJob.done, embedJob.error, embedJob.result, queryClient]);

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
      if (!issue.person_public_id) {
        navigate("/projects");
        return;
      }
      setSourcingFor({
        id: issue.person_public_id,
        name: issue.person_name ?? issue.person_public_id,
      });
      window.requestAnimationFrame(() =>
        document
          .getElementById("source-plan")
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
      return;
    }
    if (issue.action === "set-threshold") {
      document
        .getElementById("threshold-card")
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <div className="pad vq-page">
      <header className="vq-masthead">
        <div>
          <span className="vq-eyebrow">{tr("Voiceprint library", "声纹库")}</span>
          <h1>{tr("Library quality", "声纹库质量")}</h1>
        </div>
        <span className="spacer" />
        <Link className="btn ghost" to="/voiceprints">
          {tr("Manage library", "管理声纹库")}
        </Link>
      </header>

      {toast && <div className="vq-toast">{toast}</div>}
      {isLoading && <div className="vq-empty">{tr("Loading…", "加载中…")}</div>}
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
            busy={embedMut.isPending || (embedJobId !== null && !embedJob.done)}
            activePersonId={sourcingFor?.id ?? null}
          />
          {sourcingFor && (
            <SourcePlan
              personId={sourcingFor.id}
              personName={sourcingFor.name}
              onClose={() => setSourcingFor(null)}
            />
          )}
          <PeopleMatrix
            people={health.people}
            onFindSources={(person) =>
              setSourcingFor({ id: person.public_id, name: person.name })
            }
          />
        </>
      )}
    </div>
  );
}

/** Section eyebrow + title, the page's recurring instrument-panel header. */
function PanelHead(props: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="vq-panel-head">
      <div className="vq-panel-titles">
        <span className="vq-eyebrow">{props.eyebrow}</span>
        <h2>{props.title}</h2>
      </div>
      {props.children}
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
  const blocked = health.person_count - health.usable_person_count;
  return (
    <section className="vq-panel">
      <PanelHead eyebrow={tr("Diagnostics", "体检")} title={tr("Library vitals", "库体检")}>
        <span className="vq-model mono">{health.model}</span>
      </PanelHead>

      <div className="vq-readouts">
        <Readout
          value={`${health.usable_person_count}`}
          suffix={`/${health.person_count}`}
          label={tr("matchable", "人可参与匹配")}
          tone={blocked > 0 ? "bad" : "good"}
        />
        <Readout
          value={String(health.matching_sample_count)}
          label={tr("matching samples", "匹配样本")}
        />
        <Readout
          value={fmtSeconds(health.matching_seconds)}
          label={tr("matching audio", "匹配音频")}
        />
        <Readout
          value={String(health.critical_count)}
          label={tr("critical", "严重问题")}
          tone={health.critical_count > 0 ? "bad" : "good"}
        />
      </div>

      <div className="vq-healthbar">
        {buckets.map(
          (bucket) =>
            bucket.count > 0 && (
              <span
                key={bucket.key}
                className={`vq-healthbar-seg av-${bucket.key}`}
                style={{ width: `${(bucket.count / total) * 100}%` }}
                title={`${availabilityLabel(bucket.key)}: ${bucket.count}`}
              >
                <b>{bucket.count}</b>
              </span>
            ),
        )}
      </div>
      <div className="vq-legend">
        {buckets.map((bucket) => (
          <span key={bucket.key} className={bucket.count === 0 ? "vq-dim" : ""}>
            <i className={`vq-dot av-${bucket.key}`} />
            {availabilityLabel(bucket.key)}
            <b className="mono">{bucket.count}</b>
          </span>
        ))}
        {backlog && backlog.missing_sample_count > 0 && (
          <span className="warn">
            {tr(
              `${backlog.missing_sample_count} sample(s) awaiting embedding`,
              `${backlog.missing_sample_count} 个样本待嵌入`,
            )}
          </span>
        )}
      </div>
    </section>
  );
}

function Readout(props: {
  value: string;
  suffix?: string;
  label: string;
  tone?: "good" | "bad";
}) {
  return (
    <div className="vq-readout">
      <div className={`vq-readout-value ${props.tone ?? ""}`}>
        {props.value}
        {props.suffix && <span className="vq-readout-suffix">{props.suffix}</span>}
      </div>
      <div className="vq-readout-label">{props.label}</div>
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
    <section className="vq-panel vq-panel-feature" id="threshold-card">
      <PanelHead
        eyebrow={tr("Calibration", "校准")}
        title={tr("Acceptance threshold", "接受阈值")}
      >
        <span className="vq-populations mono">
          <i className="vq-dot pop-genuine" />
          {calibration.genuine_scores.length} {tr("same", "同人")}
          <em>·</em>
          <i className="vq-dot pop-impostor" />
          {calibration.impostor_scores.length} {tr("other", "异人")}
        </span>
      </PanelHead>

      <ThresholdInstrument
        calibration={calibration}
        candidate={candidate}
        suggested={suggested}
        onChange={setCandidate}
      />

      <div className="vq-costs">
        <CostReadout
          title={tr("At this threshold", "此阈值下")}
          threshold={candidate}
          cost={cost}
          variant="candidate"
          delta={
            currentCost
              ? cost.false_reject_count - currentCost.false_reject_count
              : undefined
          }
        />
        {currentCost && (
          <CostReadout
            title={tr("Active now", "当前生效")}
            threshold={currentCost.threshold}
            cost={currentCost}
            variant="current"
          />
        )}
        {suggested !== null && (
          <CostReadout
            title={tr("Suggested", "建议值")}
            threshold={suggested}
            cost={costAt(calibration, suggested)}
            variant="suggested"
          />
        )}
      </div>

      {suggested !== null && (
        <p className="vq-note">
          <span className="vq-note-tag">{tr("Why", "依据")}</span>
          {suggestionReason(calibration)}
        </p>
      )}
      {calibration.low_confidence && (
        <p className="vq-note warn">
          <span className="vq-note-tag">{tr("Caveat", "注意")}</span>
          {tr(
            "Small library — the suggestion points a direction, not a precise operating point. Adding samples will move it.",
            "库样本量偏小 —— 建议值只指方向,不是精确工作点。补样本后会变化。",
          )}
        </p>
      )}
      {warnings.map((warning, index) => (
        <p className="vq-note warn" key={warning}>
          <span className="vq-note-tag">{tr("Coupling", "耦合")}</span>
          {couplingWarning(warningKinds[index] ?? "", warning)}
        </p>
      ))}

      <div className="vq-actions">
        <button
          className="btn primary"
          disabled={!dirty || pending}
          onClick={() => props.onApply(Number(candidate.toFixed(3)))}
        >
          {tr(`Apply ${candidate.toFixed(3)}`, `应用 ${candidate.toFixed(3)}`)}
        </button>
        {suggested !== null && (
          <button
            className="btn"
            disabled={pending}
            onClick={() => setCandidate(suggested)}
          >
            {tr("Jump to suggested", "跳到建议值")}
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
        <span className="vq-config-state mono">
          {configured === null
            ? tr(`unset · default ${builtinDefault}`, `未配置 · 默认 ${builtinDefault}`)
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
  variant: "candidate" | "current" | "suggested";
  delta?: number;
}) {
  const { cost, delta } = props;
  return (
    <div className={`vq-cost ${props.variant}`}>
      <div className="vq-cost-title">
        <i className={`vq-dot cur-${props.variant}`} />
        {props.title}
      </div>
      <div className="vq-cost-threshold">{props.threshold.toFixed(3)}</div>
      <div className="vq-cost-lines">
        <span className={cost.false_reject_count > 0 ? "is-bad" : "is-ok"}>
          <em>{tr("misses", "误拒")}</em>
          <b>{cost.false_reject_count}</b>
          <small>{(cost.false_reject_rate * 100).toFixed(0)}%</small>
          {delta !== undefined && delta !== 0 && (
            <u className={delta < 0 ? "better" : "worse"}>
              {delta < 0 ? "▼" : "▲"}
              {Math.abs(delta)}
            </u>
          )}
        </span>
        <span className={cost.false_accept_count > 0 ? "is-critical" : "is-ok"}>
          <em>{tr("wrong names", "误纳")}</em>
          <b>{cost.false_accept_count}</b>
          <small>{(cost.false_accept_rate * 100).toFixed(0)}%</small>
        </span>
      </div>
    </div>
  );
}

const PLOT_MIN = 0.3;
const PLOT_MAX = 0.95;
const PLOT_BINS = 26;
const AXIS_TICKS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

function toPercent(value: number): number {
  return ((value - PLOT_MIN) / (PLOT_MAX - PLOT_MIN)) * 100;
}

function fromPercent(percent: number): number {
  return PLOT_MIN + (percent / 100) * (PLOT_MAX - PLOT_MIN);
}

function histogram(scores: number[]): number[] {
  const counts = new Array(PLOT_BINS).fill(0);
  for (const score of scores) {
    const clamped = Math.min(Math.max(score, PLOT_MIN), PLOT_MAX - 1e-9);
    counts[Math.floor(((clamped - PLOT_MIN) / (PLOT_MAX - PLOT_MIN)) * PLOT_BINS)] += 1;
  }
  return counts;
}

/**
 * The page's centrepiece: both score populations on one axis, with the empty
 * band between them drawn explicitly.
 *
 * That band is the whole argument — a threshold inside it separates the two
 * populations with room to spare, one inside the genuine mass is silently
 * rejecting correct matches. Every summary statistic we could print instead
 * (EER, percentiles) asks the reader to reconstruct this picture in their head.
 *
 * Geometry is percentage-based end to end (bars, cursors, ticks, handle) and
 * the drag track is hand-rolled rather than an `<input type=range>`: a native
 * thumb is inset by half its width, so its centre never lines up with a
 * percentage-positioned cursor line, and a calibration instrument whose
 * pointer disagrees with its own plot is worse than no instrument.
 */
function ThresholdInstrument(props: {
  calibration: Calibration;
  candidate: number;
  suggested: number | null;
  onChange: (value: number) => void;
}) {
  const { calibration, candidate, suggested } = props;
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const genuine = useMemo(
    () => histogram(calibration.genuine_scores),
    [calibration.genuine_scores],
  );
  const impostor = useMemo(
    () => histogram(calibration.impostor_scores),
    [calibration.impostor_scores],
  );
  const peak = Math.max(1, ...genuine, ...impostor);

  // The safe band: above every impostor observed, below the weak tail of the
  // genuine ones. Only meaningful when the populations actually separate.
  const bandStart = calibration.impostor?.max ?? null;
  const bandEnd = calibration.genuine?.p5 ?? null;
  const hasBand =
    bandStart !== null && bandEnd !== null && bandEnd - bandStart > 0.005;

  const setFromClientX = (clientX: number) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    props.onChange(Number(fromPercent(ratio * 100).toFixed(3)));
  };

  const nudge = (step: number) => {
    props.onChange(
      Number(Math.min(Math.max(candidate + step, PLOT_MIN), PLOT_MAX).toFixed(3)),
    );
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const large = event.shiftKey ? 0.05 : 0.005;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      event.preventDefault();
      nudge(-large);
    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      event.preventDefault();
      nudge(large);
    } else if (event.key === "Home") {
      event.preventDefault();
      props.onChange(PLOT_MIN);
    } else if (event.key === "End") {
      event.preventDefault();
      props.onChange(PLOT_MAX);
    }
  };

  const cursors: { key: string; value: number; label: string }[] = [];
  cursors.push({
    key: "current",
    value: calibration.current_threshold,
    label: tr("active", "生效"),
  });
  if (suggested !== null)
    cursors.push({ key: "suggested", value: suggested, label: tr("suggested", "建议") });

  return (
    <div className="vq-instrument">
      <div className="vq-cursor-labels">
        {cursors.map((cursor) => (
          <span
            key={cursor.key}
            className={`vq-cursor-label cur-${cursor.key}`}
            style={{ left: `${toPercent(cursor.value)}%` }}
          >
            {cursor.label} {cursor.value.toFixed(3)}
          </span>
        ))}
      </div>

      <div className="vq-plot">
        {hasBand && (
          <div
            className="vq-band"
            style={{
              left: `${toPercent(bandStart)}%`,
              width: `${toPercent(bandEnd) - toPercent(bandStart)}%`,
            }}
          >
            <span className="vq-band-label">{tr("safe band", "安全区")}</span>
          </div>
        )}
        <div className="vq-gridlines" aria-hidden="true">
          {AXIS_TICKS.map((tick) => (
            <i key={tick} style={{ left: `${toPercent(tick)}%` }} />
          ))}
        </div>
        <div className="vq-bars">
          {genuine.map((_, index) => (
            <div className="vq-bin" key={index}>
              <span
                className="vq-bar impostor"
                style={{ height: `${(impostor[index] / peak) * 100}%` }}
              />
              <span
                className="vq-bar genuine"
                style={{ height: `${(genuine[index] / peak) * 100}%` }}
              />
            </div>
          ))}
        </div>
        {cursors.map((cursor) => (
          <i
            key={cursor.key}
            className={`vq-cursor cur-${cursor.key}`}
            style={{ left: `${toPercent(cursor.value)}%` }}
          />
        ))}
        <i
          className={`vq-cursor cur-candidate ${dragging ? "on" : ""}`}
          style={{ left: `${toPercent(candidate)}%` }}
        />
      </div>

      <div className="vq-axis" aria-hidden="true">
        {AXIS_TICKS.map((tick) => (
          <span key={tick} style={{ left: `${toPercent(tick)}%` }}>
            {tick.toFixed(2)}
          </span>
        ))}
      </div>

      <div
        className={`vq-track ${dragging ? "on" : ""}`}
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label={tr("Candidate threshold", "候选阈值")}
        aria-valuemin={PLOT_MIN}
        aria-valuemax={PLOT_MAX}
        aria-valuenow={candidate}
        aria-valuetext={candidate.toFixed(3)}
        onKeyDown={onKeyDown}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          setDragging(true);
          setFromClientX(event.clientX);
        }}
        onPointerMove={(event) => {
          if (!dragging) return;
          setFromClientX(event.clientX);
        }}
        onPointerUp={(event) => {
          event.currentTarget.releasePointerCapture(event.pointerId);
          setDragging(false);
        }}
        onPointerCancel={() => setDragging(false)}
      >
        <div className="vq-track-rail" />
        <div className="vq-track-fill" style={{ width: `${toPercent(candidate)}%` }} />
        <div className="vq-handle" style={{ left: `${toPercent(candidate)}%` }}>
          <span className="vq-handle-readout">{candidate.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

function IssueQueue(props: {
  issues: LibraryIssue[];
  onAction: (issue: LibraryIssue) => void;
  busy: boolean;
  /** The person whose sourcing plan is open below, marked so the two connect. */
  activePersonId: string | null;
}) {
  const { issues } = props;
  const [showInfo, setShowInfo] = useState(false);
  const visible = showInfo ? issues : issues.filter((item) => item.severity !== "info");
  const infoCount = issues.filter((item) => item.severity === "info").length;

  const actionLabel = (action: string) => {
    if (action === "embed") return tr("Backfill", "补齐嵌入");
    if (action === "review-samples") return tr("Review", "查看样本");
    if (action === "capture") return tr("Capture", "补采样本");
    if (action === "set-threshold") return tr("Tune", "调整阈值");
    return tr("Open", "打开");
  };

  return (
    <section className="vq-panel">
      <PanelHead eyebrow={tr("Queue", "队列")} title={tr("What to fix", "待办队列")}>
        <span className="spacer" />
        {infoCount > 0 && (
          <button
            className={`btn ghost ${showInfo ? "on" : ""}`}
            onClick={() => setShowInfo((value) => !value)}
          >
            {showInfo
              ? tr("Hide suggestions", "隐藏提示")
              : tr(`+${infoCount} suggestions`, `+${infoCount} 条提示`)}
          </button>
        )}
      </PanelHead>

      {visible.length === 0 && (
        <div className="vq-empty">
          <b>{tr("All clear", "一切正常")}</b>
          {tr("Nothing is blocking matching.", "没有阻塞匹配的问题。")}
        </div>
      )}

      <ol className="vq-issues">
        {visible.map((issue, index) => {
          const text = issueText(issue);
          const active =
            issue.action === "capture" &&
            !!issue.person_public_id &&
            issue.person_public_id === props.activePersonId;
          return (
            <li
              key={`${issue.kind}-${issue.person_public_id ?? "library"}-${index}`}
              className={`vq-issue sev-${issue.severity} ${active ? "on" : ""}`}
            >
              <span className="vq-issue-index mono">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className="vq-issue-body">
                <div className="vq-issue-title">
                  <span className={`vq-sev sev-${issue.severity}`}>
                    {severityLabel(issue.severity)}
                  </span>
                  {text.title}
                </div>
                <p className="vq-issue-detail">{text.detail}</p>
              </div>
              <button
                className="btn vq-issue-action"
                disabled={props.busy && issue.action === "embed"}
                onClick={() => props.onAction(issue)}
              >
                {active ? tr("Showing", "已展开") : actionLabel(issue.action)}
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
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
 */
function SourcePlan(props: { personId: string; personName: string; onClose: () => void }) {
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

function PeopleMatrix(props: {
  people: PersonHealth[];
  onFindSources: (person: PersonHealth) => void;
}) {
  const longest = Math.max(1, ...props.people.map((person) => person.matching_seconds));
  return (
    <section className="vq-panel">
      <PanelHead eyebrow={tr("Roster", "名册")} title={tr("People", "人员矩阵")} />
      <div className="vq-matrix">
        <div className="vq-matrix-head">
          <span>{tr("Name", "姓名")}</span>
          <span>{tr("Availability", "可用性")}</span>
          <span className="ta-r">{tr("Matching", "匹配样本")}</span>
          <span>{tr("Audio", "有效时长")}</span>
          <span className="ta-r">{tr("Sources", "来源")}</span>
          <span />
        </div>
        {props.people.map((person) => (
          <div className="vq-matrix-row" key={person.public_id}>
            <Link
              className="vq-matrix-name"
              to={`/voiceprints?person=${encodeURIComponent(person.public_id)}`}
            >
              {person.name}
            </Link>
            <span>
              <span className={`vq-sev av-${person.availability}`}>
                {availabilityLabel(person.availability)}
              </span>
            </span>
            <span className="mono ta-r" data-label={tr("matching", "匹配")}>
              {person.matching_sample_count}/{person.enabled_sample_count}
              {person.missing_embedding_count > 0 && (
                <b className="warn"> −{person.missing_embedding_count}</b>
              )}
            </span>
            <span className="vq-matrix-audio" data-label={tr("audio", "时长")}>
              <i
                style={{ width: `${(person.matching_seconds / longest) * 100}%` }}
                className={person.matching_seconds < 20 ? "thin" : ""}
              />
              <em className="mono">{fmtSeconds(person.matching_seconds)}</em>
            </span>
            <span className="mono ta-r" data-label={tr("sources", "来源")}>
              {person.project_count}
            </span>
            <span className="ta-r">
              <button
                className="btn ghost vq-matrix-action"
                onClick={() => props.onFindSources(person)}
              >
                {tr("Top up", "补采")}
              </button>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
