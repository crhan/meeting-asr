import { useEffect, useMemo, useRef, useState } from "react";
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
import { fmtSeconds } from "../lib/format";
import { useJobStream } from "../lib/useJobStream";
import { PanelHead } from "../components/PanelHead";
import { SampleSourcePanel } from "../components/SampleSourcePanel";

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
    case "no-samples":
      return {
        title: tr(issue.title, `${name} 一条声纹样本都没有`),
        detail: tr(
          issue.detail,
          "这个人在库里存在,但从来没有采集过任何样本,因此永远不会被匹配到。去他说过话的项目里采几条,或者把这个条目删掉。",
        ),
      };
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
/** Share of scores satisfying the predicate; 0 for an empty population. */
function rateOf(scores: number[], predicate: (score: number) => boolean): number {
  return scores.length ? scores.filter(predicate).length / scores.length : 0;
}

/**
 * Price a candidate threshold locally, so dragging the cursor stays instant.
 *
 * Must stay identical to `VoiceprintCalibrationReport.cost_at` in
 * `voiceprint_calibration.py`, including the scaling: above the export cap the
 * score arrays are an evenly downsampled view of the library, so counting them
 * directly would report the cap rather than the real number of wrong matches.
 * The rate comes from the sample, the count from that rate applied to the true
 * population in `genuine.count` / `impostor.count`.
 */
function costAt(calibration: Calibration, threshold: number): ThresholdCost {
  const rejectRate = rateOf(calibration.genuine_scores, (s) => s < threshold);
  const acceptRate = rateOf(calibration.impostor_scores, (s) => s >= threshold);
  return {
    threshold,
    false_reject_count: Math.round(rejectRate * (calibration.genuine?.count ?? 0)),
    false_reject_rate: rejectRate,
    false_accept_count: Math.round(acceptRate * (calibration.impostor?.count ?? 0)),
    false_accept_rate: acceptRate,
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
      // Land on the rows the issue is about. Sending the operator to the
      // person's full sample list makes them re-derive which N of the rows
      // were meant, from a page that gives no way to tell.
      const filter = issue.kind === "overlapped-samples" ? "&filter=overlap" : "";
      navigate(
        `/voiceprints?person=${encodeURIComponent(issue.person_public_id)}${filter}`,
      );
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
              couplingBounds={thresholdQuery.data?.coupling_bounds ?? {}}
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
            <SampleSourcePanel
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
  couplingBounds: Record<string, number>;
  pending: boolean;
  onApply: (value: number | null) => void;
}) {
  const { calibration, configured, builtinDefault, warnings, warningKinds, pending } =
    props;
  const [candidate, setCandidate] = useState(calibration.current_threshold);

  useEffect(() => {
    setCandidate(calibration.current_threshold);
  }, [calibration.current_threshold]);

  // Same comparison the server makes, against bounds the server published.
  const candidateWarningKinds = useMemo(
    () =>
      Object.entries(props.couplingBounds)
        .filter(([, bound]) => candidate <= bound)
        .map(([kind]) => kind),
    [props.couplingBounds, candidate],
  );
  // Reuse the server's English prose when it already described this kind for
  // the active threshold; otherwise the localized branch carries the meaning.
  const activeWarningFor = (kind: string) =>
    warnings[warningKinds.indexOf(kind)] ?? "";

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
      {/* These counts price the threshold rule alone. Matching also rescues a
          clearly-leading score below the threshold, and pricing that would need
          each probe's runner-up, which this all-pairs sweep does not have --
          so the numbers bound the outcome rather than predict it. */}
      <p className="vq-note subtle">
        <span className="vq-note-tag">{tr("Scope", "口径")}</span>
        {tr(
          "Costs model the threshold rule only (accept when score ≥ threshold). Matching additionally rescues a score that clearly leads the runner-up, so real misses are no worse than shown and real wrong names no better.",
          "这里只按阈值规则计价(分数 ≥ 阈值才接受)。实际匹配还会救回「明显领先第二名」的低分,所以真实误拒不会比这更多、真实误纳不会比这更少。",
        )}
      </p>
      {calibration.low_confidence && (
        <p className="vq-note warn">
          <span className="vq-note-tag">{tr("Caveat", "注意")}</span>
          {tr(
            "Small library — the suggestion points a direction, not a precise operating point. Adding samples will move it.",
            "库样本量偏小 —— 建议值只指方向,不是精确工作点。补样本后会变化。",
          )}
        </p>
      )}
      {/* Judged on the CANDIDATE, not the active threshold. Warnings that
          only appear after the PUT are a receipt; the whole point of a
          coupling warning is to reach the user while the value is still a
          proposal. Bounds come from the server so the verdict cannot differ
          from the one it would reach itself. */}
      {candidateWarningKinds.map((kind) => (
        <p className="vq-note warn" key={kind}>
          <span className="vq-note-tag">{tr("Coupling", "耦合")}</span>
          {couplingWarning(kind, activeWarningFor(kind))}
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
