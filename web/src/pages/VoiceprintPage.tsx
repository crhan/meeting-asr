import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createPerson,
  deletePerson,
  deleteSample,
  excludeQualitySamples,
  getLibrary,
  getPersonSamples,
  getQuality,
  mergePeople,
  renamePerson,
  sampleClipUrl,
  setSampleStatus,
  type QualityPerson,
  type QualitySample,
  type VoiceprintPerson,
  type VoiceprintSample,
} from "../api/client";
import { tr } from "../lib/i18n";
import { confirmDialog } from "../lib/confirm";
import { promptDialog } from "../lib/prompt";
import { useClipAudio } from "../lib/useClipAudio";
import { Modal } from "../components/Modal";

type SortMode = "quality" | "name" | "samples";
type SampleFilter = "all" | "issues" | "matching" | "excluded" | "confirmed" | "unembedded";
type PersonView = VoiceprintPerson & { quality?: QualityPerson };
type SampleView = VoiceprintSample & { quality?: QualitySample };

function fmtMs(ms: number): string {
  const t = Math.round(ms / 1000);
  return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, "0")}`;
}

function statusLabel(status: string): string {
  if (status === "active") return tr("Unconfirmed · matching", "未确认身份 · 参与匹配");
  if (status === "verified-active") return tr("Confirmed · matching", "身份已确认 · 参与匹配");
  if (status === "quarantined") return tr("Unconfirmed · excluded", "未确认身份 · 不参与匹配");
  if (status === "verified-quarantined") return tr("Confirmed · excluded", "身份已确认 · 不参与匹配");
  if (status === "rejected") return tr("Rejected", "已废弃");
  return status;
}

function identityLabel(sample: SampleView): string {
  return sample.identity_confirmed ? tr("Identity confirmed", "身份已确认") : tr("Identity unconfirmed", "身份未确认");
}

function matchingLabel(sample: SampleView): string {
  return sample.matching_enabled ? tr("Used for matching", "参与匹配") : tr("Excluded from matching", "不参与匹配");
}

function statusForAxes(identityConfirmed: boolean, matchingEnabled: boolean): string {
  if (matchingEnabled) return identityConfirmed ? "verified-active" : "active";
  return identityConfirmed ? "verified-quarantined" : "quarantined";
}

function qualityLabel(label: string): string {
  if (label === "critical") return tr("critical", "严重");
  if (label === "warning") return tr("issue", "疑点");
  if (label === "ok") return tr("ok", "正常");
  if (label === "verified") return tr("trusted", "可信");
  if (label === "verified-disabled") return tr("confirmed excluded", "确认但排除");
  if (label === "unknown") return tr("unknown", "待评估");
  if (label === "quarantined" || label === "verified-quarantined" || label === "rejected")
    return statusLabel(label);
  return label;
}

function qualityClass(sample: SampleView): string {
  const label = sample.quality?.label;
  if (label === "critical") return "low";
  if (label === "warning" || !sample.quality) return "mid";
  return "ok";
}

function compareByName(a: { name: string; public_id: string }, b: { name: string; public_id: string }): number {
  return (
    a.name.localeCompare(b.name, "zh-Hans-CN-u-co-pinyin", { sensitivity: "base" }) ||
    a.public_id.localeCompare(b.public_id)
  );
}

function issueRank(person: PersonView): number {
  return (person.quality?.critical_count ?? 0) * 1000 + (person.quality?.suspicious_count ?? 0);
}

function sampleIssueRank(sample: SampleView): number {
  if (sample.quality?.label === "critical") return 0;
  if (sample.quality?.label === "warning") return 1;
  if (!sample.quality) return 2;
  if (!sample.matching_enabled) return 4;
  if (sample.identity_confirmed) return 5;
  return 3;
}

function matchesSampleFilter(sample: SampleView, filter: SampleFilter): boolean {
  if (filter === "issues") return sample.quality?.label === "critical" || sample.quality?.label === "warning";
  if (filter === "matching") return sample.matching_enabled;
  if (filter === "excluded") return !sample.matching_enabled;
  if (filter === "confirmed") return sample.identity_confirmed;
  if (filter === "unembedded") return !sample.quality;
  return true;
}

export function VoiceprintPage() {
  const queryClient = useQueryClient();
  const libraryQuery = useQuery({ queryKey: ["vp-library"], queryFn: getLibrary });
  const qualityQuery = useQuery({ queryKey: ["vp-quality"], queryFn: getQuality });
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("quality");
  const [sampleFilter, setSampleFilter] = useState<SampleFilter>("all");
  const [editMode, setEditMode] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const audio = useClipAudio();

  const invalidatePerson = (personRef?: string) => {
    queryClient.invalidateQueries({ queryKey: ["vp-library"] });
    queryClient.invalidateQueries({ queryKey: ["vp-quality"] });
    if (personRef) queryClient.invalidateQueries({ queryKey: ["vp-person", personRef] });
  };

  const createMut = useMutation({
    mutationFn: (name: string) => createPerson(name),
    onSuccess: (person) => {
      invalidatePerson(person.public_id);
      setSelected(person.public_id);
    },
  });

  const renameMut = useMutation({
    mutationFn: ({ personRef, name }: { personRef: string; name: string }) => renamePerson(personRef, name),
    onSuccess: (person) => {
      invalidatePerson(person.public_id);
      setSelected(person.public_id);
    },
  });

  const deletePersonMut = useMutation({
    mutationFn: (personRef: string) => deletePerson(personRef),
    onSuccess: () => {
      setSelected(null);
      invalidatePerson();
    },
  });

  const statusMut = useMutation({
    mutationFn: ({ samplePublicId, status }: { personRef: string; samplePublicId: string; status: string }) =>
      setSampleStatus(samplePublicId, status),
    onSuccess: (_row, variables) => invalidatePerson(variables.personRef),
  });

  const deleteSampleMut = useMutation({
    mutationFn: ({
      personRef,
      samplePublicId,
    }: {
      personRef: string;
      samplePublicId: string;
      lastSample: boolean;
    }) => deleteSample(personRef, samplePublicId),
    onSuccess: (_row, variables) => {
      if (variables.lastSample) {
        setSelected(null);
        setToast(
          tr(
            "Last sample deleted; the person was removed from the library.",
            "已删除最后一条样本，人物已从声纹库移除。",
          ),
        );
      }
      invalidatePerson(variables.personRef);
    },
  });

  const excludeQualityMut = useMutation({
    mutationFn: ({ personRef, samplePublicIds }: { personRef: string; samplePublicIds?: string[] }) =>
      excludeQualitySamples(personRef, samplePublicIds),
    onSuccess: (result, variables) => {
      invalidatePerson(variables.personRef);
      setToast(
        tr(
          `Excluded ${result.updated_count} low-quality sample(s) from matching.`,
          `已将 ${result.updated_count} 条低质样本排除出匹配。`,
        ),
      );
    },
  });

  const mergeMut = useMutation({
    mutationFn: ({ fromRef, intoRef }: { fromRef: string; intoRef: string }) =>
      mergePeople(fromRef, intoRef),
    onSuccess: (person, variables) => {
      // The source person no longer exists; invalidating its samples query would
      // refetch into a 404 global toast -- remove the cache entry instead.
      queryClient.removeQueries({ queryKey: ["vp-person", variables.fromRef] });
      queryClient.invalidateQueries({ queryKey: ["vp-library"] });
      queryClient.invalidateQueries({ queryKey: ["vp-quality"] });
      queryClient.invalidateQueries({ queryKey: ["vp-person", person.public_id] });
      setSelected(person.public_id);
      setToast(tr(`Merged into "${person.name}".`, `已合并到「${person.name}」。`));
    },
  });

  // Person being merged (the SOURCE, which the merge deletes).
  const [mergeSource, setMergeSource] = useState<PersonView | null>(null);

  const qualityByPerson = useMemo(() => {
    return new Map((qualityQuery.data?.people ?? []).map((person) => [person.public_id, person]));
  }, [qualityQuery.data]);

  const people = useMemo<PersonView[]>(() => {
    const merged = (libraryQuery.data?.people ?? []).map((person) => ({
      ...person,
      quality: qualityByPerson.get(person.public_id),
    }));
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? merged.filter(
          (person) =>
            person.name.toLowerCase().includes(needle) ||
            person.public_id.toLowerCase().includes(needle),
        )
      : merged;
    return [...filtered].sort((a, b) => {
      if (sortMode === "name") return compareByName(a, b);
      if (sortMode === "samples") return b.sample_count - a.sample_count || compareByName(a, b);
      return issueRank(b) - issueRank(a) || compareByName(a, b);
    });
  }, [libraryQuery.data, qualityByPerson, query, sortMode]);

  const selectedPerson = (selected && people.find((person) => person.public_id === selected)) || people[0];
  const isLoading = libraryQuery.isLoading || qualityQuery.isLoading;
  const error = libraryQuery.error || qualityQuery.error;

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  if (error) return <div className="error-box">{(error as Error).message}</div>;

  return (
    <div>
      <div className="vp-head">
        <div>
          <div className="row gap">
            <h1 style={{ marginRight: 8 }}>{tr("Voiceprints", "声纹库")}</h1>
            <span className="subtle mono">
              {qualityQuery.data?.model ?? ""} · {qualityQuery.data?.sample_count ?? 0}{" "}
              {tr("samples", "样本")}
            </span>
          </div>
          <div className="vp-summary subtle mono">
            <span className="warn">
              {qualityQuery.data?.suspicious_count ?? 0} {tr("issues", "疑点")}
            </span>
            <span style={{ color: "var(--red)" }}>
              {qualityQuery.data?.critical_count ?? 0} {tr("critical", "严重")}
            </span>
          </div>
        </div>
        <button
          className={`btn ${editMode ? "primary" : ""}`}
          onClick={() => setEditMode((value) => !value)}
        >
          {editMode ? tr("Done", "完成") : tr("Edit", "编辑")}
        </button>
      </div>

      <div className="vp-controls">
        <div className="vp-control-block vp-control-search">
          <input
            className="search vp-search"
            placeholder={tr("Search people…", "搜索人物…")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="vp-control-block">
          <div className="vp-control-label">{tr("Sort", "排序")}</div>
          <div className="vp-control-group">
            {(["quality", "name", "samples"] as const).map((mode) => (
              <button
                key={mode}
                className={`chip ${sortMode === mode ? "on" : ""}`}
                onClick={() => setSortMode(mode)}
              >
                {mode === "quality"
                  ? tr("Quality first", "质量问题")
                  : mode === "name"
                    ? tr("Name", "姓名")
                    : tr("Samples", "样本数")}
              </button>
            ))}
          </div>
        </div>
        <div className="vp-control-block">
          <div className="vp-control-label">{tr("Samples", "样本筛选")}</div>
          <div className="vp-control-group">
            {(["all", "issues", "matching", "excluded", "confirmed", "unembedded"] as const).map((filter) => (
              <button
                key={filter}
                className={`chip ${sampleFilter === filter ? "on" : ""}`}
                onClick={() => setSampleFilter(filter)}
              >
                {filter === "all"
                  ? tr("All", "全部")
                  : filter === "issues"
                    ? tr("Issues", "有问题")
                    : filter === "matching"
                      ? tr("Matching", "参与匹配")
                      : filter === "excluded"
                        ? tr("Excluded", "不参与")
                        : filter === "confirmed"
                          ? tr("Confirmed", "已确认")
                          : tr("Unembedded", "未嵌入")}
              </button>
            ))}
          </div>
        </div>
        {editMode && (
          <div className="vp-control-block vp-control-edit">
            <button
              className="btn"
              onClick={async () => {
                const name = await promptDialog({
                  title: tr("New person", "新建人物"),
                  message: tr("New person name:", "新人物姓名："),
                  confirmLabel: tr("Create", "创建"),
                });
                if (name?.trim()) createMut.mutate(name.trim());
              }}
            >
              + {tr("New person", "新建人物")}
            </button>
          </div>
        )}
      </div>

      <div className="vp-layout">
        <div className="speaker-list flat vp-side">
          {people.map((person) => (
            <button
              key={person.public_id}
              className={`speaker-card ${selectedPerson?.public_id === person.public_id ? "active" : ""}`}
              onClick={() => setSelected(person.public_id)}
            >
              <div className="speaker-card-top">
                <span className="speaker-name">{person.name}</span>
                {(person.quality?.critical_count ?? 0) > 0 && (
                  <span className="badge state-broken">{person.quality?.critical_count}</span>
                )}
                {(person.quality?.suspicious_count ?? 0) > 0 &&
                  (person.quality?.critical_count ?? 0) === 0 && (
                    <span
                      className="badge"
                      style={{ color: "var(--yellow)", borderColor: "var(--yellow)" }}
                    >
                      {person.quality?.suspicious_count}
                    </span>
                  )}
              </div>
              <div className="speaker-card-meta subtle">
                {(person.quality?.active_sample_count ?? person.sample_count)}/{person.sample_count}{" "}
                {tr("used", "参与")} · {tr("mean", "均值")}{" "}
                {person.quality?.mean_score?.toFixed(2) ?? "—"} · {person.project_count}{" "}
                {tr("projects", "项目")}
              </div>
            </button>
          ))}
        </div>

        <div className="vp-main">
          {selectedPerson ? (
            <PersonDetail
              person={selectedPerson}
              sampleFilter={sampleFilter}
              editMode={editMode}
              audio={audio}
              pendingSampleId={
                statusMut.isPending
                  ? (statusMut.variables?.samplePublicId ?? null)
                  : deleteSampleMut.isPending
                    ? (deleteSampleMut.variables?.samplePublicId ?? null)
                    : null
              }
              onRename={(name) => renameMut.mutate({ personRef: selectedPerson.public_id, name })}
              onDeletePerson={() => deletePersonMut.mutate(selectedPerson.public_id)}
              onMerge={() => setMergeSource(selectedPerson)}
              onSetStatus={(samplePublicId, status) =>
                statusMut.mutate({ personRef: selectedPerson.public_id, samplePublicId, status })
              }
              onDeleteSample={(samplePublicId, lastSample) =>
                deleteSampleMut.mutate({
                  personRef: selectedPerson.public_id,
                  samplePublicId,
                  lastSample,
                })
              }
              onExcludeIssues={(samplePublicIds) =>
                excludeQualityMut.mutate({
                  personRef: selectedPerson.public_id,
                  samplePublicIds,
                })
              }
            />
          ) : (
            <div className="placeholder">{tr("No people.", "暂无人物。")}</div>
          )}
        </div>
      </div>
      {mergeSource && (
        <MergePersonModal
          source={mergeSource}
          people={people.filter((person) => person.public_id !== mergeSource.public_id)}
          onPick={async (target) => {
            const ok = await confirmDialog({
              message: tr(
                `Merge all ${mergeSource.sample_count} sample(s) of "${mergeSource.name}" into "${target.name}" and delete "${mergeSource.name}"? This cannot be undone.`,
                `把「${mergeSource.name}」的全部 ${mergeSource.sample_count} 条样本并入「${target.name}」并删除「${mergeSource.name}」？此操作不可撤销。`,
              ),
              confirmLabel: tr("Merge", "合并"),
              danger: true,
            });
            setMergeSource(null);
            if (ok)
              mergeMut.mutate({ fromRef: mergeSource.public_id, intoRef: target.public_id });
          }}
          onClose={() => setMergeSource(null)}
        />
      )}
      {toast && (
        <div className="toast" onClick={() => setToast(null)}>
          {toast}
        </div>
      )}
    </div>
  );
}

/** Pick the person a duplicate entry should be merged into; likely targets
 *  (voiceprint-nearest neighbors) come first. */
function MergePersonModal(props: {
  source: PersonView;
  people: PersonView[];
  onPick: (target: PersonView) => void;
  onClose: () => void;
}) {
  const { source, people } = props;
  const neighborScores = new Map(
    (source.quality?.closest_people ?? []).map((n) => [n.public_id, n.score]),
  );
  const rows = [...people].sort((a, b) => {
    const na = neighborScores.get(a.public_id);
    const nb = neighborScores.get(b.public_id);
    if (na != null || nb != null) return (nb ?? -1) - (na ?? -1);
    return compareByName(a, b);
  });
  return (
    <Modal
      title={tr(`Merge "${source.name}" into…`, `把「${source.name}」合并到…`)}
      onClose={props.onClose}
    >
      <div className="people-list">
        {rows.length === 0 ? (
          <div className="subtle pad">{tr("No other people.", "没有其他人物。")}</div>
        ) : (
          rows.map((person) => (
            <button
              key={person.public_id}
              className="person-row"
              onClick={() => props.onPick(person)}
            >
              <span className="person-name">{person.name}</span>
              <span className="person-id mono">
                {person.sample_count} {tr("samples", "样本")}
                {neighborScores.has(person.public_id) &&
                  ` · ${tr("similarity", "相似")} ${neighborScores.get(person.public_id)!.toFixed(2)}`}
              </span>
            </button>
          ))
        )}
      </div>
    </Modal>
  );
}

function PersonDetail(props: {
  person: PersonView;
  sampleFilter: SampleFilter;
  editMode: boolean;
  audio: ReturnType<typeof useClipAudio>;
  /** Sample with a status/delete mutation in flight: its action chips disable. */
  pendingSampleId: string | null;
  onRename: (name: string) => void;
  onDeletePerson: () => void;
  onMerge: () => void;
  onSetStatus: (samplePublicId: string, status: string) => void;
  onDeleteSample: (samplePublicId: string, lastSample: boolean) => void;
  onExcludeIssues: (samplePublicIds?: string[]) => void;
}) {
  const { person, sampleFilter, editMode, audio } = props;
  const { data, isLoading, error } = useQuery({
    queryKey: ["vp-person", person.public_id],
    queryFn: () => getPersonSamples(person.public_id),
  });
  const qualityBySample = useMemo(() => {
    return new Map((person.quality?.samples ?? []).map((sample) => [sample.sample_public_id, sample]));
  }, [person.quality]);

  const allSamples = useMemo<SampleView[]>(() => {
    return (data?.samples ?? []).map((sample) => ({
      ...sample,
      quality: qualityBySample.get(sample.public_id),
    }));
  }, [data, qualityBySample]);

  const samples = useMemo<SampleView[]>(() => {
    return allSamples
      .filter((sample) => matchesSampleFilter(sample, sampleFilter))
      .sort(
        (a, b) =>
          sampleIssueRank(a) - sampleIssueRank(b) ||
          a.project_id.localeCompare(b.project_id) ||
          a.begin_time_ms - b.begin_time_ms,
      );
  }, [allSamples, sampleFilter]);
  const issueSamples = useMemo(
    () =>
      allSamples.filter(
        (sample) => sample.quality?.label === "critical" || sample.quality?.label === "warning",
      ),
    [allSamples],
  );
  const riskyProjects = person.quality?.projects.filter((project) => project.suspicious_count > 0) ?? [];

  if (error) return <div className="error-box">{(error as Error).message}</div>;
  if (isLoading || !data) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;

  return (
    <div>
      <div className="transcript-head">
        <div>
          <h2>{data.person.name}</h2>
          <div className="subtle">
            {person.quality?.active_sample_count ?? person.sample_count}/{person.sample_count}{" "}
            {tr("used for matching", "参与匹配")} · {tr("mean", "均值")}{" "}
            {person.quality?.mean_score?.toFixed(2) ?? "—"}
          </div>
        </div>
        {editMode && (
          <div className="row gap">
            <button
              className="btn"
              onClick={async () => {
                const name = await promptDialog({
                  title: tr("Rename", "改名"),
                  message: tr("Rename to:", "改名为："),
                  defaultValue: data.person.name,
                  confirmLabel: tr("Rename", "改名"),
                });
                if (name?.trim() && name.trim() !== data.person.name) props.onRename(name.trim());
              }}
            >
              {tr("Rename", "改名")}
            </button>
            <button
              className="btn ghost"
              title={tr(
                "Merge this (duplicate) person's samples into another person.",
                "把这个（重复的）人物的样本并入另一个人物。",
              )}
              onClick={props.onMerge}
            >
              {tr("Merge into…", "合并到…")}
            </button>
            <button
              className="btn ghost"
              onClick={async () => {
                if (
                  await confirmDialog({
                    message: tr(
                      `Delete "${data.person.name}" and all samples?`,
                      `删除「${data.person.name}」及全部样本？`,
                    ),
                    confirmLabel: tr("Delete person", "删除人物"),
                    danger: true,
                  })
                )
                  props.onDeletePerson();
              }}
            >
              {tr("Delete person", "删除人物")}
            </button>
          </div>
        )}
      </div>
      {samples.length === 0 ? (
        <div className="placeholder">{tr("No samples match the filter.", "没有符合筛选的样本。")}</div>
      ) : (
        <>
          <VoiceprintHealthPanel
            person={person}
            issueSamples={issueSamples}
            riskyProjects={riskyProjects}
            editMode={editMode}
            onExcludeIssues={() => props.onExcludeIssues(issueSamples.map((sample) => sample.public_id))}
          />
          <div className="segments">
            {samples.map((sample) => (
              <SampleRow
                key={sample.public_id}
                personRef={person.public_id}
                sample={sample}
                sampleCount={data.samples.length}
                audio={audio}
                editMode={editMode}
                pending={props.pendingSampleId === sample.public_id}
                onSetStatus={props.onSetStatus}
                onDelete={props.onDeleteSample}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function VoiceprintHealthPanel(props: {
  person: PersonView;
  issueSamples: SampleView[];
  riskyProjects: NonNullable<QualityPerson["projects"]>;
  editMode: boolean;
  onExcludeIssues: () => void;
}) {
  const { person, issueSamples, riskyProjects, editMode } = props;
  const q = person.quality;
  if (!q) return null;
  const hasRisk = q.suspicious_count > 0 || q.critical_count > 0;
  return (
    <div className={`vp-health ${hasRisk ? "risk" : ""}`}>
      <div className="vp-health-main">
        <div className="vp-health-metrics">
          <span>
            {tr("matching", "参与匹配")} <strong>{q.active_sample_count}</strong>/{q.sample_count}
          </span>
          <span>
            {tr("mean", "均值")} <strong>{q.mean_score?.toFixed(2) ?? "—"}</strong>
          </span>
          <span className={q.critical_count > 0 ? "danger-text" : q.suspicious_count > 0 ? "warn" : ""}>
            {q.suspicious_count} {tr("issues", "疑点")} · {q.critical_count} {tr("critical", "严重")}
          </span>
        </div>
        {q.closest_people.length > 0 && (
          <div className="vp-health-line subtle">
            {tr("Closest others", "相近人物")}：
            {q.closest_people.map((neighbor) => (
              <span key={neighbor.public_id} className="mono">
                {neighbor.name} {neighbor.score.toFixed(2)}
              </span>
            ))}
          </div>
        )}
        {riskyProjects.length > 0 && (
          <div className="vp-health-projects">
            {riskyProjects.slice(0, 4).map((project) => (
              <Link
                key={project.project_id}
                className="badge vp-project-risk"
                to={`/projects/${encodeURIComponent(project.project_id)}/speakers`}
                title={tr("Open this project's speaker review", "打开该项目的 speaker review")}
              >
                {project.project_id} · {project.suspicious_count} {tr("issues", "疑点")} ·{" "}
                {project.min_score?.toFixed(2) ?? "—"}
              </Link>
            ))}
          </div>
        )}
      </div>
      {editMode && issueSamples.length > 0 && (
        <button className="btn ghost danger" onClick={props.onExcludeIssues}>
          {tr("Exclude issue samples", "排除疑点样本")} ({issueSamples.length})
        </button>
      )}
    </div>
  );
}

function SampleRow(props: {
  personRef: string;
  sample: SampleView;
  sampleCount: number;
  audio: ReturnType<typeof useClipAudio>;
  editMode: boolean;
  /** A status/delete mutation for THIS sample is in flight: disable its actions.
   *  Both chips flip axes of the same status field, so a double-click while the
   *  first PATCH is in flight silently reverts the user's intent. */
  pending: boolean;
  onSetStatus: (samplePublicId: string, status: string) => void;
  onDelete: (samplePublicId: string, lastSample: boolean) => void;
}) {
  const { personRef, sample, sampleCount, audio, editMode, pending } = props;
  const key = `${personRef}:${sample.public_id}`;
  const playing = audio.playingKey === key;
  const scoreText = sample.quality?.score == null ? "—" : sample.quality.score.toFixed(2);
  const qualityText = sample.quality ? qualityLabel(sample.quality.label) : tr("unembedded", "未嵌入");

  return (
    <div key={sample.public_id} className={`segment ${playing ? "playing" : ""}`}>
      <button className="play-btn" onClick={() => audio.toggle(key, sampleClipUrl(personRef, sample.public_id))}>
        {playing ? "⏸" : "▶"}
      </button>
      <div className="segment-body">
        <div className="segment-meta subtle mono">
          {fmtMs(sample.begin_time_ms)} ·{" "}
          <Link
            className="vp-project-link"
            to={`/projects/${encodeURIComponent(sample.project_id)}/speakers`}
            title={tr("Open this sample's source project", "打开该样本来源项目")}
          >
            {sample.project_id}
          </Link>{" "}
          ·{" "}
          <span className={`score-badge ${qualityClass(sample)}`}>
            {scoreText} {qualityText}
          </span>{" "}
          <span className={`badge status-pill ${sample.status}`}>{identityLabel(sample)}</span>
          <span className={`badge status-pill ${sample.matching_enabled ? "active" : "quarantined"}`}>
            {matchingLabel(sample)}
          </span>
        </div>
        <div className="segment-text">{sample.transcript_text}</div>
        {sample.quality?.reason && (
          <div className="subtle" style={{ fontSize: 11.5, marginTop: 3 }}>
            {sample.quality.reason}
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
            <div className="seg-progress-bar" style={{ width: `${audio.progress * 100}%` }} />
          </div>
        )}
        {editMode && (
          <div className="vp-sample-actions">
            <button
              className={`chip ${sample.identity_confirmed ? "on" : ""}`}
              disabled={pending}
              onClick={() =>
                props.onSetStatus(
                  sample.public_id,
                  statusForAxes(!sample.identity_confirmed, sample.matching_enabled),
                )
              }
            >
              {sample.identity_confirmed ? tr("Unconfirm identity", "取消身份确认") : tr("Confirm identity", "确认身份")}
            </button>
            <button
              className={`chip ${sample.matching_enabled ? "on" : ""}`}
              disabled={pending}
              onClick={() =>
                props.onSetStatus(
                  sample.public_id,
                  statusForAxes(sample.identity_confirmed, !sample.matching_enabled),
                )
              }
            >
              {sample.matching_enabled ? tr("Exclude from matching", "排除匹配") : tr("Use for matching", "参与匹配")}
            </button>
            {sample.status !== "rejected" && (
              <button
                className="chip danger"
                disabled={pending}
                onClick={() => props.onSetStatus(sample.public_id, "rejected")}
              >
                {tr("Reject", "废弃")}
              </button>
            )}
          </div>
        )}
      </div>
      {editMode && (
        <button
          className="reassign-btn"
          title={tr("Delete sample", "删除样本")}
          disabled={pending}
          onClick={async () => {
            const lastSample = sampleCount <= 1;
            if (
              await confirmDialog({
                // Deleting the last sample hard-deletes the whole person row -- say so.
                message: lastSample
                  ? tr(
                      `This is the last sample of "${sample.speaker_name}"; deleting it also removes the person from the library.`,
                      `这是「${sample.speaker_name}」的最后一条样本，删除后该人物将一并从声纹库移除。`,
                    )
                  : tr("Delete this sample?", "删除这条样本？"),
                confirmLabel: lastSample ? tr("Delete person", "删除人物") : tr("Delete", "删除"),
                danger: true,
              })
            )
              props.onDelete(sample.public_id, lastSample);
          }}
        >
          🗑
        </button>
      )}
    </div>
  );
}
