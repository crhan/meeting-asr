import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createPerson,
  deletePerson,
  deleteSample,
  getLibrary,
  getPersonSamples,
  getQuality,
  renamePerson,
  sampleClipUrl,
  setSampleStatus,
  type QualityPerson,
  type VoiceprintPerson,
} from "../api/client";
import { tr } from "../lib/i18n";
import { useClipAudio } from "../lib/useClipAudio";

type Tab = "library" | "quality";

function fmtMs(ms: number): string {
  const t = Math.round(ms / 1000);
  return `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, "0")}`;
}

export function VoiceprintPage() {
  const [tab, setTab] = useState<Tab>("library");
  return (
    <div>
      <div className="row gap" style={{ marginBottom: 14 }}>
        <h1 style={{ marginRight: 8 }}>{tr("Voiceprints", "声纹库")}</h1>
        {(["library", "quality"] as const).map((t) => (
          <button key={t} className={`chip ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>
            {t === "library" ? tr("Library", "库") : tr("Quality", "质量")}
          </button>
        ))}
      </div>
      {tab === "library" ? <LibraryTab /> : <QualityTab />}
    </div>
  );
}

// ---- Library ---------------------------------------------------------------

function LibraryTab() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["vp-library"], queryFn: getLibrary });
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const audio = useClipAudio();

  const createMut = useMutation({
    mutationFn: (name: string) => createPerson(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["vp-library"] }),
  });

  const filtered = useMemo(() => {
    const people = data?.people ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return people;
    return people.filter(
      (p) => p.name.toLowerCase().includes(q) || p.public_id.toLowerCase().includes(q),
    );
  }, [data, query]);

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  if (error) return <div className="error-box">{(error as Error).message}</div>;

  return (
    <div className="vp-layout">
      <div className="vp-side">
        <div className="vp-side-head">
          <input
            className="search"
            placeholder={tr("Search people…", "搜索人物…")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ marginBottom: 8 }}
          />
          <button
            className="btn"
            onClick={() => {
              const name = window.prompt(tr("New person name:", "新人物姓名："));
              if (name?.trim()) createMut.mutate(name.trim());
            }}
          >
            + {tr("New person", "新建人物")}
          </button>
        </div>
        <div className="speaker-list flat">
          {filtered.map((p) => (
            <button
              key={p.public_id}
              className={`speaker-card ${selected === p.public_id ? "active" : ""}`}
              onClick={() => setSelected(p.public_id)}
            >
              <div className="speaker-card-top">
                <span className="speaker-name">{p.name}</span>
              </div>
              <div className="speaker-card-meta subtle">
                {p.sample_count} {tr("samples", "样本")} · {p.project_count}{" "}
                {tr("projects", "项目")} · {p.embedded_sample_count}/{p.sample_count}{" "}
                {tr("embedded", "已嵌入")}
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="vp-main">
        {selected ? (
          <PersonSamples
            ref_={selected}
            audio={audio}
            onChanged={() => queryClient.invalidateQueries({ queryKey: ["vp-library"] })}
            onDeleted={() => setSelected(null)}
          />
        ) : (
          <div className="placeholder">{tr("Select a person.", "选择一个人物。")}</div>
        )}
      </div>
    </div>
  );
}

function PersonSamples(props: {
  ref_: string;
  audio: ReturnType<typeof useClipAudio>;
  onChanged: () => void;
  onDeleted: () => void;
}) {
  const { ref_, audio } = props;
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["vp-person", ref_],
    queryFn: () => getPersonSamples(ref_),
  });
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["vp-person", ref_] });
    props.onChanged();
  };
  const renameMut = useMutation({
    mutationFn: (name: string) => renamePerson(ref_, name),
    onSuccess: invalidate,
  });
  // Deletion can't reuse `invalidate`: refetching ["vp-person", ref_] for a person that
  // no longer exists 404s and strands the pane on stale data. Refresh the sidebar, then
  // tell the parent to clear the selection so this pane unmounts cleanly.
  const delPersonMut = useMutation({
    mutationFn: () => deletePerson(ref_),
    onSuccess: () => {
      props.onChanged();
      props.onDeleted();
    },
  });
  const delSampleMut = useMutation({
    mutationFn: (samplePublicId: string) => deleteSample(ref_, samplePublicId),
    // Deleting the person's last sample also removes the now-empty person, so refetching
    // ["vp-person", ref_] would 404 and strand this pane on errored data. When it was the
    // last sample, clear the selection (unmount) like the whole-person delete; otherwise
    // just refresh in place.
    onSuccess: () => {
      if (data && data.samples.length <= 1) {
        props.onChanged();
        props.onDeleted();
      } else {
        invalidate();
      }
    },
  });

  if (isLoading || !data) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  const p = data.person as VoiceprintPerson;

  return (
    <div>
      <div className="transcript-head">
        <h2>{p.name}</h2>
        <div className="row gap">
          <button
            className="btn"
            onClick={() => {
              const name = window.prompt(tr("Rename to:", "改名为："), p.name);
              if (name?.trim() && name.trim() !== p.name) renameMut.mutate(name.trim());
            }}
          >
            {tr("Rename", "改名")}
          </button>
          <button
            className="btn ghost"
            onClick={() => {
              if (window.confirm(tr(`Delete "${p.name}" and all samples?`, `删除「${p.name}」及全部样本？`)))
                delPersonMut.mutate();
            }}
          >
            {tr("Delete person", "删除人物")}
          </button>
        </div>
      </div>
      <div className="segments">
        {data.samples.map((s) => {
          const key = `${ref_}:${s.public_id}`;
          const playing = audio.playingKey === key;
          return (
            <div key={s.public_id} className={`segment ${playing ? "playing" : ""}`}>
              <button
                className="play-btn"
                onClick={() => audio.toggle(key, sampleClipUrl(ref_, s.public_id))}
              >
                {playing ? "⏸" : "▶"}
              </button>
              <div className="segment-body">
                <div className="segment-meta subtle mono">
                  {fmtMs(s.begin_time_ms)} · {s.project_id} ·{" "}
                  <span className={`badge status-pill ${s.status}`}>{s.status}</span>
                </div>
                <div className="segment-text">{s.transcript_text}</div>
                {playing && (
                  <div className="seg-progress">
                    <div className="seg-progress-bar" style={{ width: `${audio.progress * 100}%` }} />
                  </div>
                )}
              </div>
              <button
                className="reassign-btn"
                title={tr("Delete sample", "删除样本")}
                onClick={() => {
                  if (window.confirm(tr("Delete this sample?", "删除这条样本？")))
                    delSampleMut.mutate(s.public_id);
                }}
              >
                🗑
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- Quality ---------------------------------------------------------------

function QualityTab() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["vp-quality"], queryFn: getQuality });
  const [selected, setSelected] = useState<string | null>(null);
  const audio = useClipAudio();

  const statusMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => setSampleStatus(id, status),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["vp-quality"] }),
  });
  const deleteMut = useMutation({
    mutationFn: ({ personRef, samplePublicId }: { personRef: string; samplePublicId: string }) =>
      deleteSample(personRef, samplePublicId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["vp-quality"] });
      queryClient.invalidateQueries({ queryKey: ["vp-library"] });
    },
  });

  if (isLoading) return <div className="placeholder">{tr("Analyzing…", "分析中…")}</div>;
  if (error) return <div className="error-box">{(error as Error).message}</div>;
  if (!data) return null;

  const sorted = [...data.people].sort(
    (a, b) => b.critical_count - a.critical_count || b.suspicious_count - a.suspicious_count,
  );
  // Fall back to the first row when the selection is stale (e.g. the person dropped out
  // of the list), so the highlighted card and the rendered samples never disagree.
  const sel = (selected && sorted.find((p) => p.public_id === selected)) || sorted[0];

  return (
    <div>
      <div className="subtle mono" style={{ marginBottom: 10 }}>
        {data.model} · {data.sample_count} {tr("samples", "样本")} ·{" "}
        <span className="warn">{data.suspicious_count} {tr("suspicious", "疑点")}</span> ·{" "}
        <span style={{ color: "var(--red)" }}>{data.critical_count} {tr("critical", "严重")}</span>
      </div>
      <div className="vp-layout">
        <div className="speaker-list flat vp-side">
          {sorted.map((p) => (
            <button
              key={p.public_id}
              className={`speaker-card ${sel?.public_id === p.public_id ? "active" : ""}`}
              onClick={() => setSelected(p.public_id)}
            >
              <div className="speaker-card-top">
                <span className="speaker-name">{p.name}</span>
                {p.critical_count > 0 && <span className="badge state-broken">{p.critical_count}</span>}
                {p.suspicious_count > 0 && p.critical_count === 0 && (
                  <span className="badge" style={{ color: "var(--yellow)", borderColor: "var(--yellow)" }}>
                    {p.suspicious_count}
                  </span>
                )}
              </div>
              <div className="speaker-card-meta subtle">
                {p.active_sample_count}/{p.sample_count} {tr("active", "活跃")} ·{" "}
                {tr("mean", "均值")} {p.mean_score?.toFixed(2) ?? "—"}
              </div>
            </button>
          ))}
        </div>
        <div className="vp-main">
          {sel ? (
            <QualitySamples
              person={sel}
              audio={audio}
              onSetStatus={(id, status) => statusMut.mutate({ id, status })}
              onDelete={(id) => deleteMut.mutate({ personRef: sel.public_id, samplePublicId: id })}
            />
          ) : (
            <div className="placeholder">{tr("No people.", "暂无人物。")}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function QualitySamples(props: {
  person: QualityPerson;
  audio: ReturnType<typeof useClipAudio>;
  onSetStatus: (samplePublicId: string, status: string) => void;
  onDelete: (samplePublicId: string) => void;
}) {
  const { person, audio, onSetStatus, onDelete } = props;
  return (
    <div>
      <div className="transcript-head">
        <h2>{person.name}</h2>
      </div>
      <div className="segments">
        {person.samples.map((s) => {
          // Quality clips need the owning person ref to build the URL.
          const url = sampleClipUrl(person.public_id, s.sample_public_id);
          const key = `q:${s.sample_public_id}`;
          const playing = audio.playingKey === key;
          return (
            <div key={s.sample_public_id} className={`segment ${playing ? "playing" : ""}`}>
              <button className="play-btn" onClick={() => audio.toggle(key, url)}>
                {playing ? "⏸" : "▶"}
              </button>
              <div className="segment-body">
                <div className="segment-meta subtle mono">
                  {fmtMs(s.begin_time_ms)} · {s.project_id} ·{" "}
                  <span className={`score-badge ${s.label === "critical" ? "low" : s.label === "warning" ? "mid" : "ok"}`}>
                    {s.score?.toFixed(2) ?? "—"} {s.label}
                  </span>{" "}
                  <span className={`badge status-pill ${s.status}`}>{s.status}</span>
                </div>
                <div className="segment-text">{s.transcript_text}</div>
                {s.reason && <div className="subtle" style={{ fontSize: 11.5, marginTop: 3 }}>{s.reason}</div>}
                <div className="row gap" style={{ marginTop: 6 }}>
                  <button className="chip" onClick={() => onSetStatus(s.sample_public_id, "active")}>
                    {tr("Active", "活跃")}
                  </button>
                  <button className="chip" onClick={() => onSetStatus(s.sample_public_id, "quarantined")}>
                    {tr("Quarantine", "隔离")}
                  </button>
                  <button className="chip" onClick={() => onSetStatus(s.sample_public_id, "verified-active")}>
                    {tr("Verify", "确认")}
                  </button>
                </div>
              </div>
              <button
                className="reassign-btn"
                title={tr("Delete sample", "删除样本")}
                onClick={() => {
                  if (window.confirm(tr("Delete this sample?", "删除这条样本？")))
                    onDelete(s.sample_public_id);
                }}
              >
                🗑
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
