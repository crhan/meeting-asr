import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  clipUrl,
  getSpeakerReview,
  saveSpeakerReview,
  type Person,
  type ReviewSpeaker,
  type SaveSpeakerReviewBody,
  type SpeakerReview,
  type SpeakerSegment,
} from "../api/client";
import { tr } from "../lib/i18n";
import { IdentityPicker, type IdentitySelection } from "../components/IdentityPicker";
import { SpeakerPicker } from "../components/SpeakerPicker";

interface SpeakerEdit {
  name: string;
  person_id: number | null;
  person_public_id: string | null;
  ignored: boolean;
}

function segKey(seg: { sentence_id: number | null; begin_time_ms: number; end_time_ms: number }) {
  return `${seg.sentence_id ?? "x"}|${seg.begin_time_ms}|${seg.end_time_ms}`;
}

function fmtMs(ms: number): string {
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function fmtDur(ms: number): string {
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h${m}m`;
  if (m > 0) return `${m}m${s}s`;
  return `${s}s`;
}

const STATUS_LABEL: Record<string, [string, string]> = {
  conflict: ["Conflict", "冲突"],
  mismatch: ["Mismatch", "不一致"],
  ignored: ["Ignored", "已忽略"],
  review: ["Review", "待定"],
  matched: ["Matched", "已匹配"],
  confirmed: ["Confirmed", "已确认"],
};

export function SpeakerReviewPage() {
  const { ref = "" } = useParams();
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["speakers", ref],
    queryFn: () => getSpeakerReview(ref),
  });

  // Working edits layered over the loaded baseline.
  const [edits, setEdits] = useState<Map<number, SpeakerEdit>>(new Map());
  const [reassign, setReassign] = useState<Map<string, number>>(new Map());
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<"all" | "review" | "low">("all");
  const [picking, setPicking] = useState<ReviewSpeaker | null>(null);
  const [reassigning, setReassigning] = useState<SpeakerSegment | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // Reset working state whenever a fresh session loads.
  useEffect(() => {
    if (data) {
      setEdits(new Map());
      setReassign(new Map());
      setSelectedId((prev) => prev ?? data.speakers[0]?.speaker_id ?? null);
    }
  }, [data]);

  const effective = (s: ReviewSpeaker): ReviewSpeaker => {
    const e = edits.get(s.speaker_id);
    if (!e) return s;
    return {
      ...s,
      current_name: e.name,
      person_id: e.person_id,
      person_public_id: e.person_public_id,
      ignored: e.ignored,
    };
  };

  // Effective segment ownership after reassignments.
  const segmentsBySpeaker = useMemo(() => {
    const map = new Map<number, SpeakerSegment[]>();
    if (!data) return map;
    for (const s of data.speakers) map.set(s.speaker_id, []);
    for (const s of data.speakers) {
      for (const seg of s.segments) {
        const owner = reassign.get(segKey(seg)) ?? s.speaker_id;
        if (!map.has(owner)) map.set(owner, []);
        map.get(owner)!.push(seg);
      }
    }
    for (const segs of map.values())
      segs.sort((a, b) => a.begin_time_ms - b.begin_time_ms);
    return map;
  }, [data, reassign]);

  const dirty = edits.size > 0 || reassign.size > 0;

  const saveMutation = useMutation({
    mutationFn: (body: SaveSpeakerReviewBody) => saveSpeakerReview(ref, body),
    onSuccess: (res) => {
      setToast(
        tr(
          `Saved. ${res.reassigned_count} reassigned, ${res.deleted_sample_count} samples invalidated.`,
          `已保存。重指派 ${res.reassigned_count} 句，失效声纹样本 ${res.deleted_sample_count} 个。`,
        ),
      );
      queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
    },
    onError: (e) => setToast(tr("Save failed: ", "保存失败：") + (e as Error).message),
  });

  // ---- keyboard shortcuts (when not typing) -------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!data) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const ids = data.speakers.map((s) => s.speaker_id);
      const idx = selectedId == null ? -1 : ids.indexOf(selectedId);
      if (e.key === "j") {
        setSelectedId(ids[Math.min(ids.length - 1, idx + 1)] ?? ids[0]);
      } else if (e.key === "k") {
        setSelectedId(ids[Math.max(0, idx - 1)] ?? ids[0]);
      } else if (e.key === "/" && selectedId != null) {
        e.preventDefault();
        setPicking(data.speakers.find((s) => s.speaker_id === selectedId) ?? null);
      } else if (e.key === "i" && selectedId != null) {
        toggleIgnore(selectedId);
      } else if (e.key === "a" && selectedId != null) {
        acceptMatch(selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, selectedId, edits]);

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  if (error)
    return (
      <div className="error-box">
        {tr("Failed to load: ", "加载失败：")}
        {(error as Error).message}
      </div>
    );
  if (!data) return null;

  const speakers = data.speakers.map(effective);
  const selected =
    selectedId != null
      ? (speakers.find((s) => s.speaker_id === selectedId) ?? null)
      : null;

  function applySelection(speakerId: number, sel: IdentitySelection) {
    setEdits((prev) => {
      const next = new Map(prev);
      next.set(speakerId, {
        name: sel.name,
        person_id: sel.person_id,
        person_public_id: sel.person_public_id,
        ignored: sel.ignored,
      });
      return next;
    });
    setPicking(null);
  }

  function acceptMatch(speakerId: number) {
    const s = data!.speakers.find((x) => x.speaker_id === speakerId);
    const best = s?.match?.best_name;
    if (!s || !best || (s.match?.best_score ?? 0) <= 0) return;
    const cand = s.match!.candidates.find((c) => c.name === best);
    applySelection(speakerId, {
      name: best,
      person_id: cand?.person_id ?? null,
      person_public_id: cand?.person_public_id ?? null,
      ignored: false,
    });
  }

  function toggleIgnore(speakerId: number) {
    const s = effective(data!.speakers.find((x) => x.speaker_id === speakerId)!);
    setEdits((prev) => {
      const next = new Map(prev);
      if (s.ignored) {
        next.set(speakerId, {
          name: s.label,
          person_id: null,
          person_public_id: null,
          ignored: false,
        });
      } else {
        next.set(speakerId, {
          name: s.label,
          person_id: null,
          person_public_id: null,
          ignored: true,
        });
      }
      return next;
    });
  }

  function doReassign(seg: SpeakerSegment, targetSpeakerId: number) {
    setReassign((prev) => {
      const next = new Map(prev);
      const original = seg.speaker_id;
      if (targetSpeakerId === original) next.delete(segKey(seg));
      else next.set(segKey(seg), targetSpeakerId);
      return next;
    });
    setReassigning(null);
  }

  function buildSaveBody(): SaveSpeakerReviewBody {
    const mapping: Record<string, string> = {};
    const person_mapping: Record<string, number> = {};
    const person_public_mapping: Record<string, string> = {};
    const ignored_speaker_ids: number[] = [];
    for (const s of speakers) {
      const name = s.current_name.trim() || s.label;
      mapping[s.speaker_id] = name;
      if (s.person_id != null && !s.ignored) person_mapping[s.speaker_id] = s.person_id;
      if (s.person_public_id && !s.ignored)
        person_public_mapping[s.speaker_id] = s.person_public_id;
      if (s.ignored && name === s.label) ignored_speaker_ids.push(s.speaker_id);
    }
    const reassignments = [...reassign.entries()].flatMap(([key, newId]) => {
      const seg = data!.speakers
        .flatMap((sp) => sp.segments)
        .find((sg) => segKey(sg) === key);
      if (!seg || seg.speaker_id === newId) return [];
      return [
        {
          sentence_id: seg.sentence_id,
          begin_time_ms: seg.begin_time_ms,
          end_time_ms: seg.end_time_ms,
          original_speaker_id: seg.speaker_id,
          new_speaker_id: newId,
        },
      ];
    });
    return {
      mapping,
      person_mapping,
      person_public_mapping,
      ignored_speaker_ids,
      reassignments,
    };
  }

  const unresolved = speakers.filter(
    (s) => s.status === "review" || s.status === "conflict" || s.status === "mismatch",
  ).length;

  return (
    <div className="review">
      <ReviewHeader
        review={data}
        speakerCount={speakers.length}
        unresolved={unresolved}
        dirty={dirty}
        saving={saveMutation.isPending}
        onSave={() => saveMutation.mutate(buildSaveBody())}
        onDiscard={() => {
          setEdits(new Map());
          setReassign(new Map());
        }}
      />
      <div className="review-body">
        <SpeakerSidebar
          speakers={speakers}
          segmentsBySpeaker={segmentsBySpeaker}
          selectedId={selectedId}
          editedIds={new Set([...edits.keys()])}
          onSelect={setSelectedId}
        />
        <TranscriptPane
          projectRef={ref}
          selected={selected}
          segments={selected ? (segmentsBySpeaker.get(selected.speaker_id) ?? []) : []}
          filter={filter}
          reassignKeys={reassign}
          onFilter={setFilter}
          onIdentify={() => selected && setPicking(data.speakers.find((s) => s.speaker_id === selected.speaker_id) ?? null)}
          onAccept={() => selected && acceptMatch(selected.speaker_id)}
          onIgnore={() => selected && toggleIgnore(selected.speaker_id)}
          onReassign={(seg) => setReassigning(seg)}
        />
      </div>

      {picking && (
        <IdentityPicker
          speaker={effective(picking)}
          people={data.people as Person[]}
          onSelect={(sel) => applySelection(picking.speaker_id, sel)}
          onClose={() => setPicking(null)}
        />
      )}
      {reassigning && (
        <SpeakerPicker
          speakers={speakers}
          currentSpeakerId={reassign.get(segKey(reassigning)) ?? reassigning.speaker_id ?? -1}
          sentencePreview={reassigning.text}
          onPick={(target) => doReassign(reassigning, target)}
          onClose={() => setReassigning(null)}
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

// ---- sub-components ---------------------------------------------------------

function ReviewHeader(props: {
  review: SpeakerReview;
  speakerCount: number;
  unresolved: number;
  dirty: boolean;
  saving: boolean;
  onSave: () => void;
  onDiscard: () => void;
}) {
  const { review, speakerCount, unresolved, dirty, saving, onSave, onDiscard } = props;
  const o = review.overview;
  return (
    <div className="review-head">
      <div>
        <h1>{o.title || tr("(untitled)", "（无标题）")}</h1>
        <div className="subtle mono">
          {o.project_id} · {fmtDur(o.duration_ms)} · {speakerCount}{" "}
          {tr("speakers", "位发言人")}
          {unresolved > 0 && (
            <span className="warn"> · {unresolved} {tr("unresolved", "待定")}</span>
          )}
        </div>
      </div>
      <div className="row gap">
        {dirty && (
          <button className="btn ghost" onClick={onDiscard} disabled={saving}>
            {tr("Discard", "撤销")}
          </button>
        )}
        <button className="btn primary" onClick={onSave} disabled={!dirty || saving}>
          {saving ? tr("Saving…", "保存中…") : tr("Save", "保存")}
        </button>
      </div>
    </div>
  );
}

function SpeakerSidebar(props: {
  speakers: ReviewSpeaker[];
  segmentsBySpeaker: Map<number, SpeakerSegment[]>;
  selectedId: number | null;
  editedIds: Set<number>;
  onSelect: (id: number) => void;
}) {
  const { speakers, segmentsBySpeaker, selectedId, editedIds, onSelect } = props;
  return (
    <div className="speaker-list">
      {speakers.map((s) => {
        const segs = segmentsBySpeaker.get(s.speaker_id) ?? [];
        const dur = segs.reduce((acc, seg) => acc + (seg.end_time_ms - seg.begin_time_ms), 0);
        const label = STATUS_LABEL[s.status]?.[0] ?? s.status;
        return (
          <button
            key={s.speaker_id}
            className={`speaker-card ${s.speaker_id === selectedId ? "active" : ""}`}
            onClick={() => onSelect(s.speaker_id)}
          >
            <div className="speaker-card-top">
              <span className={`status-dot status-${s.status}`} title={label} />
              <span className="speaker-name">{s.current_name || s.label}</span>
              {editedIds.has(s.speaker_id) && <span className="dot-edited" title="edited" />}
            </div>
            <div className="speaker-card-meta subtle">
              {s.label} · {segs.length} {tr("seg", "句")} · {fmtDur(dur)}
              {s.crosstalk && <span className="badge crosstalk">{tr("crosstalk", "串场")}</span>}
            </div>
            {s.match?.best_name && (
              <div className="speaker-card-match subtle mono">
                ~ {s.match.best_name}{" "}
                {s.match.best_score != null && s.match.best_score.toFixed(2)}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

function TranscriptPane(props: {
  projectRef: string;
  selected: ReviewSpeaker | null;
  segments: SpeakerSegment[];
  filter: "all" | "review" | "low";
  reassignKeys: Map<string, number>;
  onFilter: (f: "all" | "review" | "low") => void;
  onIdentify: () => void;
  onAccept: () => void;
  onIgnore: () => void;
  onReassign: (seg: SpeakerSegment) => void;
}) {
  const { projectRef, selected, segments, filter, reassignKeys } = props;
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const play = (seg: SpeakerSegment) => {
    const el = audioRef.current;
    if (!el) return;
    const key = segKey(seg);
    if (playingKey === key && !el.paused) {
      el.pause();
      setPlayingKey(null);
      return;
    }
    el.src = clipUrl(projectRef, seg.begin_time_ms, seg.end_time_ms);
    el.play();
    setPlayingKey(key);
    setProgress(0);
  };

  const filtered = segments.filter((seg) => {
    if (filter === "all") return true;
    if (seg.score == null) return false;
    if (filter === "review") return seg.score_status !== "ok" || seg.score < 0.6;
    return seg.score < 0.45;
  });

  if (!selected) return <div className="placeholder">{tr("Select a speaker.", "选择一位发言人。")}</div>;

  const canAccept = !!selected.match?.best_name && (selected.match?.best_score ?? 0) > 0;

  return (
    <div className="transcript-pane">
      <div className="transcript-head">
        <div className="row gap center">
          <span className={`status-dot status-${selected.status}`} />
          <h2>{selected.current_name || selected.label}</h2>
          <span className="badge">{STATUS_LABEL[selected.status]?.[1] ?? selected.status}</span>
          {selected.crosstalk && <span className="badge crosstalk">{tr("crosstalk", "串场")}</span>}
        </div>
        <div className="row gap">
          <button className="btn" onClick={props.onIdentify}>
            {tr("Identify", "指认")} <span className="kbd">/</span>
          </button>
          {canAccept && (
            <button className="btn" onClick={props.onAccept} title={selected.match?.best_name ?? ""}>
              {tr("Accept match", "接受匹配")} <span className="kbd">A</span>
            </button>
          )}
          <button className={`btn ghost ${selected.ignored ? "on" : ""}`} onClick={props.onIgnore}>
            {selected.ignored ? tr("Ignored", "已忽略") : tr("Ignore", "忽略")}{" "}
            <span className="kbd">I</span>
          </button>
        </div>
      </div>

      <div className="filter-bar">
        {(["all", "review", "low"] as const).map((f) => (
          <button
            key={f}
            className={`chip ${filter === f ? "on" : ""}`}
            onClick={() => props.onFilter(f)}
          >
            {f === "all" ? tr("All", "全部") : f === "review" ? tr("For review", "疑点") : tr("Low score", "低分")}
          </button>
        ))}
        <span className="subtle">
          {filtered.length}/{segments.length}
        </span>
      </div>

      <audio
        ref={audioRef}
        onTimeUpdate={(e) => {
          const el = e.currentTarget;
          if (el.duration) setProgress(el.currentTime / el.duration);
        }}
        onEnded={() => setPlayingKey(null)}
      />

      <div className="segments">
        {filtered.map((seg) => {
          const key = segKey(seg);
          const reassigned = reassignKeys.has(key);
          const playing = playingKey === key;
          return (
            <div key={key} className={`segment ${playing ? "playing" : ""} ${reassigned ? "reassigned" : ""}`}>
              <button className="play-btn" onClick={() => play(seg)} aria-label="play">
                {playing ? "⏸" : "▶"}
              </button>
              <div className="segment-body">
                <div className="segment-meta subtle mono">
                  {fmtMs(seg.begin_time_ms)}
                  {seg.score != null && (
                    <span className={`score-badge ${seg.score < 0.45 ? "low" : seg.score < 0.6 ? "mid" : "ok"}`}>
                      {seg.score.toFixed(2)}
                    </span>
                  )}
                  {reassigned && <span className="badge reassigned-badge">{tr("reassigned", "已重指派")}</span>}
                </div>
                <div className="segment-text">{seg.text}</div>
                {playing && (
                  <div className="seg-progress">
                    <div className="seg-progress-bar" style={{ width: `${progress * 100}%` }} />
                  </div>
                )}
              </div>
              <button className="reassign-btn" onClick={() => props.onReassign(seg)} title={tr("Reassign", "重指派")}>
                ⇄
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
