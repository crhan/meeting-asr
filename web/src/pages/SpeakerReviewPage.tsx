import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ApiError,
  clipUrl,
  excludeQualitySamples,
  getQuality,
  getSpeakerReview,
  rematchSpeakerReview,
  saveSpeakerReview,
  type Person,
  type QualityPerson,
  type InlineCorrectionEdit,
  type ReviewSpeaker,
  type SaveSpeakerReviewBody,
  type SpeakerReview,
  type SpeakerSegment,
} from "../api/client";
import { tr } from "../lib/i18n";
import { setUnsavedEdits } from "../lib/unsavedGuard";
import { useClipAudio } from "../lib/useClipAudio";
import { confirmDialog } from "../lib/confirm";
import { ExportsModal } from "../components/ExportsModal";
import { IdentityPicker, type IdentitySelection } from "../components/IdentityPicker";
import { anyModalOpen, Modal } from "../components/Modal";
import { SpeakerPicker } from "../components/SpeakerPicker";

interface SpeakerEdit {
  name: string;
  person_id: number | null;
  person_public_id: string | null;
  ignored: boolean;
  create_person?: boolean;
  // Identity stashed when ignoring, so un-ignoring restores the real name/binding instead
  // of the generic label (which buildSaveBody would persist, erasing a confirmed name).
  priorName?: string;
  priorPersonId?: number | null;
  priorPersonPublicId?: string | null;
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

interface ParsedSentenceLocator {
  projectId: string | null;
  sentenceId: number;
}

function parseSentenceLocator(value: string | null): ParsedSentenceLocator | null {
  if (!value?.trim()) return null;
  const raw = value.trim();
  const hashIndex = raw.indexOf("#");
  const projectRaw = hashIndex >= 0 ? raw.slice(0, hashIndex) : null;
  const sentenceRaw = hashIndex >= 0 ? raw.slice(hashIndex + 1) : raw;
  const projectId = projectRaw?.trim() || null;
  if (hashIndex >= 0 && !projectId) return null;
  const parsed = Number(sentenceRaw.trim());
  if (!Number.isInteger(parsed) || parsed < 0) return null;
  return { projectId, sentenceId: parsed };
}

function formatSentenceLocator(projectId: string, sentenceId: number | null): string | null {
  return sentenceId == null ? null : `${projectId}#${sentenceId}`;
}

function findSentence(review: SpeakerReview, sentenceId: number) {
  for (const speaker of review.speakers) {
    const segment = speaker.segments.find((seg) => seg.sentence_id === sentenceId);
    if (segment) return { speaker, segment };
  }
  return null;
}

const STATUS_LABEL: Record<string, [string, string]> = {
  conflict: ["Conflict", "冲突"],
  mismatch: ["Mismatch", "不一致"],
  ignored: ["Ignored", "已忽略"],
  review: ["Review", "待定"],
  matched: ["Matched", "已匹配"],
  confirmed: ["Confirmed", "已确认"],
};

/** Language-aware status label (the sidebar/badge previously hardcoded en vs zh). */
function statusLabel(status: string): string {
  const pair = STATUS_LABEL[status];
  return pair ? tr(pair[0], pair[1]) : status;
}

const IDENTITY_SCORE_STATUS_LABEL: Record<string, [string, string]> = {
  "identity-ok": ["ok", "正常"],
  "identity-conflict": ["closer to another person", "更像他人"],
  "identity-ambiguous": ["close to another person", "候选接近"],
  "identity-weak": ["weak voiceprint evidence", "声纹偏弱"],
  "low-info": ["too short", "过短"],
  "no-assignment": ["identity not assigned", "未绑定身份"],
};

function identityStatusLabel(status: string | null): string {
  if (!status) return tr("not available", "无状态");
  const pair = IDENTITY_SCORE_STATUS_LABEL[status];
  return pair ? tr(pair[0], pair[1]) : status;
}

function identityScoreClass(score: number, status: string | null = null): "ok" | "mid" | "low" {
  if (status === "identity-conflict") return "low";
  if (status && status !== "identity-ok") return "mid";
  if (score < 0.45) return "low";
  if (score < 0.6) return "mid";
  return "ok";
}

function identityScoreReason(seg: SpeakerSegment): string | null {
  if (seg.score == null) return null;
  if (seg.score_status && seg.score_status !== "identity-ok") {
    return identityStatusLabel(seg.score_status);
  }
  if (seg.score < 0.45) return tr("below 0.45", "低于0.45");
  if (seg.score < 0.6) return tr("below 0.60", "低于0.60");
  return null;
}

function fmtScore(value: number | null): string {
  return value == null ? "—" : value.toFixed(2);
}

function identityDiagnosticEvidence(seg: SpeakerSegment): string | null {
  const assigned = fmtScore(seg.score);
  const bestName = seg.score_best_name ?? seg.score_best_other_name;
  const bestScore = seg.score_best_score ?? seg.score_best_other_score;
  const margin = fmtScore(seg.score_margin);
  if (seg.score_status === "identity-conflict" && bestName) {
    return tr(
      `Voiceprint similarity to current identity ${assigned}; closer to ${bestName} ${fmtScore(bestScore)}; margin ${margin}.`,
      `与当前身份的声纹相似度 ${assigned}；更像 ${bestName} ${fmtScore(bestScore)}；差值 ${margin}。`,
    );
  }
  if (seg.score_status === "identity-ambiguous") {
    const candidate = bestName ? tr(` Candidate ${bestName} ${fmtScore(bestScore)}.`, ` 候选 ${bestName} ${fmtScore(bestScore)}。`) : "";
    return tr(
      `Voiceprint similarity to current identity ${assigned}; candidate scores are close; margin ${margin}.${candidate}`,
      `与当前身份的声纹相似度 ${assigned}；候选分数接近；差值 ${margin}。${candidate}`,
    );
  }
  if (seg.score_status === "identity-weak") {
    return tr(
      `Voiceprint similarity to current identity ${assigned}; evidence is weak.`,
      `与当前身份的声纹相似度 ${assigned}；证据偏弱。`,
    );
  }
  if (seg.score_status === "low-info") {
    return tr(
      "This sentence is too short or low-information for a reliable voiceprint call.",
      "这句话太短或信息量低，不能单独作为可靠声纹判断。",
    );
  }
  if (seg.score_status === "no-assignment") {
    return tr(
      "This speaker has no assigned identity, so the sentence cannot be checked against a confirmed person.",
      "这个 speaker 还没有绑定身份，所以这句话无法和确认身份比对。",
    );
  }
  if (seg.score != null && seg.score < 0.6) {
    return tr(
      `Voiceprint similarity to current identity ${assigned}; below the normal review threshold.`,
      `与当前身份的声纹相似度 ${assigned}；低于正常复核阈值。`,
    );
  }
  return null;
}

function identityDiagnosticSuggestion(seg: SpeakerSegment): string | null {
  if (seg.score_status === "identity-conflict") {
    return tr(
      "Listen to the original audio. If the speaker is the better match, move this sentence or merge the affected speaker; if the library is noisy, exclude bad samples and rematch.",
      "先听原音频。若确实是更像的那个人，把这句改派或合并相关 speaker；若是声纹库污染，先排除坏样本再重跑匹配。",
    );
  }
  if (seg.score_status === "identity-ambiguous") {
    return tr(
      "Do not move it from this score alone. Check neighboring sentences; move only when several adjacent sentences point to the same person.",
      "不要只凭这一句改派。先看相邻句，只有连续多句都指向同一个人时再移动。",
    );
  }
  if (seg.score_status === "identity-weak") {
    return tr(
      "Treat as weak evidence. Prefer a longer nearby sample or add better voiceprint samples before changing ownership.",
      "按弱证据处理。优先找附近更长的句子，或补更干净的声纹样本后再判断归属。",
    );
  }
  if (seg.score_status === "low-info") {
    return tr(
      "Ignore this as a standalone signal; use context before and after it.",
      "不要单独采信这句；结合前后上下文判断。",
    );
  }
  if (seg.score_status === "no-assignment") {
    return tr(
      "Identify this speaker or accept a reliable project match first.",
      "先给这个 speaker 指认身份，或接受一个可靠的项目匹配。",
    );
  }
  if (seg.score != null && seg.score < 0.6) {
    return tr(
      "Review the audio and compare with nearby turns before changing ownership.",
      "听原音频，并和附近轮次一起比较后再决定是否改归属。",
    );
  }
  return null;
}

function identityScoreTitle(seg: SpeakerSegment): string {
  const value = seg.score?.toFixed(2) ?? "—";
  const status = identityStatusLabel(seg.score_status);
  const reason = identityScoreReason(seg) ?? tr("none", "无");
  const evidence = identityDiagnosticEvidence(seg) ?? "";
  const suggestion = identityDiagnosticSuggestion(seg) ?? "";
  return tr(
    `Voiceprint match score: cosine similarity between this sentence's audio and the current speaker's assigned person in the voiceprint library. It is not a probability; higher means more similar. Score ${value}; status ${status}; review reason ${reason}. ${evidence} ${suggestion}`,
    `声纹匹配分：这句话的音频和当前 speaker 已绑定人物在声纹库里的相似度（cosine similarity）。它不是概率；越高表示越像。分数 ${value}；状态 ${status}；疑点原因 ${reason}。${evidence} ${suggestion}`,
  );
}

export function SpeakerReviewPage() {
  const { ref = "" } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["speakers", ref],
    queryFn: () => getSpeakerReview(ref),
  });
  const qualityQuery = useQuery({ queryKey: ["vp-quality"], queryFn: getQuality });
  const focusedSentenceLocator = useMemo(
    () => parseSentenceLocator(searchParams.get("sentence") ?? searchParams.get("sid")),
    [searchParams],
  );
  const focusedSentenceId = focusedSentenceLocator?.sentenceId ?? null;

  useEffect(() => {
    if (!focusedSentenceLocator?.projectId || !data) return;
    if (focusedSentenceLocator.projectId === data.project_id) return;
    navigate(
      `/projects/${encodeURIComponent(focusedSentenceLocator.projectId)}/speakers?sentence=${encodeURIComponent(`${focusedSentenceLocator.projectId}#${focusedSentenceLocator.sentenceId}`)}`,
      { replace: true },
    );
  }, [data, focusedSentenceLocator, navigate]);

  // Working edits layered over the loaded baseline.
  const [edits, setEdits] = useState<Map<number, SpeakerEdit>>(new Map());
  const [reassign, setReassign] = useState<Map<string, number>>(new Map());
  const [textEdits, setTextEdits] = useState<Map<string, InlineCorrectionEdit>>(new Map());
  const [deletedSpeakerIds, setDeletedSpeakerIds] = useState<Set<number>>(new Set());
  // Locally minted speakers (ASR under-split rescue): exist only as staged state until
  // save; the backend accepts reassignments to unseen speaker ids (same path the TUI uses).
  const [extraSpeakers, setExtraSpeakers] = useState<ReviewSpeaker[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<"all" | "review" | "low">("all");
  const [view, setView] = useState<"speakers" | "timeline">("speakers");
  const [picking, setPicking] = useState<ReviewSpeaker | null>(null);
  const [reassigning, setReassigning] = useState<SpeakerSegment | null>(null);
  const [editingText, setEditingText] = useState<SpeakerSegment | null>(null);
  const [merging, setMerging] = useState<ReviewSpeaker | null>(null);
  const [showExports, setShowExports] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  // Reset working state whenever a fresh session loads. Selection is preserved when the
  // speaker still exists -- a save/rematch refetch used to bounce the user back to the
  // first speaker mid-review.
  useEffect(() => {
    if (data) {
      setEdits(new Map());
      setReassign(new Map());
      setTextEdits(new Map());
      setDeletedSpeakerIds(new Set());
      setExtraSpeakers([]);
      setSelectedId((prev) =>
        prev != null && data.speakers.some((s) => s.speaker_id === prev)
          ? prev
          : (data.speakers[0]?.speaker_id ?? null),
      );
      setMerging(null);
    }
    // focusedSentenceId is handled by the locator effect below; a query-param change must not
    // reset unsaved review edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // Consume the ?sentence= locator ONCE: it must focus on arrival (deep link / jump),
  // but a save/rematch refetch of `data` must not re-hijack the selection afterwards.
  const consumedLocatorRef = useRef<string | null>(null);
  useEffect(() => {
    if (!data || focusedSentenceId == null) return;
    const key = `${data.project_id}#${focusedSentenceId}`;
    if (consumedLocatorRef.current === key) return;
    const focused = findSentence(data, focusedSentenceId);
    if (!focused) {
      const display = formatSentenceLocator(data.project_id, focusedSentenceId);
      setToast(tr(`Sentence ${display} not found.`, `未找到句子 ${display}。`));
      return;
    }
    consumedLocatorRef.current = key;
    setSelectedId(focused.speaker.speaker_id);
    setFilter("all");
  }, [data, focusedSentenceId]);

  // Loaded speakers plus locally minted ones -- every lookup that used data.speakers
  // must see the minted speakers too (reassign targets, keyboard nav, save body).
  const allSpeakers = useMemo<ReviewSpeaker[]>(
    () => (data ? [...data.speakers, ...extraSpeakers] : []),
    [data, extraSpeakers],
  );

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
    for (const s of allSpeakers) map.set(s.speaker_id, []);
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
  }, [data, allSpeakers, reassign]);

  const dirty =
    edits.size > 0 ||
    reassign.size > 0 ||
    textEdits.size > 0 ||
    deletedSpeakerIds.size > 0 ||
    extraSpeakers.length > 0;

  // Unsaved edits live only in this component's state; losing the page loses them.
  // Publish the dirty flag for app chrome (topbar GuardedNavLink + LangToggle's reload
  // confirm) and warn on reload/close via beforeunload. Browser back/forward remains
  // unguarded: useBlocker needs a data router and we stay on plain <BrowserRouter>
  // (see lib/unsavedGuard.ts).
  useEffect(() => {
    setUnsavedEdits(dirty);
    return () => setUnsavedEdits(false);
  }, [dirty]);
  useEffect(() => {
    if (!dirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  const saveMutation = useMutation({
    mutationFn: (body: SaveSpeakerReviewBody) => saveSpeakerReview(ref, body),
    onSuccess: (res) => {
      setToast(
        tr(
          `Saved. ${res.reassigned_count} reassigned, ${res.corrected_count} text corrected, ${res.created_person_count} people created, ${res.deleted_sample_count} samples invalidated.`,
          `已保存。重指派 ${res.reassigned_count} 句，文字修正 ${res.corrected_count} 句，新建人物 ${res.created_person_count} 个，失效声纹样本 ${res.deleted_sample_count} 个。`,
        ),
      );
      queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
      // The capture plan is built from the saved speaker map + reassignments and is cached with
      // staleTime: Infinity, so a rename/ignore/reassign here would otherwise leave the capture
      // page showing a plan that no longer matches what /capture/run recomputes server-side.
      queryClient.invalidateQueries({ queryKey: ["capture-plan", ref] });
      if (res.created_person_count > 0) {
        queryClient.invalidateQueries({ queryKey: ["vp-library"] });
        queryClient.invalidateQueries({ queryKey: ["vp-quality"] });
      }
    },
    onError: (e) => setToast(tr("Save failed: ", "保存失败：") + (e as Error).message),
  });

  // Standalone rematch against the current voiceprint library -- previously only
  // reachable through the risk-gated repair panel or implicitly inside save.
  const rematchMutation = useMutation({
    mutationFn: () => rematchSpeakerReview(ref),
    onSuccess: (res) => {
      setToast(
        tr(
          `Rematched: ${res.matched_count} matched, ${res.below_threshold_count} below threshold, ${res.total_count} total.`,
          `已重跑匹配：命中 ${res.matched_count}，低于阈值 ${res.below_threshold_count}，共 ${res.total_count}。`,
        ),
      );
      queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
      // The capture plan caches with staleTime: Infinity and depends on match state.
      queryClient.invalidateQueries({ queryKey: ["capture-plan", ref] });
      // The projects list's "needs review" badge may flip.
      queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
    onError: (e) => setToast(tr("Rematch failed: ", "重跑匹配失败：") + (e as Error).message),
  });

  const repairLibraryMutation = useMutation({
    mutationFn: async ({ personRef }: { personRef: string }) => {
      const excluded = await excludeQualitySamples(personRef);
      const rematched = await rematchSpeakerReview(ref);
      return { excluded, rematched };
    },
    onSuccess: ({ excluded, rematched }) => {
      setToast(
        tr(
          `Excluded ${excluded.updated_count} low-quality sample(s); refreshed ${rematched.total_count} speaker matches.`,
          `已排除 ${excluded.updated_count} 条低质样本；已刷新 ${rematched.total_count} 个 speaker 匹配。`,
        ),
      );
      queryClient.invalidateQueries({ queryKey: ["speakers", ref] });
      queryClient.invalidateQueries({ queryKey: ["vp-library"] });
      queryClient.invalidateQueries({ queryKey: ["vp-quality"] });
    },
    onError: (e) => setToast(tr("Repair failed: ", "修库失败：") + (e as Error).message),
  });

  // ---- keyboard shortcuts (when not typing) -------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!data) return;
      // While any dialog is up (pickers, text editor, confirm/prompt hosts), these
      // shortcuts would silently stage edits on the page behind it.
      if (anyModalOpen()) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const ids = allSpeakers
        .map((s) => s.speaker_id)
        .filter((id) => !deletedSpeakerIds.has(id));
      const idx = selectedId == null ? -1 : ids.indexOf(selectedId);
      if (e.key === "j") {
        setSelectedId(ids[Math.min(ids.length - 1, idx + 1)] ?? ids[0]);
      } else if (e.key === "k") {
        setSelectedId(ids[Math.max(0, idx - 1)] ?? ids[0]);
      } else if (e.key === "/" && selectedId != null) {
        e.preventDefault();
        setPicking(allSpeakers.find((s) => s.speaker_id === selectedId) ?? null);
      } else if (e.key === "i" && selectedId != null) {
        toggleIgnore(selectedId);
      } else if (e.key === "a" && selectedId != null) {
        acceptMatch(selectedId);
      } else if (e.key === "m" && selectedId != null) {
        const base = allSpeakers.find((s) => s.speaker_id === selectedId);
        if (base) setMerging(base);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, allSpeakers, selectedId, edits, deletedSpeakerIds]);

  if (isLoading) return <div className="placeholder">{tr("Loading…", "加载中…")}</div>;
  if (error) {
    // A 404 here usually means "not transcribed yet", not a broken server -- say so
    // instead of dumping a raw errno-style message.
    if (error instanceof ApiError && error.status === 404) {
      return (
        <div className="placeholder">
          {tr(
            "This project has no transcript to review yet (or the id is unknown). Run the pipeline from the Projects page first.",
            "该项目还没有可复核的转写（或项目 ID 不存在）。请先在「项目」页对它运行管线。",
          )}
        </div>
      );
    }
    return (
      <div className="error-box">
        {tr("Failed to load: ", "加载失败：")}
        {(error as Error).message}
      </div>
    );
  }
  if (!data) return null;

  const speakers = allSpeakers
    .map(effective)
    .filter((s) => !deletedSpeakerIds.has(s.speaker_id));
  const selected =
    selectedId != null
      ? (speakers.find((s) => s.speaker_id === selectedId) ?? null)
      : null;
  const qualityByPerson = new Map(
    (qualityQuery.data?.people ?? []).map((person) => [person.public_id, person]),
  );
  const selectedQualityRef =
    selected?.person_public_id ??
    selected?.match?.candidates.find((candidate) => candidate.name === selected.match?.best_name)
      ?.person_public_id ??
    null;
  const selectedQuality = selectedQualityRef ? qualityByPerson.get(selectedQualityRef) : undefined;

  function applySelection(speakerId: number, sel: IdentitySelection) {
    setEdits((prev) => {
      const next = new Map(prev);
      next.set(speakerId, {
        name: sel.name,
        person_id: sel.person_id,
        person_public_id: sel.person_public_id,
        ignored: sel.ignored,
        create_person: sel.create_person,
      });
      return next;
    });
    setPicking(null);
  }

  function acceptMatch(speakerId: number) {
    const s = allSpeakers.find((x) => x.speaker_id === speakerId);
    const best = s?.match?.best_name;
    if (!s || !best || (s.match?.best_score ?? 0) <= 0) return;
    const cand = s.match!.candidates.find((c) => c.name === best);
    applySelection(speakerId, {
      name: best,
      person_id: cand?.person_id ?? null,
      person_public_id: cand?.person_public_id ?? null,
      ignored: false,
      create_person: false,
    });
  }

  function toggleIgnore(speakerId: number) {
    const base = allSpeakers.find((x) => x.speaker_id === speakerId)!;
    const s = effective(base);
    setEdits((prev) => {
      const next = new Map(prev);
      if (s.ignored) {
        // Un-ignore: restore the identity the speaker had before it was ignored (stashed at
        // ignore time), falling back to the loaded baseline. Writing s.label here instead
        // would make buildSaveBody persist "Speaker N" over a confirmed name like "Alice".
        const prior = prev.get(speakerId);
        next.set(speakerId, {
          name: prior?.priorName ?? base.current_name,
          person_id: prior?.priorPersonId ?? base.person_id,
          person_public_id: prior?.priorPersonPublicId ?? base.person_public_id,
          ignored: false,
          create_person: false,
        });
      } else {
        // Ignore: the label is sent as the name because buildSaveBody marks a speaker ignored
        // only when name === label; stash the real identity so un-ignore can restore it.
        next.set(speakerId, {
          name: s.label,
          person_id: null,
          person_public_id: null,
          ignored: true,
          create_person: false,
          priorName: s.current_name,
          priorPersonId: s.person_id,
          priorPersonPublicId: s.person_public_id,
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

  // ASR under-split rescue: mint a brand-new staged speaker and move the sentence to it.
  // The save endpoint accepts reassignments to unseen ids (the TUI mints the same way);
  // ids are max+1, monotonic, never reused.
  function createSpeakerAndReassign(seg: SpeakerSegment) {
    const nextId = Math.max(0, ...allSpeakers.map((s) => s.speaker_id)) + 1;
    const label = `Speaker ${nextId}`;
    setExtraSpeakers((prev) => [
      ...prev,
      {
        speaker_id: nextId,
        label,
        current_name: label,
        ignored: false,
        person_id: null,
        person_public_id: null,
        status: "review",
        crosstalk: false,
        segment_count: 0,
        duration_ms: 0,
        match: null,
        segments: [],
      },
    ]);
    doReassign(seg, nextId);
    setSelectedId(nextId);
    setToast(
      tr(
        `Created ${label} and moved the sentence there. Identify it, then save.`,
        `已新建 ${label} 并把这句移入。请指认身份后保存。`,
      ),
    );
  }

  function stageTextEdit(seg: SpeakerSegment, correctedText: string) {
    const corrected = correctedText.trim();
    if (!corrected) {
      setToast(tr("Corrected text cannot be empty.", "修正后的文本不能为空。"));
      return;
    }
    const key = segKey(seg);
    setTextEdits((prev) => {
      const next = new Map(prev);
      const existing = prev.get(key);
      const original = existing?.original_text ?? seg.text.trim();
      if (corrected === original) next.delete(key);
      else {
        next.set(key, {
          sentence_id: seg.sentence_id,
          speaker_id: seg.speaker_id,
          begin_time_ms: seg.begin_time_ms,
          end_time_ms: seg.end_time_ms,
          original_text: original,
          corrected_text: corrected,
        });
      }
      return next;
    });
    setEditingText(null);
  }

  function canDeleteSpeaker(speakerId: number): boolean {
    const segs = segmentsBySpeaker.get(speakerId) ?? [];
    return segs.every((seg) => !seg.text.trim());
  }

  function doMergeSpeaker(sourceSpeakerId: number, targetSpeakerId: number) {
    if (sourceSpeakerId === targetSpeakerId) {
      setMerging(null);
      return;
    }
    const source = speakers.find((s) => s.speaker_id === sourceSpeakerId);
    const target = speakers.find((s) => s.speaker_id === targetSpeakerId);
    const sourceSegments = segmentsBySpeaker.get(sourceSpeakerId) ?? [];
    setReassign((prev) => {
      const next = new Map(prev);
      for (const seg of sourceSegments) {
        if (targetSpeakerId === seg.speaker_id) next.delete(segKey(seg));
        else next.set(segKey(seg), targetSpeakerId);
      }
      return next;
    });
    setDeletedSpeakerIds((prev) => {
      const next = new Set(prev);
      next.add(sourceSpeakerId);
      next.delete(targetSpeakerId);
      return next;
    });
    setSelectedId(targetSpeakerId);
    setMerging(null);
    setToast(
      tr(
        `Queued merge and source delete: ${source?.current_name ?? sourceSpeakerId} -> ${target?.current_name ?? targetSpeakerId}. Save to apply.`,
        `已准备合并并删除源 speaker：${source?.current_name ?? sourceSpeakerId} -> ${target?.current_name ?? targetSpeakerId}。保存后生效。`,
      ),
    );
  }

  function doDeleteSpeaker(speakerId: number) {
    const s = speakers.find((item) => item.speaker_id === speakerId);
    if (!canDeleteSpeaker(speakerId)) {
      setToast(
        tr(
          "This speaker still has non-empty sentences. Merge/reassign or clear them before deleting.",
          "这个 speaker 还有非空句子。先合并、重指派或清空后才能删除。",
        ),
      );
      return;
    }
    setDeletedSpeakerIds((prev) => new Set(prev).add(speakerId));
    setSelectedId((prev) => {
      if (prev !== speakerId) return prev;
      return speakers.find((item) => item.speaker_id !== speakerId)?.speaker_id ?? null;
    });
    setToast(
      tr(
        `Queued delete: ${s?.current_name ?? speakerId}. Save to apply.`,
        `已准备删除：${s?.current_name ?? speakerId}。保存后生效。`,
      ),
    );
  }

  async function locateSentence(rawValue: string) {
    if (!data) return;
    const locator = parseSentenceLocator(rawValue);
    if (locator == null) {
      setToast(tr("Invalid sentence locator.", "句子定位符无效。"));
      return;
    }
    if (locator.projectId && locator.projectId !== data.project_id) {
      // Jumping to another project remounts this page and destroys staged edits.
      if (
        dirty &&
        !(await confirmDialog({
          message: tr(
            "Discard unsaved speaker review edits and jump to another project?",
            "放弃未保存的 speaker review 改动并跳转到其他项目？",
          ),
          confirmLabel: tr("Discard", "放弃"),
          danger: true,
        }))
      )
        return;
      navigate(
        `/projects/${encodeURIComponent(locator.projectId)}/speakers?sentence=${encodeURIComponent(`${locator.projectId}#${locator.sentenceId}`)}`,
      );
      return;
    }
    const focused = findSentence(data, locator.sentenceId);
    if (!focused) {
      const display = formatSentenceLocator(data.project_id, locator.sentenceId);
      setToast(tr(`Sentence ${display} not found.`, `未找到句子 ${display}。`));
      return;
    }
    setSelectedId(focused.speaker.speaker_id);
    setFilter("all");
    const next = new URLSearchParams(searchParams);
    next.set("sentence", formatSentenceLocator(data.project_id, locator.sentenceId) ?? "");
    next.delete("sid");
    setSearchParams(next);
  }

  function buildSaveBody(): SaveSpeakerReviewBody {
    const mapping: Record<string, string> = {};
    const person_mapping: Record<string, number> = {};
    const person_public_mapping: Record<string, string> = {};
    const new_person_names: Record<string, string> = {};
    const ignored_speaker_ids: number[] = [];
    for (const s of speakers) {
      const name = s.current_name.trim() || s.label;
      const edit = edits.get(s.speaker_id);
      mapping[s.speaker_id] = name;
      if (edit?.create_person && !s.ignored) {
        new_person_names[s.speaker_id] = name;
      } else if (s.person_id != null && !s.ignored) {
        person_mapping[s.speaker_id] = s.person_id;
      }
      if (!edit?.create_person && s.person_public_id && !s.ignored)
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
      review_revision: data!.review_revision,
      mapping,
      person_mapping,
      person_public_mapping,
      new_person_names,
      ignored_speaker_ids,
      reassignments,
      deleted_speaker_ids: [...deletedSpeakerIds],
      correction_edits: [...textEdits.entries()].map(([key, edit]) => ({
        ...edit,
        speaker_id: reassign.get(key) ?? edit.speaker_id,
      })),
    };
  }

  const unresolved = speakers.filter(
    (s) => s.status === "review" || s.status === "conflict" || s.status === "mismatch",
  ).length;
  // Capture needs at least one named, non-ignored speaker; the backend 400s otherwise.
  // Cheaper to disable the entry with an explanation than to bounce off the error page.
  const hasNamedSpeaker = speakers.some(
    (s) => !s.ignored && s.current_name.trim() !== "" && s.current_name !== s.label,
  );
  const mergingSpeaker = merging ? effective(merging) : null;

  return (
    <div className="review">
      <ReviewHeader
        review={data}
        speakerCount={speakers.length}
        unresolved={unresolved}
        dirty={dirty}
        saving={saveMutation.isPending}
        rematching={rematchMutation.isPending}
        canCapture={hasNamedSpeaker}
        view={view}
        onViewChange={setView}
        onSave={() => saveMutation.mutate(buildSaveBody())}
        onDiscard={() => {
          setEdits(new Map());
          setReassign(new Map());
          setTextEdits(new Map());
          setDeletedSpeakerIds(new Set());
          setExtraSpeakers([]);
          setMerging(null);
        }}
        onRematch={() => rematchMutation.mutate()}
        onCapture={() => navigate(`/projects/${ref}/capture`)}
        onCorrect={() => navigate(`/projects/${ref}/corrections`)}
        onExports={() => setShowExports(true)}
      />
      {view === "timeline" ? (
        <TimelinePane
          projectRef={ref}
          projectId={data.project_id}
          speakers={speakers}
          segmentsBySpeaker={segmentsBySpeaker}
          reassignKeys={reassign}
          textEdits={textEdits}
          focusSentenceId={focusedSentenceId}
          onPickSpeaker={(id) => {
            setSelectedId(id);
            setView("speakers");
          }}
          onReassign={(seg) => setReassigning(seg)}
          onEditText={(seg) => setEditingText(seg)}
          canEditText={data.allow_correction}
        />
      ) : (
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
            projectId={data.project_id}
            selected={selected}
            segments={selected ? (segmentsBySpeaker.get(selected.speaker_id) ?? []) : []}
            filter={filter}
            reassignKeys={reassign}
            textEdits={textEdits}
            focusSentenceId={focusedSentenceId}
            onFilter={setFilter}
            onLocateSentence={locateSentence}
            onIdentify={() =>
              selected &&
              setPicking(
                allSpeakers.find((s) => s.speaker_id === selected.speaker_id) ?? null,
              )
            }
            onAccept={() => selected && acceptMatch(selected.speaker_id)}
            onIgnore={() => selected && toggleIgnore(selected.speaker_id)}
            onMerge={() =>
              selected &&
              setMerging(
                allSpeakers.find((s) => s.speaker_id === selected.speaker_id) ?? null,
              )
            }
            onDelete={() => selected && doDeleteSpeaker(selected.speaker_id)}
            canDelete={selected ? canDeleteSpeaker(selected.speaker_id) : false}
            onReassign={(seg) => setReassigning(seg)}
            onEditText={(seg) => setEditingText(seg)}
            canEditText={data.allow_correction}
            quality={selectedQuality}
            dirty={dirty}
            repairing={repairLibraryMutation.isPending}
            onRepairLibrary={(personRef) => repairLibraryMutation.mutate({ personRef })}
          />
        </div>
      )}

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
          sentencePreview={
            textEdits.get(segKey(reassigning))?.corrected_text ?? reassigning.text
          }
          onPick={(target) => doReassign(reassigning, target)}
          onCreate={() => {
            const seg = reassigning;
            setReassigning(null);
            createSpeakerAndReassign(seg);
          }}
          onClose={() => setReassigning(null)}
        />
      )}
      {editingText && (
        <SentenceTextEditor
          segment={editingText}
          edit={textEdits.get(segKey(editingText)) ?? null}
          onApply={(value) => stageTextEdit(editingText, value)}
          onClose={() => setEditingText(null)}
        />
      )}
      {merging && mergingSpeaker && (
        <SpeakerPicker
          title={tr("Merge speaker into…", "合并 speaker 到…")}
          speakers={speakers}
          currentSpeakerId={merging.speaker_id}
          sentencePreview={tr(
            `Move all current sentences from ${mergingSpeaker.current_name || mergingSpeaker.label} to the selected speaker.`,
            `把 ${mergingSpeaker.current_name || mergingSpeaker.label} 当前所有句子移动到选中的 speaker。`,
          )}
          onPick={(target) => doMergeSpeaker(merging.speaker_id, target)}
          onClose={() => setMerging(null)}
        />
      )}
      {showExports && <ExportsModal projectRef={ref} onClose={() => setShowExports(false)} />}
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
  rematching: boolean;
  canCapture: boolean;
  view: "speakers" | "timeline";
  onViewChange: (view: "speakers" | "timeline") => void;
  onSave: () => void;
  onDiscard: () => void;
  onRematch: () => void;
  onCapture: () => void;
  onCorrect: () => void;
  onExports: () => void;
}) {
  const { review, speakerCount, unresolved, dirty, saving, onSave, onDiscard, onCapture, onCorrect } =
    props;
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
        <div className="row gap" style={{ marginTop: 6 }}>
          {(["speakers", "timeline"] as const).map((v) => (
            <button
              key={v}
              className={`chip ${props.view === v ? "on" : ""}`}
              onClick={() => props.onViewChange(v)}
            >
              {v === "speakers" ? tr("By speaker", "按发言人") : tr("Timeline", "时间轴")}
            </button>
          ))}
        </div>
      </div>
      <div className="row gap">
        {/* Exports is read-only, so it stays enabled while dirty. */}
        <button className="btn ghost" onClick={props.onExports}>
          {tr("Exports", "导出")}
        </button>
        {/* Rematch rewrites speaker_matches.json and refetches the review, which resets
            staged edits -- same disable-when-dirty rule as Capture/Correct. */}
        <button
          className="btn ghost"
          onClick={props.onRematch}
          disabled={saving || dirty || props.rematching}
          title={
            dirty
              ? tr("Save changes first", "请先保存改动")
              : tr(
                  "Re-run voiceprint matching against the current library.",
                  "用当前声纹库重跑说话人匹配。",
                )
          }
        >
          {props.rematching ? tr("Rematching…", "匹配中…") : tr("Rematch", "重跑匹配")}
        </button>
        {/* Capture/Correct reload from on-disk speaker_map.json + transcript artifacts, so
            leaving with unsaved edits (dirty) would act on stale/anonymous identities -- e.g.
            "accept match -> Capture" would capture under the old name. Block until saved. */}
        <button
          className="btn ghost"
          onClick={onCorrect}
          disabled={saving || dirty}
          title={dirty ? tr("Save changes first", "请先保存改动") : undefined}
        >
          {tr("Correct text", "文字纠错")}
        </button>
        <button
          className="btn ghost"
          onClick={onCapture}
          disabled={saving || dirty || !props.canCapture}
          title={
            dirty
              ? tr("Save changes first", "请先保存改动")
              : !props.canCapture
                ? tr(
                    "Name at least one speaker first.",
                    "先给至少一位发言人命名。",
                  )
                : undefined
          }
        >
          {tr("Capture voiceprints", "采集声纹")}
        </button>
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
        const label = statusLabel(s.status);
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
              <div
                className="speaker-card-match subtle mono"
                title={tr(
                  "Aggregate voiceprint match for this speaker track.",
                  "当前 speaker 整体声纹匹配分数。",
                )}
              >
                {tr("match", "匹配")} {s.match.best_name}{" "}
                {s.match.best_score != null && (
                  <span className={`score-badge ${identityScoreClass(s.match.best_score)}`}>
                    {s.match.best_score.toFixed(2)}
                  </span>
                )}
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
  projectId: string;
  selected: ReviewSpeaker | null;
  segments: SpeakerSegment[];
  filter: "all" | "review" | "low";
  reassignKeys: Map<string, number>;
  textEdits: Map<string, InlineCorrectionEdit>;
  focusSentenceId: number | null;
  onFilter: (f: "all" | "review" | "low") => void;
  onLocateSentence: (value: string) => void;
  onIdentify: () => void;
  onAccept: () => void;
  onIgnore: () => void;
  onMerge: () => void;
  onDelete: () => void;
  canDelete: boolean;
  onReassign: (seg: SpeakerSegment) => void;
  onEditText: (seg: SpeakerSegment) => void;
  canEditText: boolean;
  quality: QualityPerson | undefined;
  dirty: boolean;
  repairing: boolean;
  onRepairLibrary: (personRef: string) => void;
}) {
  const {
    projectRef,
    projectId,
    selected,
    segments,
    filter,
    reassignKeys,
    textEdits,
    focusSentenceId,
    quality,
  } = props;
  const audioRef = useRef<HTMLAudioElement>(null);
  const segmentRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [jumpValue, setJumpValue] = useState("");
  const [progress, setProgress] = useState(0);
  const [identityPopover, setIdentityPopover] = useState<{
    key: string;
    top: number;
    scoreClass: "ok" | "mid" | "low";
    evidence: string | null;
    suggestion: string | null;
    title: string;
  } | null>(null);

  useEffect(() => {
    if (focusSentenceId != null) {
      setJumpValue(formatSentenceLocator(projectId, focusSentenceId) ?? "");
    }
  }, [focusSentenceId, projectId]);

  useEffect(() => {
    setIdentityPopover(null);
  }, [filter, selected?.speaker_id]);

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
    // A failed load (404/401 clip) never fires onended; reset so the button isn't stuck on ⏸.
    el.play().catch(() => setPlayingKey((prev) => (prev === key ? null : prev)));
    setPlayingKey(key);
    setProgress(0);
  };

  const filtered = useMemo(
    () =>
      segments.filter((seg) => {
        if (filter === "all") return true;
        if (filter === "review") {
          return (
            (seg.score_status != null && seg.score_status !== "identity-ok") ||
            (seg.score != null && seg.score < 0.6)
          );
        }
        return (
          (seg.score != null && seg.score < 0.45) || seg.score_status === "identity-weak"
        );
      }),
    [segments, filter],
  );
  const focusedKey =
    focusSentenceId == null
      ? null
      : (filtered.find((seg) => seg.sentence_id === focusSentenceId) ?? null);
  const focusedSegKey = focusedKey ? segKey(focusedKey) : null;

  useEffect(() => {
    if (!focusedSegKey) return;
    const frame = window.requestAnimationFrame(() => {
      segmentRefs.current.get(focusedSegKey)?.scrollIntoView({
        block: "center",
        behavior: "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [focusedSegKey]);

  // Stop playback when the playing clip leaves the visible list (speaker switch,
  // filter change, or the sentence was reassigned away): its pause button is gone,
  // so the audio would keep playing with no way to stop it.
  useEffect(() => {
    if (playingKey == null) return;
    if (!filtered.some((seg) => segKey(seg) === playingKey)) {
      audioRef.current?.pause();
      setPlayingKey(null);
    }
  }, [filtered, playingKey]);

  // Fresh speaker/filter without an explicit focus target starts at the top --
  // the pane used to keep the previous speaker's scroll offset.
  const segmentsBoxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!focusedSegKey) segmentsBoxRef.current?.scrollTo({ top: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.speaker_id, filter]);

  if (!selected) return <div className="placeholder">{tr("Select a speaker.", "选择一位发言人。")}</div>;

  const canAccept = !!selected.match?.best_name && (selected.match?.best_score ?? 0) > 0;

  const showIdentityPopover = (
    key: string,
    row: HTMLElement,
    scoreClass: "ok" | "mid" | "low",
    evidence: string | null,
    suggestion: string | null,
    title: string,
  ) => {
    if (!evidence && !suggestion) return;
    const rect = row.getBoundingClientRect();
    const top = Math.min(
      Math.max(rect.top - 2, 72),
      Math.max(72, window.innerHeight - 220),
    );
    setIdentityPopover({ key, top, scoreClass, evidence, suggestion, title });
  };

  const hideIdentityPopover = (key: string) => {
    setIdentityPopover((current) => (current?.key === key ? null : current));
  };

  return (
    <div className="transcript-pane">
      <div className="transcript-head">
        <div className="row gap center">
          <span className={`status-dot status-${selected.status}`} />
          <h2>{selected.current_name || selected.label}</h2>
          <span className="badge">{statusLabel(selected.status)}</span>
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
          <button className="btn ghost" onClick={props.onMerge}>
            {tr("Merge into", "合并到")} <span className="kbd">M</span>
          </button>
          <button
            className="btn ghost danger"
            onClick={props.onDelete}
            disabled={!props.canDelete}
            title={
              props.canDelete
                ? tr("Delete this empty speaker.", "删除这个空 speaker。")
                : tr(
                    "Only speakers with no non-empty sentences can be deleted.",
                    "只有没有非空句子的 speaker 才能删除。",
                  )
            }
          >
            {tr("Delete", "删除")}
          </button>
          <button className={`btn ghost ${selected.ignored ? "on" : ""}`} onClick={props.onIgnore}>
            {selected.ignored ? tr("Ignored", "已忽略") : tr("Ignore", "忽略")}{" "}
            <span className="kbd">I</span>
          </button>
        </div>
      </div>

      <div className="filter-bar">
        <form
          className="sentence-jump"
          onSubmit={(e) => {
            e.preventDefault();
            props.onLocateSentence(jumpValue);
          }}
        >
          <input
            value={jumpValue}
            onChange={(e) => setJumpValue(e.currentTarget.value)}
            placeholder={tr("project#sentence", "项目#句子")}
          />
          <button className="chip" type="submit">
            {tr("Go", "定位")}
          </button>
        </form>
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

      {quality && (
        <SpeakerQualityRepairPanel
          quality={quality}
          matchScore={selected.match?.best_score ?? null}
          dirty={props.dirty}
          repairing={props.repairing}
          onRepair={() => props.onRepairLibrary(quality.public_id)}
        />
      )}

      <audio
        ref={audioRef}
        onTimeUpdate={(e) => {
          const el = e.currentTarget;
          if (el.duration) setProgress(el.currentTime / el.duration);
        }}
        onEnded={() => setPlayingKey(null)}
      />

      <div className="segments" ref={segmentsBoxRef} onScroll={() => setIdentityPopover(null)}>
        {filtered.map((seg) => {
          const key = segKey(seg);
          const reassigned = reassignKeys.has(key);
          const textEdit = textEdits.get(key) ?? null;
          const playing = playingKey === key;
          const focused = focusSentenceId != null && seg.sentence_id === focusSentenceId;
          const scoreStatusLabel =
            seg.score_status && seg.score_status !== "identity-ok"
              ? identityStatusLabel(seg.score_status)
              : null;
          const scoreTitle =
            seg.score != null || scoreStatusLabel ? identityScoreTitle(seg) : undefined;
          const scoreReason = identityScoreReason(seg);
          const scoreEvidence = identityDiagnosticEvidence(seg);
          const scoreSuggestion = identityDiagnosticSuggestion(seg);
          const hasIdentityDetail = Boolean(scoreEvidence || scoreSuggestion);
          const scoreClass = identityScoreClass(seg.score ?? 1, seg.score_status);
          const scoreBadgeText =
            seg.score != null
              ? `${tr("voice", "声纹匹配")} ${seg.score.toFixed(2)}${scoreReason ? ` ${scoreReason}` : ""}`
              : `${tr("voice", "声纹")} ${scoreStatusLabel}`;
          const sentenceRef =
            seg.sentence_ref ?? formatSentenceLocator(projectId, seg.sentence_id);
          const displayText = textEdit?.corrected_text ?? seg.text;
          return (
            <div
              key={key}
              ref={(node) => {
                if (node) segmentRefs.current.set(key, node);
                else segmentRefs.current.delete(key);
              }}
              className={`segment ${playing ? "playing" : ""} ${reassigned ? "reassigned" : ""} ${focused ? "focused" : ""} ${hasIdentityDetail ? "has-identity-detail" : ""}`}
              data-sentence-id={seg.sentence_id ?? undefined}
              onMouseEnter={(event) =>
                showIdentityPopover(
                  key,
                  event.currentTarget,
                  scoreClass,
                  scoreEvidence,
                  scoreSuggestion,
                  [scoreEvidence, scoreSuggestion].filter(Boolean).join(" "),
                )
              }
              onMouseLeave={() => hideIdentityPopover(key)}
              onFocus={(event) =>
                showIdentityPopover(
                  key,
                  event.currentTarget,
                  scoreClass,
                  scoreEvidence,
                  scoreSuggestion,
                  [scoreEvidence, scoreSuggestion].filter(Boolean).join(" "),
                )
              }
              onBlur={(event) => {
                const nextTarget = event.relatedTarget;
                if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
                  hideIdentityPopover(key);
                }
              }}
            >
              <button className="play-btn" onClick={() => play(seg)} aria-label="play">
                {playing ? "⏸" : "▶"}
              </button>
              <div className="segment-body">
                <div className="segment-meta subtle mono">
                  {sentenceRef != null && (
                    <span className="sentence-id" title={tr("Sentence locator", "句子定位符")}>
                      {sentenceRef}
                    </span>
                  )}
                  {fmtMs(seg.begin_time_ms)}
                  {(seg.score != null || scoreStatusLabel) && (
                    <span
                      className={`score-badge ${scoreClass}`}
                      title={scoreTitle}
                      aria-label={scoreTitle}
                    >
                      {scoreBadgeText}
                    </span>
                  )}
                  {reassigned && <span className="badge reassigned-badge">{tr("reassigned", "已重指派")}</span>}
                  {textEdit && (
                    <span className="badge text-edited-badge">{tr("edited", "已编辑")}</span>
                  )}
                </div>
                <div className="segment-text">{displayText}</div>
                {playing && (
                  <div className="seg-progress">
                    <div className="seg-progress-bar" style={{ width: `${progress * 100}%` }} />
                  </div>
                )}
              </div>
              <div className="segment-actions">
                {props.canEditText && (
                  <button
                    className="segment-action-btn text-edit-btn"
                    onClick={() => props.onEditText(seg)}
                    title={tr("Edit this sentence text.", "编辑这句话的文字。")}
                    aria-label={tr("Edit text", "编辑文字")}
                  >
                    ✎
                  </button>
                )}
                <button
                  className="segment-action-btn reassign-btn"
                  onClick={() => props.onReassign(seg)}
                  title={tr(
                    "Change this sentence's speaker assignment.",
                    "修改这句话的 speaker 归属。",
                  )}
                  aria-label={tr("Change owner", "修改归属")}
                >
                  ⇄ {tr("Owner", "归属")}
                </button>
              </div>
            </div>
          );
        })}
      </div>
      {identityPopover && (
        <div
          className={`identity-popover ${identityPopover.scoreClass}`}
          style={{ top: identityPopover.top }}
          title={identityPopover.title}
        >
          {identityPopover.evidence && (
            <div className="identity-evidence">{identityPopover.evidence}</div>
          )}
          {identityPopover.suggestion && (
            <div className="identity-suggestion">
              <span className="identity-suggestion-label">
                {tr("Suggestion", "建议")}
              </span>
              <span>
                {tr(": ", "：")}
                {identityPopover.suggestion}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Chronological view across ALL speakers: rebuilds the meeting's flow so cross-speaker
 *  context (who replied to whom) is visible while reassigning/correcting. */
function TimelinePane(props: {
  projectRef: string;
  projectId: string;
  speakers: ReviewSpeaker[];
  segmentsBySpeaker: Map<number, SpeakerSegment[]>;
  reassignKeys: Map<string, number>;
  textEdits: Map<string, InlineCorrectionEdit>;
  focusSentenceId: number | null;
  onPickSpeaker: (speakerId: number) => void;
  onReassign: (seg: SpeakerSegment) => void;
  onEditText: (seg: SpeakerSegment) => void;
  canEditText: boolean;
}) {
  const {
    projectRef,
    projectId,
    speakers,
    segmentsBySpeaker,
    reassignKeys,
    textEdits,
    focusSentenceId,
  } = props;
  const audio = useClipAudio();
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const rows = useMemo(() => {
    const merged: { seg: SpeakerSegment; owner: ReviewSpeaker }[] = [];
    for (const owner of speakers) {
      for (const seg of segmentsBySpeaker.get(owner.speaker_id) ?? []) {
        merged.push({ seg, owner });
      }
    }
    merged.sort((a, b) => a.seg.begin_time_ms - b.seg.begin_time_ms);
    return merged;
  }, [speakers, segmentsBySpeaker]);

  const focusedRow =
    focusSentenceId == null
      ? null
      : (rows.find((row) => row.seg.sentence_id === focusSentenceId) ?? null);
  const focusedRowKey = focusedRow ? segKey(focusedRow.seg) : null;
  useEffect(() => {
    if (!focusedRowKey) return;
    const frame = window.requestAnimationFrame(() => {
      rowRefs.current.get(focusedRowKey)?.scrollIntoView({
        block: "center",
        behavior: "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [focusedRowKey]);

  return (
    // Reuses .transcript-pane's layout/scroll rules; timeline-specific bits layer on top.
    <div className="transcript-pane timeline-pane">
      <div className="segments">
        {rows.map(({ seg, owner }) => {
          const key = segKey(seg);
          const playing = audio.playingKey === key;
          const reassigned = reassignKeys.has(key);
          const textEdit = textEdits.get(key) ?? null;
          const focused = focusSentenceId != null && seg.sentence_id === focusSentenceId;
          const displayText = textEdit?.corrected_text ?? seg.text;
          const sentenceRef =
            seg.sentence_ref ?? formatSentenceLocator(projectId, seg.sentence_id);
          return (
            <div
              key={key}
              ref={(node) => {
                if (node) rowRefs.current.set(key, node);
                else rowRefs.current.delete(key);
              }}
              className={`segment ${playing ? "playing" : ""} ${reassigned ? "reassigned" : ""} ${focused ? "focused" : ""}`}
              data-sentence-id={seg.sentence_id ?? undefined}
            >
              <button
                className="play-btn"
                onClick={() =>
                  audio.toggle(key, clipUrl(projectRef, seg.begin_time_ms, seg.end_time_ms))
                }
                aria-label="play"
              >
                {playing ? "⏸" : "▶"}
              </button>
              <div className="segment-body">
                <div className="segment-meta subtle mono">
                  <button
                    className="chip timeline-speaker"
                    onClick={() => props.onPickSpeaker(owner.speaker_id)}
                    title={tr("Open this speaker's pane", "打开该发言人")}
                  >
                    <span className={`status-dot status-${owner.status}`} />
                    {owner.current_name || owner.label}
                  </button>
                  {fmtMs(seg.begin_time_ms)}
                  {sentenceRef != null && <span className="sentence-id">{sentenceRef}</span>}
                  {reassigned && (
                    <span className="badge reassigned-badge">{tr("reassigned", "已重指派")}</span>
                  )}
                  {textEdit && (
                    <span className="badge text-edited-badge">{tr("edited", "已编辑")}</span>
                  )}
                </div>
                <div className="segment-text">{displayText}</div>
                {playing && (
                  <div className="seg-progress">
                    <div className="seg-progress-bar" style={{ width: `${audio.progress * 100}%` }} />
                  </div>
                )}
              </div>
              <div className="segment-actions">
                {props.canEditText && (
                  <button
                    className="segment-action-btn text-edit-btn"
                    onClick={() => props.onEditText(seg)}
                    title={tr("Edit this sentence text.", "编辑这句话的文字。")}
                    aria-label={tr("Edit text", "编辑文字")}
                  >
                    ✎
                  </button>
                )}
                <button
                  className="segment-action-btn reassign-btn"
                  onClick={() => props.onReassign(seg)}
                  title={tr(
                    "Change this sentence's speaker assignment.",
                    "修改这句话的 speaker 归属。",
                  )}
                  aria-label={tr("Change owner", "修改归属")}
                >
                  ⇄ {tr("Owner", "归属")}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SentenceTextEditor(props: {
  segment: SpeakerSegment;
  edit: InlineCorrectionEdit | null;
  onApply: (value: string) => void;
  onClose: () => void;
}) {
  const { segment, edit, onApply, onClose } = props;
  const originalText = edit?.original_text ?? segment.text.trim();
  const [value, setValue] = useState(edit?.corrected_text ?? segment.text);
  const trimmed = value.trim();
  const canApply = trimmed.length > 0 && trimmed !== originalText;
  // Dirty vs the LOADED baseline (a staged edit re-opened unchanged closes freely);
  // Esc / backdrop / Cancel would otherwise silently drop un-staged typing.
  const loadedBaseline = (edit?.corrected_text ?? segment.text).trim();
  const guardedClose = () => {
    if (trimmed === loadedBaseline) {
      onClose();
      return;
    }
    void confirmDialog({
      message: tr(
        "Discard this un-staged text edit?",
        "放弃这条尚未暂存的文字修改？",
      ),
      confirmLabel: tr("Discard", "放弃"),
      danger: true,
    }).then((ok) => {
      if (ok) onClose();
    });
  };
  const sentenceLabel =
    segment.sentence_ref ??
    (segment.sentence_id == null
      ? tr("no sentence id", "无句子 ID")
      : String(segment.sentence_id));
  return (
    <Modal
      title={tr("Edit transcript text", "编辑转写文本")}
      onClose={guardedClose}
      footer={
        <div className="row gap">
          <button className="btn ghost" onClick={guardedClose}>
            {tr("Cancel", "取消")}
          </button>
          <button
            className="btn"
            disabled={!edit && trimmed === originalText}
            onClick={() => onApply(originalText)}
            title={tr("Revert this staged edit.", "撤销这条暂存文字修改。")}
          >
            {tr("Revert", "恢复原文")}
          </button>
          <button
            className="btn primary"
            disabled={!canApply}
            onClick={() => onApply(value)}
          >
            {tr("Stage edit", "暂存修改")}
          </button>
        </div>
      }
    >
      <div className="segment-editor-meta subtle mono">
        {sentenceLabel} · {fmtMs(segment.begin_time_ms)}
      </div>
      <div className="sentence-preview">
        <div className="subtle">{tr("Original", "原文")}</div>
        <div>{originalText}</div>
      </div>
      <textarea
        className="text-edit-area"
        autoFocus
        value={value}
        onChange={(e) => setValue(e.currentTarget.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && canApply) {
            e.preventDefault();
            onApply(value);
          }
        }}
      />
      <div className="subtle">
        {tr("Cmd/Ctrl+Enter stages this edit.", "Cmd/Ctrl+Enter 暂存修改。")}
      </div>
    </Modal>
  );
}

function SpeakerQualityRepairPanel(props: {
  quality: QualityPerson;
  matchScore: number | null;
  dirty: boolean;
  repairing: boolean;
  onRepair: () => void;
}) {
  const { quality, matchScore, dirty, repairing } = props;
  const riskyProjects = quality.projects.filter((project) => project.suspicious_count > 0);
  const hasRisk = quality.suspicious_count > 0 || quality.critical_count > 0;
  const nearThreshold = matchScore != null && matchScore < 0.82;
  if (!hasRisk && !nearThreshold) return null;
  return (
    <div className={`speaker-quality-panel ${hasRisk ? "risk" : ""}`}>
      <div className="speaker-quality-main">
        <div className="speaker-quality-title">
          {tr("Voiceprint library diagnosis", "声纹库诊断")} · {quality.name}
        </div>
        <div className="speaker-quality-line subtle">
          {tr("matching", "参与匹配")} {quality.active_sample_count}/{quality.sample_count} ·{" "}
          {tr("mean", "均值")} {quality.mean_score?.toFixed(2) ?? "—"} ·{" "}
          <span className={quality.critical_count > 0 ? "danger-text" : hasRisk ? "warn" : ""}>
            {quality.suspicious_count} {tr("issues", "疑点")} / {quality.critical_count}{" "}
            {tr("critical", "严重")}
          </span>
        </div>
        {riskyProjects.length > 0 && (
          <div className="speaker-quality-line">
            {riskyProjects.slice(0, 3).map((project) => (
              <span key={project.project_id} className="badge vp-project-risk">
                {project.project_id} · {project.suspicious_count} {tr("issues", "疑点")} ·{" "}
                {project.min_score?.toFixed(2) ?? "—"}
              </span>
            ))}
          </div>
        )}
        {quality.closest_people.length > 0 && (
          <div className="speaker-quality-line subtle">
            {tr("Closest others", "相近人物")}：
            {quality.closest_people.map((person) => (
              <span key={person.public_id} className="mono">
                {person.name} {person.score.toFixed(2)}
              </span>
            ))}
          </div>
        )}
      </div>
      {hasRisk && (
        // The rematch invalidates the review query, which resets all staged edits;
        // disable while dirty (same rule as Capture/Correct) instead of confirming —
        // there is no way to run it without losing the staged work.
        <button
          className="btn ghost danger"
          disabled={repairing || dirty}
          title={dirty ? tr("Save changes first", "请先保存改动") : undefined}
          onClick={props.onRepair}
        >
          {repairing
            ? tr("Repairing…", "修复中…")
            : tr("Exclude issues + rematch", "排除疑点并重跑匹配")}
        </button>
      )}
    </div>
  );
}
