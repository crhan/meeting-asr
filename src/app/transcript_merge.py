"""Merge several single-meeting projects into one transcript package.

A meeting that is paused for a break is often split by DingTalk flash-record
into N segments, each landing as an independent ``meeting-asr`` project. This
module rebuilds those segments into a single time-ordered transcript whose
speakers are unified *across* segments.

The unification is the hard part. ``speaker_id`` is per-project (project A's
"Speaker 1" is unrelated to project B's), but ``speakers/speaker_person_map``
maps each local ``speaker_id`` to a stable voiceprint person public id
(``vpp-...``). The same ``vpp`` across projects is the same human, regardless of
local id or even of whether that segment bothered to name them. So we key global
identity on ``vpp`` first, fall back to the human-assigned display name, and only
then to a per-segment anonymous label that is *never* merged across segments.

Design notes (see AGENTS.md "Project Merge Notes"):

* Merge is a stateless pure function. It reads existing project artifacts and
  writes a self-contained output package; it never writes back into any project
  and ``merge.json`` is a read-only manifest of *this* output, not session state.
* We rebuild from ``asr/sentences.json`` and re-render with the existing
  renderers, rather than concatenating the pre-rendered ``exports/*.txt``.
  Concatenating text cannot re-attribute speakers, so cross-segment unification
  would be impossible.
* The merged timeline is *concatenated*: segment k's timestamps are shifted by
  the running sum of earlier segments' audio durations, so the stream is
  monotonic and non-overlapping. Real wall-clock (``meeting_time``) is preserved
  in the per-segment headers and ``merge.json`` but kept out of the timeline,
  because a break would otherwise punch a multi-minute hole into the subtitle.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.project_models import ProjectManifest
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.project_manager import load_manifest, project_paths
from app.speaker_labeling import (
    load_project_ignored_speakers,
    load_speaker_person_mapping,
    load_transcript_result,
    render_named_speaker_text,
    render_named_srt,
)
from app.utils import safe_write_json, safe_write_text
from app.voiceprint_people import get_voiceprint_person
from app.voiceprint_store import (
    get_default_voiceprint_db_path,
    get_voiceprint_db_path,
)

SCHEMA_VERSION = 1
CHINA_TZ = timezone(timedelta(hours=8))
SEGMENT_DIVIDER = "─" * 60

# Names the UI saves for speakers a reviewer deliberately left unidentified.
# These must not be treated as a real cross-segment identity.
_PLACEHOLDER_NAME_RE = re.compile(
    r"^(待确认发言人|待确认|未知发言人|未知|待定|发言人|说话人|speaker)\s*\d*$",
    re.IGNORECASE,
)
# Narrow / non-breaking spaces a macOS Chinese IME can inject into names.
_INVISIBLE_SPACES = ("\u202f", "\u00a0", "\u2009", "\u2060", "\ufeff")


# ---------------------------------------------------------------------------
# Loaded segment
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MergeSegment:
    """One project loaded as a merge input.

    Holds the normalized transcript (plus the corrected variant when present)
    and the per-project speaker metadata needed to resolve global identities.
    """

    project_dir: Path
    project_id: str
    title: str
    meeting_time: str | None
    created_at: str | None
    duration_seconds: float | None
    source_filename: str | None
    result: TranscriptResult
    result_corrected: TranscriptResult | None
    speaker_map: dict[int, str]
    person_map: dict[int, str]
    ignored: set[int]

    def local_speaker_ids(self) -> set[int]:
        """Return every speaker id that appears in raw or corrected sentences."""
        ids: set[int] = set()
        for source in (self.result, self.result_corrected):
            if source is None:
                continue
            for sentence in source.sentences:
                if sentence.speaker_id is not None:
                    ids.add(int(sentence.speaker_id))
        # Named/voiceprinted speakers may have been fully filtered out of the
        # visible sentences yet still carry an identity worth surfacing.
        ids.update(self.speaker_map)
        ids.update(self.person_map)
        return ids


def _load_speaker_name_map(path: Path) -> dict[int, str]:
    """Load ``speaker_map.json`` as ``{speaker_id: display_name}``."""
    if not path.exists():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        mapping[int(key)] = str(value)
    return mapping


def load_merge_segment(
    project_dir: Path, *, include_low_information: bool = False
) -> MergeSegment:
    """
    Load one project into a :class:`MergeSegment`.

    Args:
        project_dir: Resolved project root.
        include_low_information: Keep backchannel-only speaker tracks instead of
            applying the default low-information filter. Must be uniform across
            segments so the same person is not trimmed by different standards.

    Returns:
        Loaded merge segment.
    """
    manifest: ProjectManifest = load_manifest(project_dir)
    paths = project_paths(project_dir)
    result = load_transcript_result(
        paths.asr_dir / "sentences.json",
        include_low_information=include_low_information,
    )
    corrected_path = paths.asr_dir / "sentences_corrected.json"
    result_corrected = (
        load_transcript_result(
            corrected_path, include_low_information=include_low_information
        )
        if corrected_path.exists()
        else None
    )
    speaker_map = _load_speaker_name_map(paths.speakers_dir / "speaker_map.json")
    person_map = {
        int(key): str(value)
        for key, value in load_speaker_person_mapping(
            paths.speakers_dir / "speaker_person_map.json"
        ).items()
        if value is not None
    }
    ignored = load_project_ignored_speakers(project_dir)
    duration = manifest.audio.get("duration_seconds")
    return MergeSegment(
        project_dir=project_dir,
        project_id=manifest.project_id,
        title=manifest.title,
        meeting_time=manifest.source.meeting_time,
        created_at=manifest.created_at,
        duration_seconds=float(duration) if duration is not None else None,
        source_filename=manifest.source.filename,
        result=result,
        result_corrected=result_corrected,
        speaker_map=speaker_map,
        person_map=person_map,
        ignored=ignored,
    )


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def name_fold(name: str) -> str:
    """
    Normalize a display name into a stable comparison key.

    NFKC + strip + collapse internal whitespace + drop IME-injected invisible
    spaces, then casefold the ASCII parts (Chinese is unaffected). Parenthetical
    aliases such as ``张辉洲(尺木)`` are kept whole on purpose.

    Args:
        name: Raw display name.

    Returns:
        Folded comparison key.
    """
    folded = unicodedata.normalize("NFKC", name)
    for space in _INVISIBLE_SPACES:
        folded = folded.replace(space, " ")
    folded = " ".join(folded.split())
    return folded.casefold()


def is_placeholder_name(name: str | None) -> bool:
    """Return whether a saved name is an unidentified-speaker placeholder."""
    if name is None:
        return True
    stripped = name.strip()
    if not stripped:
        return True
    return bool(_PLACEHOLDER_NAME_RE.match(stripped))


# ---------------------------------------------------------------------------
# Global speaker identity
# ---------------------------------------------------------------------------

# An identity key is one of:
#   ("vpp", "vpp-xxxx")          stable voiceprint person -- merges across segments
#   ("name", folded_name)        named but not voiceprinted -- merges by name
#   ("anon", segment_index, id)  unidentified -- never merges across segments
IdentityKey = tuple


@dataclass(slots=True)
class GlobalIdentityTable:
    """Result of folding per-segment speakers into global identities."""

    assignment: dict[tuple[int, int], int]
    mapping: dict[int, str]
    identities: list[dict]
    warnings: list[str] = field(default_factory=list)


def _raw_identity_key(
    segment_index: int,
    local_id: int,
    segment: MergeSegment,
) -> IdentityKey:
    """Resolve the pre-promotion identity key for one local speaker."""
    if local_id in segment.ignored:
        # Deliberately kept anonymous: never attribute, even with a voiceprint.
        return ("anon", segment_index, local_id)
    vpp = segment.person_map.get(local_id)
    if vpp:
        return ("vpp", vpp)
    name = segment.speaker_map.get(local_id)
    if name and not is_placeholder_name(name):
        return ("name", name_fold(name))
    return ("anon", segment_index, local_id)


def build_global_identities(
    segments: Sequence[MergeSegment],
    *,
    name_to_vpp: bool = True,
    vpp_name_resolver: Callable[[str], str | None] | None = None,
) -> GlobalIdentityTable:
    """
    Fold per-segment speakers into a single global identity space.

    Args:
        segments: Segments already in final timeline order.
        name_to_vpp: Promote name-only speakers onto a voiceprint identity when
            their name matches a voiceprint person's authoritative name.
        vpp_name_resolver: Maps a ``vpp`` id to its authoritative display name
            (typically the voiceprint store). ``None`` means resolve from the
            segments alone.

    Returns:
        Assignment, render mapping, audit trail and warnings.
    """
    warnings: list[str] = []

    # Pass 1: raw identity key and original name per (segment, local speaker).
    raw_key: dict[tuple[int, int], IdentityKey] = {}
    local_name: dict[tuple[int, int], str | None] = {}
    for index, segment in enumerate(segments):
        for local_id in segment.local_speaker_ids():
            cell = (index, local_id)
            raw_key[cell] = _raw_identity_key(index, local_id, segment)
            local_name[cell] = segment.speaker_map.get(local_id)

    # Pass 2: authoritative name per vpp, and the name -> vpp promotion table.
    vpp_members: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for cell, key in raw_key.items():
        if key[0] == "vpp":
            vpp_members[key[1]].append(cell)
    vpp_display: dict[str, str | None] = {}
    for vpp, members in vpp_members.items():
        resolved = vpp_name_resolver(vpp) if vpp_name_resolver else None
        if not resolved:
            for cell in sorted(members):
                candidate = local_name.get(cell)
                if candidate and not is_placeholder_name(candidate):
                    resolved = candidate
                    break
        vpp_display[vpp] = resolved

    fold_to_vpps: dict[str, set[str]] = defaultdict(set)
    if name_to_vpp:
        for vpp, name in vpp_display.items():
            if name:
                fold_to_vpps[name_fold(name)].add(vpp)
    # A folded name shared by two distinct voiceprint people is ambiguous; do
    # not promote it (and warn) rather than fold two humans together.
    promote_fold_to_vpp: dict[str, str] = {}
    for fold, vpps in fold_to_vpps.items():
        if len(vpps) == 1:
            promote_fold_to_vpp[fold] = next(iter(vpps))
        else:
            warnings.append(
                f"显示名 {fold!r} 同时匹配多个声纹人 {sorted(vpps)}，跳过 name→vpp 提升"
            )

    # Pass 3: canonical key (apply promotion) per cell.
    promoted: set[tuple[int, int]] = set()
    canonical: dict[tuple[int, int], IdentityKey] = {}
    for cell, key in raw_key.items():
        if name_to_vpp and key[0] == "name" and key[1] in promote_fold_to_vpp:
            canonical[cell] = ("vpp", promote_fold_to_vpp[key[1]])
            promoted.add(cell)
        else:
            canonical[cell] = key

    # Pass 4: assign global ids in first-appearance order over the timeline.
    key_to_global: dict[IdentityKey, int] = {}
    first_original_name: dict[IdentityKey, str] = {}

    def _touch(key: IdentityKey, original: str | None) -> None:
        if key not in key_to_global:
            key_to_global[key] = len(key_to_global)
        if (
            key not in first_original_name
            and original
            and not is_placeholder_name(original)
        ):
            first_original_name[key] = original

    for index, segment in enumerate(segments):
        for sentence in segment.result.sentences:
            if sentence.speaker_id is None:
                continue
            cell = (index, int(sentence.speaker_id))
            key = canonical.get(cell)
            if key is None:
                continue
            _touch(key, local_name.get(cell))
    # Speakers present only in metadata (filtered out of sentences) still need a
    # stable id so the audit trail and mapping stay complete.
    for cell, key in canonical.items():
        _touch(key, local_name.get(cell))

    assignment = {cell: key_to_global[key] for cell, key in canonical.items()}
    global_to_key = {value: key for key, value in key_to_global.items()}

    # Build render mapping. Every global id is mapped so the renderer never
    # silently falls back (which would turn an unmapped person into "Speaker X").
    mapping: dict[int, str] = {}
    anon_label_index = 0
    for global_id in range(len(key_to_global)):
        key = global_to_key[global_id]
        if key[0] == "vpp":
            name = vpp_display.get(key[1])
            if name:
                mapping[global_id] = name
            else:
                mapping[global_id] = speaker_id_to_label(anon_label_index)
                anon_label_index += 1
        elif key[0] == "name":
            mapping[global_id] = first_original_name.get(key, key[1])
        else:
            mapping[global_id] = speaker_id_to_label(anon_label_index)
            anon_label_index += 1

    identities = _build_identity_audit(
        segments=segments,
        canonical=canonical,
        assignment=assignment,
        global_to_key=global_to_key,
        mapping=mapping,
        vpp_display=vpp_display,
        promoted=promoted,
        local_name=local_name,
    )
    return GlobalIdentityTable(assignment, mapping, identities, warnings)


def _build_identity_audit(
    *,
    segments: Sequence[MergeSegment],
    canonical: dict[tuple[int, int], IdentityKey],
    assignment: dict[tuple[int, int], int],
    global_to_key: dict[int, IdentityKey],
    mapping: dict[int, str],
    vpp_display: dict[str, str | None],
    promoted: set[tuple[int, int]],
    local_name: dict[tuple[int, int], str | None],
) -> list[dict]:
    """Build the per-identity audit trail emitted into ``merge.json``."""
    members: dict[int, list[dict]] = defaultdict(list)
    sentence_counts: dict[int, int] = defaultdict(int)
    name_variants: dict[int, set[str]] = defaultdict(set)
    promoted_globals: set[int] = set()

    for cell, global_id in assignment.items():
        index, local_id = cell
        name = local_name.get(cell)
        members[global_id].append(
            {
                "order": index,
                "project_id": segments[index].project_id,
                "local_speaker_id": local_id,
                "local_name": name,
            }
        )
        if cell in promoted:
            promoted_globals.add(global_id)
        if name and not is_placeholder_name(name):
            name_variants[global_id].add(name)

    for index, segment in enumerate(segments):
        for sentence in segment.result.sentences:
            if sentence.speaker_id is None:
                continue
            global_id = assignment.get((index, int(sentence.speaker_id)))
            if global_id is not None:
                sentence_counts[global_id] += 1

    identities: list[dict] = []
    for global_id in range(len(global_to_key)):
        key = global_to_key[global_id]
        entry: dict = {
            "global_id": global_id,
            "display_name": mapping[global_id],
            "identity_kind": key[0],
            "sentence_count": sentence_counts.get(global_id, 0),
            "members": sorted(members[global_id], key=lambda m: m["order"]),
        }
        if key[0] == "vpp":
            entry["vpp"] = key[1]
        if global_id in promoted_globals:
            entry["promoted_from_name"] = True
        variants = name_variants.get(global_id, set())
        if len(variants) > 1:
            entry["name_conflicts"] = sorted(variants)
        identities.append(entry)
    return identities


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SegmentMeta:
    """Per-segment metadata recorded in the merged package."""

    order: int
    part: str
    project_id: str
    project_dir: str
    title: str
    meeting_time: str | None
    duration_seconds: float | None
    duration_source: str
    clock_offset_ms: int
    sentence_count: int
    ignored_speaker_count: int
    corrected: bool


def _offset_sentences(
    sentences: Iterable[SentenceSegment],
    *,
    segment_index: int,
    offset_ms: int,
    assignment: dict[tuple[int, int], int],
) -> list[SentenceSegment]:
    """Return time-shifted, globally-relabeled copies of ``sentences``."""
    shifted: list[SentenceSegment] = []
    for sentence in sentences:
        global_id = (
            assignment.get((segment_index, int(sentence.speaker_id)))
            if sentence.speaker_id is not None
            else None
        )
        shifted.append(
            SentenceSegment(
                begin_time_ms=sentence.begin_time_ms + offset_ms,
                end_time_ms=sentence.end_time_ms + offset_ms,
                text=sentence.text,
                speaker_id=global_id if global_id is not None else sentence.speaker_id,
                sentence_id=None,
            )
        )
    return shifted


def _segment_duration_ms(
    segment: MergeSegment, sentences: Sequence[SentenceSegment]
) -> tuple[int, str]:
    """Return the timeline advance for a segment and its provenance."""
    if segment.duration_seconds:
        return int(round(segment.duration_seconds * 1000)), "audio"
    max_end = max((s.end_time_ms for s in sentences), default=0)
    return int(max_end), "inferred"


def build_merged_result(
    segments: Sequence[MergeSegment],
    assignment: dict[tuple[int, int], int],
    *,
    corrected: bool,
) -> tuple[TranscriptResult, list[SegmentMeta]]:
    """
    Concatenate segments into one transcript on a continuous timeline.

    Args:
        segments: Segments in final order.
        assignment: ``(segment_index, local_id) -> global_id`` mapping.
        corrected: Use the corrected sentence variant when available.

    Returns:
        The merged transcript and per-segment metadata.
    """
    merged: list[SentenceSegment] = []
    metas: list[SegmentMeta] = []
    offset_ms = 0
    for index, segment in enumerate(segments):
        use_corrected = corrected and segment.result_corrected is not None
        source = segment.result_corrected if use_corrected else segment.result
        shifted = _offset_sentences(
            source.sentences,
            segment_index=index,
            offset_ms=offset_ms,
            assignment=assignment,
        )
        merged.extend(shifted)
        # The timeline always advances by the *audio* duration so raw and
        # corrected variants share identical offsets even if polish changed the
        # sentence count.
        advance_ms, duration_source = _segment_duration_ms(
            segment, segment.result.sentences
        )
        metas.append(
            SegmentMeta(
                order=index,
                part=f"段{index + 1}",
                project_id=segment.project_id,
                project_dir=str(segment.project_dir),
                title=segment.title,
                meeting_time=segment.meeting_time,
                duration_seconds=segment.duration_seconds,
                duration_source=duration_source,
                clock_offset_ms=offset_ms,
                sentence_count=len(shifted),
                ignored_speaker_count=len(segment.ignored),
                corrected=use_corrected,
            )
        )
        offset_ms += advance_ms
    full_text = "".join(sentence.text for sentence in merged)
    detected = sorted(
        {sentence.speaker_id for sentence in merged if sentence.speaker_id is not None}
    )
    return TranscriptResult(full_text, merged, detected), metas


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-like timestamp, assuming +08:00 when no zone is given.

    Real project data mixes timezone-aware and naive ``meeting_time`` values, so
    comparing them directly raises ``TypeError``. Naive values are assumed to be
    Asia/Shanghai wall-clock.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CHINA_TZ)
    return parsed


def sort_segments(
    segments: Sequence[MergeSegment], *, keep_order: bool
) -> tuple[list[MergeSegment], str, list[str]]:
    """
    Order segments along the meeting timeline.

    Args:
        segments: Loaded segments in command-line order.
        keep_order: Skip sorting and keep command-line order.

    Returns:
        Ordered segments, the order source, and any warnings.
    """
    if keep_order:
        return list(segments), "cli", []
    keys: list[datetime] = []
    for segment in segments:
        parsed = _parse_timestamp(segment.meeting_time) or _parse_timestamp(
            segment.created_at
        )
        if parsed is None:
            return (
                list(segments),
                "cli_fallback",
                ["部分段缺少可解析的 meeting_time/created_at，回退命令行顺序"],
            )
        keys.append(parsed)
    ordered = [
        segment for _, segment in sorted(zip(keys, segments), key=lambda pair: pair[0])
    ]
    return ordered, "meeting_time", []


# ---------------------------------------------------------------------------
# Voiceprint name resolution
# ---------------------------------------------------------------------------


def make_vpp_name_resolver(
    store_dir: Path | None = None,
) -> Callable[[str], str | None]:
    """Return a cached resolver from voiceprint person id to display name."""
    db_path = (
        get_voiceprint_db_path(store_dir)
        if store_dir is not None
        else get_default_voiceprint_db_path()
    )
    cache: dict[str, str | None] = {}

    def resolve(vpp: str) -> str | None:
        if vpp not in cache:
            row = get_voiceprint_person(vpp, db_path)
            cache[vpp] = row.name if row else None
        return cache[vpp]

    return resolve


# ---------------------------------------------------------------------------
# Top-level merge
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MergeResult:
    """Everything needed to render and persist a merged package."""

    meeting: dict
    mapping: dict[int, str]
    identities: list[dict]
    order_source: str
    use_corrected: bool
    merged_raw: TranscriptResult
    metas_raw: list[SegmentMeta]
    merged_corrected: TranscriptResult | None
    metas_corrected: list[SegmentMeta] | None
    warnings: list[str]


def merge_projects(
    project_dirs: Sequence[Path],
    *,
    use_corrected: bool = True,
    name_to_vpp: bool = True,
    include_low_information: bool = False,
    keep_order: bool = False,
    store_dir: Path | None = None,
    title: str | None = None,
    vpp_name_resolver: Callable[[str], str | None] | None = None,
) -> MergeResult:
    """
    Merge several projects into one cross-segment-consistent transcript.

    Args:
        project_dirs: Resolved project roots (one is allowed; it degenerates to a
            direct re-export of that project).
        use_corrected: Emit the corrected variant when any segment has one.
        name_to_vpp: Promote name-only speakers onto matching voiceprint people.
        include_low_information: Keep backchannel-only speaker tracks.
        keep_order: Keep command-line order instead of sorting by meeting time.
        store_dir: Voiceprint store directory for authoritative names.
        title: Override the merged meeting title.
        vpp_name_resolver: Injected name resolver (tests); defaults to the store.

    Returns:
        A :class:`MergeResult` ready for :func:`write_merge_outputs`.

    Raises:
        ValueError: If no project is given.
    """
    if not project_dirs:
        raise ValueError("project merge requires at least one project.")

    warnings: list[str] = []
    loaded: list[MergeSegment] = []
    seen: set[str] = set()
    for project_dir in project_dirs:
        segment = load_merge_segment(
            project_dir, include_low_information=include_low_information
        )
        if segment.project_id in seen:
            warnings.append(f"重复 project {segment.project_id} 已去重")
            continue
        seen.add(segment.project_id)
        loaded.append(segment)

    ordered, order_source, order_warnings = sort_segments(loaded, keep_order=keep_order)
    warnings.extend(order_warnings)

    resolver = vpp_name_resolver
    if resolver is None:
        resolver = make_vpp_name_resolver(store_dir)
    table = build_global_identities(
        ordered, name_to_vpp=name_to_vpp, vpp_name_resolver=resolver
    )
    warnings.extend(table.warnings)

    merged_raw, metas_raw = build_merged_result(
        ordered, table.assignment, corrected=False
    )
    merged_corrected: TranscriptResult | None = None
    metas_corrected: list[SegmentMeta] | None = None
    if use_corrected and any(segment.result_corrected for segment in ordered):
        merged_corrected, metas_corrected = build_merged_result(
            ordered, table.assignment, corrected=True
        )
        missing = [meta.part for meta in metas_corrected if not meta.corrected]
        if missing:
            warnings.append(
                f"以下段无 polish 文本，corrected 版用原文兜底: {', '.join(missing)}"
            )

    meeting = _build_meeting_metadata(ordered, table, title=title)
    return MergeResult(
        meeting=meeting,
        mapping=table.mapping,
        identities=table.identities,
        order_source=order_source,
        use_corrected=merged_corrected is not None,
        merged_raw=merged_raw,
        metas_raw=metas_raw,
        merged_corrected=merged_corrected,
        metas_corrected=metas_corrected,
        warnings=warnings,
    )


def _build_meeting_metadata(
    segments: Sequence[MergeSegment],
    table: GlobalIdentityTable,
    *,
    title: str | None,
) -> dict:
    """Compose merged-meeting metadata (title, span, participants)."""
    meeting_time = segments[0].meeting_time if segments else None
    meeting_end: str | None = None
    last = segments[-1] if segments else None
    if last is not None:
        last_dt = _parse_timestamp(last.meeting_time)
        if last_dt is not None and last.duration_seconds:
            meeting_end = (
                last_dt + timedelta(seconds=last.duration_seconds)
            ).isoformat()
    participants = [
        entry["display_name"]
        for entry in table.identities
        if entry["identity_kind"] in ("vpp", "name")
    ]
    resolved_title = title or (segments[0].title if segments else "合并转写")
    return {
        "title": resolved_title,
        "meeting_time": meeting_time,
        "meeting_end": meeting_end,
        "participants": participants,
    }


# ---------------------------------------------------------------------------
# Rendering and persistence
# ---------------------------------------------------------------------------


def _format_clock(ms: int) -> str:
    """Format a millisecond offset as HH:MM:SS."""
    total = max(0, int(ms)) // 1000
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _segment_header(meta: SegmentMeta, total: int, *, corrected: bool) -> str:
    """Build the human-readable boundary header for one segment."""
    duration = (
        _format_clock(int(meta.duration_seconds * 1000))
        if meta.duration_seconds
        else "?"
    )
    pieces = [
        f"# ━━ {meta.part}/{total} · {meta.project_id} · {meta.title}",
    ]
    detail = [
        f"会议时间 {meta.meeting_time or '未知'}",
        f"时长 {duration}",
        f"合并轴起点 {_format_clock(meta.clock_offset_ms)}",
        f"{meta.sentence_count} 句",
    ]
    if corrected and not meta.corrected:
        detail.append("无 polish（用原文）")
    pieces.append("#    " + " · ".join(detail))
    pieces.append(f"# {SEGMENT_DIVIDER}")
    return "\n".join(pieces)


def _slice_by_counts(
    sentences: Sequence[SentenceSegment], counts: Iterable[int]
) -> list[list[SentenceSegment]]:
    """Split a flat sentence list back into per-segment slices."""
    slices: list[list[SentenceSegment]] = []
    cursor = 0
    for count in counts:
        slices.append(list(sentences[cursor : cursor + count]))
        cursor += count
    return slices


def render_merged_text(
    merged: TranscriptResult,
    metas: Sequence[SegmentMeta],
    mapping: dict[int, str],
    *,
    corrected: bool,
) -> str:
    """
    Render the merged transcript with per-segment boundary headers.

    A single segment degenerates to a plain named transcript with no header, so
    its output matches a direct project export.
    """
    if len(metas) <= 1:
        return render_named_speaker_text(merged, mapping)
    total = len(metas)
    slices = _slice_by_counts(merged.sentences, (meta.sentence_count for meta in metas))
    blocks: list[str] = []
    for meta, segment_sentences in zip(metas, slices):
        header = _segment_header(meta, total, corrected=corrected)
        body = render_named_speaker_text(
            TranscriptResult("", segment_sentences, []), mapping
        ).rstrip("\n")
        blocks.append(f"{header}\n\n{body}" if body else header)
    return "\n\n".join(blocks) + "\n"


def merge_payload(result: MergeResult) -> dict:
    """Build the ``merge.json`` manifest payload (also used by ``--json``)."""
    # ``metas_raw`` always carries ``corrected=False`` (it is the raw build);
    # whether a segment actually contributed polished text lives in the
    # corrected build, so read the per-segment flag from there.
    corrected_by_order = (
        {meta.order: meta.corrected for meta in result.metas_corrected}
        if result.metas_corrected is not None
        else {}
    )

    def _segment_dict(meta: SegmentMeta) -> dict:
        return {
            "order": meta.order,
            "part": meta.part,
            "project_id": meta.project_id,
            "project_dir": meta.project_dir,
            "title": meta.title,
            "meeting_time": meta.meeting_time,
            "duration_seconds": meta.duration_seconds,
            "duration_source": meta.duration_source,
            "clock_offset_ms": meta.clock_offset_ms,
            "sentence_count": meta.sentence_count,
            "ignored_speaker_count": meta.ignored_speaker_count,
            "corrected": corrected_by_order.get(meta.order, False),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "meeting-asr project merge",
        "timeline_mode": "concatenated",
        "use_corrected": result.use_corrected,
        "order_source": result.order_source,
        "meeting": result.meeting,
        "segments": [_segment_dict(meta) for meta in result.metas_raw],
        "identities": result.identities,
        "warnings": result.warnings,
    }


@dataclass(slots=True)
class MergeOutputs:
    """Paths written by :func:`write_merge_outputs`."""

    out_dir: Path
    transcript: Path
    transcript_corrected: Path | None
    subtitle: Path
    subtitle_corrected: Path | None
    manifest: Path


def write_merge_outputs(
    result: MergeResult, out_dir: Path, *, force: bool = False
) -> MergeOutputs:
    """
    Write the merged package to ``out_dir``.

    Args:
        result: Merge result to persist.
        out_dir: Output directory (created if missing).
        force: Allow writing into a non-empty directory.

    Returns:
        The written paths.

    Raises:
        FileExistsError: If ``out_dir`` is non-empty and ``force`` is not set.
    """
    out_dir = out_dir.expanduser()
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"输出目录非空: {out_dir}（用 --force 覆盖，或换一个 --out）"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = out_dir / "transcript_merged.txt"
    safe_write_text(
        transcript,
        render_merged_text(
            result.merged_raw, result.metas_raw, result.mapping, corrected=False
        ),
    )
    subtitle = out_dir / "subtitle_merged.srt"
    safe_write_text(subtitle, render_named_srt(result.merged_raw, result.mapping))

    transcript_corrected: Path | None = None
    subtitle_corrected: Path | None = None
    if result.merged_corrected is not None and result.metas_corrected is not None:
        transcript_corrected = out_dir / "transcript_merged_corrected.txt"
        safe_write_text(
            transcript_corrected,
            render_merged_text(
                result.merged_corrected,
                result.metas_corrected,
                result.mapping,
                corrected=True,
            ),
        )
        subtitle_corrected = out_dir / "subtitle_merged_corrected.srt"
        safe_write_text(
            subtitle_corrected,
            render_named_srt(result.merged_corrected, result.mapping),
        )

    manifest = safe_write_json(out_dir / "merge.json", merge_payload(result))
    return MergeOutputs(
        out_dir=out_dir,
        transcript=transcript,
        transcript_corrected=transcript_corrected,
        subtitle=subtitle,
        subtitle_corrected=subtitle_corrected,
        manifest=manifest,
    )


def default_output_dir(result: MergeResult) -> Path:
    """Return a deterministic default ``--out`` directory under the CWD."""
    meeting_time = result.meeting.get("meeting_time")
    parsed = _parse_timestamp(meeting_time)
    date_part = parsed.strftime("%Y%m%d") if parsed else "undated"
    first_id = result.metas_raw[0].project_id if result.metas_raw else "p"
    short = first_id.removeprefix("p-")[:8] or "merge"
    return Path.cwd() / f"merged-{date_part}-{short}"
