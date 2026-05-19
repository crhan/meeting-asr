"""Cross-project speaker voiceprint capture."""

from __future__ import annotations

import json
import math
import re
import struct
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.project_manager import ensure_project_dirs, load_manifest, resolve_project_source_path, save_manifest
from app.speaker_labeling import load_transcript_result
from app.voiceprint_audio import trim_embedding_audio_silence
from app.voiceprint_embedding import embed_audio_file
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    store_voiceprint_samples,
)
from app.voiceprint_people import get_voiceprint_person

MIN_SELECTION_SCORE = 0.30
MIN_RECOMMENDED_SCORE = 0.55


@dataclass(frozen=True, slots=True)
class VoiceprintClip:
    """One planned or written cross-project voiceprint clip."""

    path: Path
    rel_path: str
    source_begin_time_ms: int
    source_end_time_ms: int
    clip_begin_time_ms: int
    clip_end_time_ms: int
    text: str
    selection_score: float = 0.0
    selection_reason: str = "legacy"
    audio_score: float | None = None
    audio_reason: str = "not-analyzed"
    recommended: bool = True

    @property
    def duration_seconds(self) -> float:
        """Return clip duration in seconds."""
        return (self.clip_end_time_ms - self.clip_begin_time_ms) / 1000


@dataclass(frozen=True, slots=True)
class VoiceprintSpeaker:
    """Voiceprint references selected for one speaker."""

    speaker_id: int
    name: str
    person_id: int | None
    person_public_id: str | None
    clips: list[VoiceprintClip]


@dataclass(frozen=True, slots=True)
class VoiceprintCaptureSummary:
    """Result of capturing cross-project voiceprint references."""

    store_dir: Path
    db_path: Path
    clip_dir: Path
    speakers: list[VoiceprintSpeaker]
    dry_run: bool
    target_sample_count: int = 0

    @property
    def sample_count(self) -> int:
        """Return total selected sample count."""
        return sum(len(speaker.clips) for speaker in self.speakers)


def capture_voiceprints(
    project_dir: Path,
    *,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None = None,
    dry_run: bool = False,
    progress: CliProgressReporter | None = None,
) -> VoiceprintCaptureSummary:
    """
    Capture speaker reference clips into the global voiceprint store.

    Args:
        project_dir: Project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional XDG-style voiceprint store directory.
        dry_run: Only plan clips when true.
        progress: Optional progress reporter.

    Returns:
        Capture summary.
    """
    _validate_capture_options(sample_count, max_seconds, padding_seconds)
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    source = resolve_project_source_path(paths.root, manifest)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    resolved_store_dir = _resolve_store_dir(store_dir)
    clip_dir = get_voiceprint_clip_dir(resolved_store_dir)
    db_path = get_voiceprint_db_path(resolved_store_dir)
    identities = _load_required_speaker_identities(paths.speakers_dir, db_path)
    speakers = _build_voiceprint_speakers(
        clip_dir,
        manifest.project_id,
        result,
        identities,
        sample_count,
        max_seconds,
        padding_seconds,
    )
    if not speakers:
        raise RuntimeError(
            "No named speaker segments are available for voiceprint capture. "
            "Run meeting-asr project review and confirm speaker names first."
        )
    if dry_run:
        emit_progress(progress, "Voiceprint clips planned", total=_clip_count(speakers), completed=_clip_count(speakers))
    if not dry_run:
        speakers = _persist_voiceprint_capture(
            paths.root,
            manifest,
            source,
            speakers,
            resolved_store_dir,
            db_path,
            progress,
            target_sample_count=sample_count,
        )
    return VoiceprintCaptureSummary(resolved_store_dir, db_path, clip_dir, speakers, dry_run, sample_count)


def plan_voiceprint_capture(
    project_dir: Path,
    *,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None = None,
) -> VoiceprintCaptureSummary:
    """
    Plan voiceprint clips without writing WAV files or SQLite rows.

    Args:
        project_dir: Project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional XDG-style voiceprint store directory.

    Returns:
        Dry-run capture summary.
    """
    return capture_voiceprints(
        project_dir,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        store_dir=store_dir,
        dry_run=True,
        progress=None,
    )


def persist_voiceprint_capture_selection(
    project_dir: Path,
    *,
    planned: VoiceprintCaptureSummary,
    selected_clip_rel_paths: set[str] | frozenset[str],
    progress: CliProgressReporter | None = None,
) -> VoiceprintCaptureSummary:
    """
    Persist only the capture clips accepted by human review.

    Args:
        project_dir: Project root.
        planned: Dry-run capture summary.
        selected_clip_rel_paths: Accepted clip relative paths.
        progress: Optional progress reporter.

    Returns:
        Captured summary containing only accepted clips.
    """
    if not selected_clip_rel_paths:
        raise ValueError("No voiceprint clips were selected for capture.")
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    source = resolve_project_source_path(paths.root, manifest)
    speakers = _filter_voiceprint_speakers(planned.speakers, selected_clip_rel_paths)
    if not speakers:
        raise ValueError("No selected voiceprint clips matched the capture plan.")
    speakers = _persist_voiceprint_capture(
        paths.root,
        manifest,
        source,
        speakers,
        planned.store_dir,
        planned.db_path,
        progress,
        target_sample_count=planned.target_sample_count,
    )
    return VoiceprintCaptureSummary(planned.store_dir, planned.db_path, planned.clip_dir, speakers, False, planned.target_sample_count)


def _filter_voiceprint_speakers(
    speakers: list[VoiceprintSpeaker],
    selected_clip_rel_paths: set[str] | frozenset[str],
) -> list[VoiceprintSpeaker]:
    """Return speakers containing only selected clips."""
    filtered: list[VoiceprintSpeaker] = []
    for speaker in speakers:
        clips = [clip for clip in speaker.clips if clip.rel_path in selected_clip_rel_paths]
        if clips:
            filtered.append(
                VoiceprintSpeaker(
                    speaker_id=speaker.speaker_id,
                    name=speaker.name,
                    person_id=speaker.person_id,
                    person_public_id=speaker.person_public_id,
                    clips=clips,
                )
            )
    return filtered


def _persist_voiceprint_capture(
    project_root: Path,
    manifest,
    source: Path,
    speakers: list[VoiceprintSpeaker],
    store_dir: Path,
    db_path: Path,
    progress: CliProgressReporter | None,
    *,
    target_sample_count: int,
) -> list[VoiceprintSpeaker]:
    """
    Write clips, store SQLite rows, and update the project manifest pointer.

    Args:
        project_root: Project root.
        manifest: Loaded project manifest.
        source: Source media path.
        speakers: Selected voiceprint speakers.
        store_dir: Global voiceprint store directory.
        db_path: SQLite database path.
        progress: Optional progress reporter.
    """
    speakers = _write_voiceprint_clips(source, speakers, progress)
    speakers = _select_central_voiceprint_clips(speakers, target_sample_count)
    if not speakers:
        raise RuntimeError("No voiceprint clips passed audio quality checks.")
    emit_progress(progress, "Indexing voiceprint samples")
    samples = _stored_samples(project_root, manifest.project_id, source, speakers)
    store_voiceprint_samples(samples, db_path)
    manifest.speakers["voiceprints"] = {"store_dir": str(store_dir), "db_path": str(db_path), "sample_count": len(samples)}
    manifest.status = "voiceprinted"
    save_manifest(project_root, manifest)
    emit_progress(progress, "Voiceprint capture complete")
    return speakers


def _validate_capture_options(sample_count: int, max_seconds: float, padding_seconds: float) -> None:
    """
    Validate voiceprint capture options.

    Args:
        sample_count: Requested clips per speaker.
        max_seconds: Requested maximum clip length.
        padding_seconds: Requested context padding.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1.")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0.")
    if padding_seconds < 0:
        raise ValueError("padding_seconds must be >= 0.")


def _resolve_store_dir(store_dir: Path | None) -> Path:
    """
    Resolve an optional voiceprint store directory.

    Args:
        store_dir: Optional store directory.

    Returns:
        Absolute store directory.
    """
    if store_dir is None:
        return get_voiceprint_db_path().parent
    return store_dir.expanduser().resolve()


@dataclass(frozen=True, slots=True)
class _SpeakerIdentity:
    """Resolved project speaker identity for voiceprint capture."""

    name: str
    person_id: int | None
    person_public_id: str | None


def _load_required_speaker_identities(speakers_dir: Path, db_path: Path) -> dict[int, _SpeakerIdentity]:
    """
    Load named speaker identities required for voiceprint references.

    Args:
        speakers_dir: Project speakers directory.
        db_path: Voiceprint SQLite path.

    Returns:
        Project speaker id to identity mapping.
    """
    path = speakers_dir / "speaker_map.json"
    if not path.exists():
        raise FileNotFoundError("Speaker mapping does not exist. Run meeting-asr project review first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in payload.items()}
    person_map = _load_speaker_person_refs(speakers_dir / "speaker_person_map.json")
    return {
        speaker_id: _identity_for_capture(speaker_id, name, person_map, db_path)
        for speaker_id, name in names.items()
    }


def _load_speaker_person_refs(path: Path) -> dict[int, int | str]:
    """
    Load project speaker to voiceprint person references.

    Args:
        path: Mapping JSON path.

    Returns:
        Speaker id to numeric or public voiceprint person reference.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload.items() if isinstance(value, int | str)}


def _identity_for_capture(
    speaker_id: int,
    name: str,
    person_map: dict[int, int | str],
    db_path: Path,
) -> _SpeakerIdentity:
    """
    Resolve a voiceprint person id for one project speaker.

    Args:
        speaker_id: Project speaker id.
        name: Display name from the legacy speaker map.
        person_map: Project speaker to voiceprint person id map.
        db_path: Voiceprint SQLite path.

    Returns:
        Speaker identity with stable person ids when present.
    """
    person_ref = person_map.get(speaker_id)
    if person_ref is None:
        person = get_voiceprint_person(name, db_path)
        if person is not None:
            return _SpeakerIdentity(name, person.speaker_id, person.public_id)
        return _SpeakerIdentity(name, None, None)
    person = get_voiceprint_person(person_ref, db_path)
    if person is None:
        raise LookupError(f"speaker_person_map.json points to missing voiceprint person id {person_ref} for {name}.")
    return _SpeakerIdentity(name, person.speaker_id, person.public_id)


def _build_voiceprint_speakers(
    clip_dir: Path,
    project_id: str,
    result: TranscriptResult,
    identities: dict[int, _SpeakerIdentity],
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
) -> list[VoiceprintSpeaker]:
    """
    Select reference clips for all identified speakers in the transcript.

    Args:
        clip_dir: Global voiceprint clip directory.
        project_id: Project id.
        result: Normalized transcript result.
        identities: Speaker id to resolved identity mapping.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.

    Returns:
        Selected voiceprint speakers.
    """
    speakers: list[VoiceprintSpeaker] = []
    grouped = _speaker_segments_by_id(result)
    for speaker_id in sorted(grouped):
        identity = _identified_speaker_identity(speaker_id, identities)
        if identity is None:
            continue
        selected = _select_segments(grouped[speaker_id], result.sentences, sample_count)
        clips = _build_clips(clip_dir, project_id, speaker_id, selected, max_seconds, padding_seconds)
        if clips:
            speakers.append(VoiceprintSpeaker(speaker_id, identity.name, identity.person_id, identity.person_public_id, clips))
    return speakers


def _identified_speaker_identity(
    speaker_id: int,
    identities: dict[int, _SpeakerIdentity],
) -> _SpeakerIdentity | None:
    """
    Return a human name only when the speaker is actually identified.

    Args:
        speaker_id: Speaker id.
        identities: Speaker id to mapped identity.

    Returns:
        Human identity, or ``None`` for anonymous fallback labels.
    """
    identity = identities.get(speaker_id)
    if identity is None:
        return None
    name = identity.name.strip()
    if not name or name == speaker_id_to_label(speaker_id):
        return None
    return identity


def _speaker_segments_by_id(result: TranscriptResult) -> dict[int, list[SentenceSegment]]:
    """
    Group usable transcript segments by speaker id.

    Args:
        result: Normalized transcript result.

    Returns:
        Speaker id to usable sentence segments.
    """
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for segment in result.sentences:
        if segment.speaker_id is None or not segment.text.strip():
            continue
        if segment.end_time_ms <= segment.begin_time_ms:
            continue
        grouped[segment.speaker_id].append(segment)
    return grouped


@dataclass(frozen=True, slots=True)
class _ScoredSegment:
    """One voiceprint candidate segment with selection diagnostics."""

    segment: SentenceSegment
    score: float
    reason: str
    recommended: bool = False


def _select_segments(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    sample_count: int,
) -> list[_ScoredSegment]:
    """
    Select the longest segments and return them in timeline order.

    Args:
        segments: Candidate segments for one speaker.
        all_segments: Full transcript segments for boundary scoring.
        sample_count: Maximum segment count.

    Returns:
        Selected segments.
    """
    candidate_count = max(sample_count, 12)
    scored = _scored_unique_segments(segments, all_segments)
    selected = _select_diverse_segments(scored, candidate_count, min_gap_ms=10_000)
    if len(selected) < candidate_count:
        selected = _select_diverse_segments(scored, candidate_count, min_gap_ms=2_000)
    if len(selected) < candidate_count:
        selected = _select_diverse_segments(scored, candidate_count, min_gap_ms=0)
    recommended_ids = _recommended_segment_ids(selected, sample_count)
    marked = [
        _ScoredSegment(item.segment, item.score, item.reason, id(item.segment) in recommended_ids)
        for item in selected
    ]
    return sorted(marked, key=lambda item: item.segment.begin_time_ms)


def _recommended_segment_ids(segments: list[_ScoredSegment], sample_count: int) -> set[int]:
    """
    Pick default checked samples from a good-enough, time-spread pool.

    The score is a usability gate, not a target to maximize. Always taking the
    highest scores overfits toward one speaking style and often clusters around
    long dense monologues, so defaults should cover the speaker's timeline.
    """
    if sample_count <= 0 or not segments:
        return set()
    eligible = [item for item in segments if item.score >= MIN_RECOMMENDED_SCORE]
    if len(eligible) < sample_count:
        eligible = segments
    return {id(item.segment) for item in _spread_segments_by_time(eligible, sample_count)}


def _spread_segments_by_time(segments: list[_ScoredSegment], sample_count: int) -> list[_ScoredSegment]:
    """Return up to sample_count segments spread across the speaker timeline."""
    ordered = sorted(segments, key=lambda item: item.segment.begin_time_ms)
    if len(ordered) <= sample_count:
        return ordered
    if sample_count == 1:
        return [ordered[len(ordered) // 2]]
    last_index = len(ordered) - 1
    indices = [round(index * last_index / (sample_count - 1)) for index in range(sample_count)]
    return [ordered[index] for index in indices]


def _scored_unique_segments(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
) -> list[_ScoredSegment]:
    """Return unique candidate segments sorted by descending quality score."""
    seen: set[tuple[int, int, str]] = set()
    scored: list[_ScoredSegment] = []
    for segment in segments:
        key = (segment.begin_time_ms, segment.end_time_ms, _normalized_text(segment.text))
        if key in seen:
            continue
        seen.add(key)
        candidate = _score_segment(segment, all_segments)
        if candidate.score >= MIN_SELECTION_SCORE:
            scored.append(candidate)
    return sorted(scored, key=lambda item: (-item.score, item.segment.begin_time_ms))


def _select_diverse_segments(
    candidates: list[_ScoredSegment],
    sample_count: int,
    *,
    min_gap_ms: int,
) -> list[_ScoredSegment]:
    """Select high-scoring segments while avoiding nearby duplicates."""
    selected: list[_ScoredSegment] = []
    for candidate in candidates:
        if len(selected) >= sample_count:
            break
        if any(_segments_too_close(candidate.segment, item.segment, min_gap_ms) for item in selected):
            continue
        selected.append(candidate)
    return selected


def _segments_too_close(left: SentenceSegment, right: SentenceSegment, min_gap_ms: int) -> bool:
    """Return whether two candidate segments are too close for diverse sampling."""
    if left.begin_time_ms <= right.end_time_ms and right.begin_time_ms <= left.end_time_ms:
        return True
    return abs(left.begin_time_ms - right.begin_time_ms) < min_gap_ms


def _score_segment(segment: SentenceSegment, all_segments: list[SentenceSegment]) -> _ScoredSegment:
    """Score one transcript segment for voiceprint capture."""
    duration_ms = _segment_duration_ms(segment)
    text = segment.text.strip()
    if _is_low_information_text(_normalized_text(text)):
        return _ScoredSegment(segment, 0.0, "low-information")
    duration_score = _duration_score(duration_ms)
    text_score = _text_score(text)
    boundary_score = _boundary_score(segment, all_segments)
    score = round(0.45 * duration_score + 0.35 * text_score + 0.20 * boundary_score, 3)
    reason = f"duration={duration_score:.2f}, text={text_score:.2f}, boundary={boundary_score:.2f}"
    return _ScoredSegment(segment, score, reason)


def _duration_score(duration_ms: int) -> float:
    """Score duration, preferring 6-18 second speech samples."""
    seconds = duration_ms / 1000
    if seconds < 3:
        return max(0.0, seconds / 3 * 0.45)
    if seconds <= 18:
        return 1.0
    if seconds <= 30:
        return 0.8
    return 0.55


def _text_score(text: str) -> float:
    """Score text content, penalizing filler-only fragments."""
    normalized = _normalized_text(text)
    if not normalized:
        return 0.0
    if _is_low_information_text(normalized):
        return 0.1
    length_score = min(1.0, len(normalized) / 24)
    unique_score = min(1.0, len(set(normalized)) / 10)
    return round(0.65 * length_score + 0.35 * unique_score, 3)


def _boundary_score(segment: SentenceSegment, all_segments: list[SentenceSegment]) -> float:
    """Score speaker-boundary safety using neighboring transcript segments."""
    sorted_segments = sorted(all_segments, key=lambda item: (item.begin_time_ms, item.end_time_ms))
    index = next((i for i, item in enumerate(sorted_segments) if item is segment), -1)
    if index < 0:
        return 0.7
    previous_score = _neighbor_score(segment, sorted_segments[index - 1] if index else None, before=True)
    next_score = _neighbor_score(segment, sorted_segments[index + 1] if index + 1 < len(sorted_segments) else None, before=False)
    return min(previous_score, next_score)


def _neighbor_score(segment: SentenceSegment, neighbor: SentenceSegment | None, *, before: bool) -> float:
    """Score one neighboring segment for possible speaker overlap."""
    if neighbor is None or neighbor.speaker_id == segment.speaker_id:
        return 1.0
    gap = segment.begin_time_ms - neighbor.end_time_ms if before else neighbor.begin_time_ms - segment.end_time_ms
    if gap < 0:
        return 0.2
    if gap < 500:
        return 0.45
    if gap < 1200:
        return 0.75
    return 1.0


def _normalized_text(text: str) -> str:
    """Return compact text for quality heuristics."""
    return re.sub(r"\s+", "", text.strip().casefold())


def _is_low_information_text(text: str) -> bool:
    """Return whether text is mostly filler/backchannel content."""
    filler_pattern = r"(嗯+|啊+|呃+|哦+|对+|是+|好+|可以|就是|然后|那个|这个|ok|嗯哼|哈哈)+"
    return re.fullmatch(filler_pattern, text) is not None


def _segment_duration_ms(segment: SentenceSegment) -> int:
    """
    Return a segment duration in milliseconds.

    Args:
        segment: Transcript segment.

    Returns:
        Non-negative duration.
    """
    return max(0, segment.end_time_ms - segment.begin_time_ms)


def _build_clips(
    clip_dir: Path,
    project_id: str,
    speaker_id: int,
    segments: list[_ScoredSegment],
    max_seconds: float,
    padding_seconds: float,
) -> list[VoiceprintClip]:
    """
    Build output clip descriptors for selected segments.

    Args:
        clip_dir: Global voiceprint clip directory.
        project_id: Project id.
        speaker_id: Speaker id.
        segments: Selected source segments.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.

    Returns:
        Clip descriptors.
    """
    clips: list[VoiceprintClip] = []
    for index, candidate in enumerate(segments, start=1):
        clip = _build_clip(clip_dir, project_id, speaker_id, index, candidate, max_seconds, padding_seconds)
        if clip.duration_seconds > 0:
            clips.append(clip)
    return clips


def _build_clip(
    clip_dir: Path,
    project_id: str,
    speaker_id: int,
    index: int,
    candidate: _ScoredSegment,
    max_seconds: float,
    padding_seconds: float,
) -> VoiceprintClip:
    """
    Build one output clip descriptor.

    Args:
        clip_dir: Global voiceprint clip directory.
        project_id: Project id.
        speaker_id: Speaker id.
        index: One-based clip index.
        candidate: Source transcript segment with selection diagnostics.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around the segment.

    Returns:
        Clip descriptor.
    """
    padding_ms = int(round(padding_seconds * 1000))
    max_ms = int(round(max_seconds * 1000))
    segment = candidate.segment
    clip_begin = max(0, segment.begin_time_ms - padding_ms)
    clip_end = min(segment.end_time_ms + padding_ms, clip_begin + max_ms)
    rel_path = f"clips/{project_id}/speaker_{speaker_id}/clip_{index:03d}.wav"
    path = clip_dir.parent / rel_path
    return VoiceprintClip(
        path,
        rel_path,
        segment.begin_time_ms,
        segment.end_time_ms,
        clip_begin,
        clip_end,
        segment.text.strip(),
        candidate.score,
        candidate.reason,
        recommended=candidate.recommended,
    )


def _write_voiceprint_clips(
    source: Path,
    speakers: list[VoiceprintSpeaker],
    progress: CliProgressReporter | None,
) -> list[VoiceprintSpeaker]:
    """
    Write all planned voiceprint clips.

    Args:
        source: Source media path.
        speakers: Planned voiceprint speakers.
        progress: Optional progress reporter.
    """
    captured_speakers: list[VoiceprintSpeaker] = []
    emit_progress(progress, "Writing voiceprint clips", total=_clip_count(speakers), completed=0)
    for speaker in speakers:
        captured_clips: list[VoiceprintClip] = []
        for clip in speaker.clips:
            extract_audio_clip(
                source,
                clip.path,
                start_seconds=clip.clip_begin_time_ms / 1000,
                duration_seconds=clip.duration_seconds,
            )
            clip = _with_audio_quality(clip, _analyze_wav_quality(clip.path))
            if not _should_skip_audio_sample(clip):
                captured_clips.append(clip)
            emit_progress(progress, f"Captured {speaker.name} voiceprint clip", advance=1)
        if captured_clips:
            captured_speakers.append(
                VoiceprintSpeaker(speaker.speaker_id, speaker.name, speaker.person_id, speaker.person_public_id, captured_clips)
            )
    return captured_speakers


def _with_audio_quality(clip: VoiceprintClip, quality: tuple[float | None, str]) -> VoiceprintClip:
    """Return clip metadata annotated with WAV quality diagnostics."""
    score, reason = quality
    return VoiceprintClip(
        clip.path,
        clip.rel_path,
        clip.source_begin_time_ms,
        clip.source_end_time_ms,
        clip.clip_begin_time_ms,
        clip.clip_end_time_ms,
        clip.text,
        clip.selection_score,
        clip.selection_reason,
        score,
        reason,
        clip.recommended,
    )


def _analyze_wav_quality(path: Path) -> tuple[float | None, str]:
    """Return a simple 0-1 WAV quality score and reason."""
    try:
        with wave.open(str(path), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
            sample_width = reader.getsampwidth()
    except (wave.Error, OSError, EOFError):
        return None, "audio=unknown"
    if sample_width != 2 or not frames:
        return None, "audio=unknown"
    samples = [value[0] for value in struct.iter_unpack("<h", frames[: len(frames) - len(frames) % 2])]
    if not samples:
        return None, "audio=unknown"
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    silent_ratio = sum(1 for sample in samples if abs(sample) < 80) / len(samples)
    if rms < 80 or silent_ratio > 0.85:
        return round(min(rms / 400, 1.0), 3), "audio=low-volume"
    if peak > 32000:
        return 0.35, "audio=clipped"
    return round(min(rms / 1600, 1.0), 3), "audio=ok"


def _should_skip_audio_sample(clip: VoiceprintClip) -> bool:
    """Return whether an analyzed clip should be excluded from storage."""
    return clip.audio_reason in {"audio=low-volume", "audio=clipped"}


def _select_central_voiceprint_clips(
    speakers: list[VoiceprintSpeaker],
    target_sample_count: int,
) -> list[VoiceprintSpeaker]:
    """Select final clips by embedding centrality when extra candidates exist."""
    if target_sample_count <= 0:
        return speakers
    selected_speakers: list[VoiceprintSpeaker] = []
    for speaker in speakers:
        clips = _select_central_clips_for_speaker(speaker.clips, target_sample_count)
        if clips:
            selected_speakers.append(
                VoiceprintSpeaker(speaker.speaker_id, speaker.name, speaker.person_id, speaker.person_public_id, clips)
            )
    return selected_speakers


def _select_central_clips_for_speaker(clips: list[VoiceprintClip], target_sample_count: int) -> list[VoiceprintClip]:
    """Return final clips closest to the candidate embedding centroid."""
    if len(clips) <= target_sample_count:
        return clips
    if len(clips) < 3:
        recommended = [clip for clip in clips if clip.recommended]
        return (recommended or clips)[:target_sample_count]
    ranked = _rank_clips_by_embedding_centrality(clips)
    if not ranked:
        ranked = [(clip, clip.selection_score) for clip in clips]
    selected = [clip for clip, _score in sorted(ranked, key=lambda item: item[1], reverse=True)[:target_sample_count]]
    return sorted(selected, key=lambda clip: clip.source_begin_time_ms)


def _rank_clips_by_embedding_centrality(clips: list[VoiceprintClip]) -> list[tuple[VoiceprintClip, float]]:
    """Rank clips by distance to the candidate embedding centroid."""
    vectors: list[tuple[VoiceprintClip, list[float]]] = []
    for clip in clips:
        try:
            embedding_path = clip.path.with_name(f"{clip.path.stem}_embedding.wav")
            trim_embedding_audio_silence(clip.path, embedding_path)
            vectors.append((clip, _normalize_vector(embed_audio_file(embedding_path, provider=None))))
        except Exception:
            return []
    centroid = _normalize_vector([sum(values) / len(vectors) for values in zip(*(vector for _clip, vector in vectors))])
    return [(clip, _cosine(vector, centroid) + 0.05 * clip.selection_score) for clip, vector in vectors]


def _normalize_vector(vector: list[float]) -> list[float]:
    """Return a unit vector, preserving zero vectors."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _clip_count(speakers: list[VoiceprintSpeaker]) -> int:
    """
    Count clips across selected speakers.

    Args:
        speakers: Selected voiceprint speakers.

    Returns:
        Total clip count.
    """
    return sum(len(speaker.clips) for speaker in speakers)


def _stored_samples(
    project_root: Path,
    project_id: str,
    source: Path,
    speakers: list[VoiceprintSpeaker],
) -> list[StoredVoiceprintSample]:
    """
    Build SQLite sample rows for captured clips.

    Args:
        project_root: Project root.
        project_id: Project id.
        source: Source media path.
        speakers: Captured voiceprint speakers.

    Returns:
        Samples ready for SQLite storage.
    """
    samples: list[StoredVoiceprintSample] = []
    for speaker in speakers:
        for clip in speaker.clips:
            samples.append(_stored_sample(project_root, project_id, source, speaker, clip))
    return samples


def _stored_sample(
    project_root: Path,
    project_id: str,
    source: Path,
    speaker: VoiceprintSpeaker,
    clip: VoiceprintClip,
) -> StoredVoiceprintSample:
    """
    Build one SQLite sample row.

    Args:
        project_root: Project root.
        project_id: Project id.
        source: Source media path.
        speaker: Captured speaker.
        clip: Captured clip.

    Returns:
        Sample ready for SQLite storage.
    """
    return StoredVoiceprintSample(
        speaker_name=speaker.name,
        person_id=speaker.person_id,
        project_id=project_id,
        project_path=project_root,
        project_speaker_id=speaker.speaker_id,
        source_path=source,
        clip_path=clip.path,
        clip_rel_path=clip.rel_path,
        source_begin_time_ms=clip.source_begin_time_ms,
        source_end_time_ms=clip.source_end_time_ms,
        clip_begin_time_ms=clip.clip_begin_time_ms,
        clip_end_time_ms=clip.clip_end_time_ms,
        transcript_text=clip.text,
    )
