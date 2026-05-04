"""Cross-project speaker voiceprint capture."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.project_manager import ensure_project_dirs, load_manifest, resolve_project_source_path, save_manifest
from app.speaker_labeling import load_transcript_result
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    store_voiceprint_samples,
)
from app.voiceprint_people import get_voiceprint_person


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
        _persist_voiceprint_capture(paths.root, manifest, source, speakers, resolved_store_dir, db_path, progress)
    return VoiceprintCaptureSummary(resolved_store_dir, db_path, clip_dir, speakers, dry_run)


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
    _persist_voiceprint_capture(paths.root, manifest, source, speakers, planned.store_dir, planned.db_path, progress)
    return VoiceprintCaptureSummary(planned.store_dir, planned.db_path, planned.clip_dir, speakers, False)


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
) -> None:
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
    _write_voiceprint_clips(source, speakers, progress)
    emit_progress(progress, "Indexing voiceprint samples")
    samples = _stored_samples(project_root, manifest.project_id, source, speakers)
    store_voiceprint_samples(samples, db_path)
    manifest.speakers["voiceprints"] = {"store_dir": str(store_dir), "db_path": str(db_path), "sample_count": len(samples)}
    manifest.status = "voiceprinted"
    save_manifest(project_root, manifest)
    emit_progress(progress, "Voiceprint capture complete")


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
) -> int | None:
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
        selected = _select_segments(grouped[speaker_id], sample_count)
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


def _select_segments(segments: list[SentenceSegment], sample_count: int) -> list[SentenceSegment]:
    """
    Select the longest segments and return them in timeline order.

    Args:
        segments: Candidate segments for one speaker.
        sample_count: Maximum segment count.

    Returns:
        Selected segments.
    """
    longest = sorted(segments, key=_segment_duration_ms, reverse=True)[:sample_count]
    return sorted(longest, key=lambda segment: segment.begin_time_ms)


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
    segments: list[SentenceSegment],
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
    for index, segment in enumerate(segments, start=1):
        clip = _build_clip(clip_dir, project_id, speaker_id, index, segment, max_seconds, padding_seconds)
        if clip.duration_seconds > 0:
            clips.append(clip)
    return clips


def _build_clip(
    clip_dir: Path,
    project_id: str,
    speaker_id: int,
    index: int,
    segment: SentenceSegment,
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
        segment: Source transcript segment.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around the segment.

    Returns:
        Clip descriptor.
    """
    padding_ms = int(round(padding_seconds * 1000))
    max_ms = int(round(max_seconds * 1000))
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
    )


def _write_voiceprint_clips(
    source: Path,
    speakers: list[VoiceprintSpeaker],
    progress: CliProgressReporter | None,
) -> None:
    """
    Write all planned voiceprint clips.

    Args:
        source: Source media path.
        speakers: Planned voiceprint speakers.
        progress: Optional progress reporter.
    """
    emit_progress(progress, "Writing voiceprint clips", total=_clip_count(speakers), completed=0)
    for speaker in speakers:
        for clip in speaker.clips:
            extract_audio_clip(
                source,
                clip.path,
                start_seconds=clip.clip_begin_time_ms / 1000,
                duration_seconds=clip.duration_seconds,
            )
            emit_progress(progress, f"Captured {speaker.name} voiceprint clip", advance=1)


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
