"""Cross-project speaker voiceprint capture."""

from __future__ import annotations

import json
import math
import shutil
import struct
import tempfile
import wave
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.project_manager import (
    ensure_project_dirs,
    load_manifest,
    resolve_project_audio_path,
    save_manifest,
)
from app.speaker_labeling import load_transcript_result
from app.voiceprint_audio import trim_embedding_audio_silence
from app.voiceprint_embedding import embed_audio_file
from app.voiceprint_segment_selection import (
    ScoredVoiceprintSegment,
    select_voiceprint_segments,
)
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    list_any_embedded_sample_ids,
    store_voiceprint_samples_with_rows,
)
from app.voiceprint_people import get_voiceprint_person
from app.voiceprint_models import VoiceprintSampleRow


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
class VoiceprintCapturedSample:
    """One sample actually present in the registry after capture."""

    sample_id: int
    public_id: str
    clip_path: Path
    embedded: bool


@dataclass(frozen=True, slots=True)
class VoiceprintCaptureDecision:
    """Selection and execution result for one project speaker."""

    speaker_id: int
    name: str
    person_id: int | None
    person_public_id: str | None
    existing_sample_count: int
    decision: str
    reason: str
    samples: list[VoiceprintCapturedSample] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceprintCaptureSummary:
    """Result of capturing cross-project voiceprint references."""

    store_dir: Path
    db_path: Path
    clip_dir: Path
    speakers: list[VoiceprintSpeaker]
    dry_run: bool
    target_sample_count: int = 0
    project_id: str = ""
    decisions: list[VoiceprintCaptureDecision] = field(default_factory=list)
    only_needed: bool = False
    min_samples: int = 10

    @property
    def sample_count(self) -> int:
        """Return total selected sample count."""
        return sum(len(speaker.clips) for speaker in self.speakers)

    @property
    def failed_count(self) -> int:
        """Return the number of speakers whose capture failed."""
        return sum(1 for item in self.decisions if item.decision == "failed")


def capture_voiceprints(
    project_dir: Path,
    *,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None = None,
    dry_run: bool = False,
    speaker_ids: set[int] | None = None,
    only_needed: bool = False,
    min_samples: int = 10,
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
        speaker_ids: Optional project speaker ids to capture.
        only_needed: Skip people who already have enough samples.
        min_samples: Existing-sample threshold used by ``only_needed``.
        progress: Optional progress reporter.

    Returns:
        Capture summary.
    """
    _validate_capture_options(
        sample_count, max_seconds, padding_seconds, min_samples=min_samples
    )
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    source = resolve_project_audio_path(paths.root, manifest)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    resolved_store_dir = _resolve_store_dir(store_dir)
    clip_dir = get_voiceprint_clip_dir(resolved_store_dir)
    db_path = get_voiceprint_db_path(resolved_store_dir)
    identities = _load_required_speaker_identities(
        paths.speakers_dir, db_path, speaker_ids=speaker_ids
    )
    speakers, decisions = _plan_voiceprint_speakers(
        clip_dir,
        manifest.project_id,
        result,
        identities,
        sample_count,
        max_seconds,
        padding_seconds,
        speaker_ids=speaker_ids,
        only_needed=only_needed,
        min_samples=min_samples,
    )
    if not speakers and not decisions:
        # ValueError: this is a user-input condition ("name someone first"), not a server
        # fault -- the web boundary maps it to 400 with the message intact even on
        # non-loopback binds (500 detail gets scrubbed there).
        raise ValueError(
            "No named speaker segments are available for voiceprint capture. "
            "Run meeting-asr project review and confirm speaker names first."
        )
    if dry_run:
        emit_progress(
            progress,
            "Voiceprint clips planned",
            total=_clip_count(speakers),
            completed=_clip_count(speakers),
        )
    if not dry_run and speakers:
        persisted = _persist_voiceprint_capture(
            paths.root,
            manifest,
            source,
            speakers,
            resolved_store_dir,
            db_path,
            progress,
            target_sample_count=sample_count,
            continue_on_error=True,
        )
        speakers = persisted.speakers
        decisions = _apply_capture_results(decisions, persisted, db_path)
    return VoiceprintCaptureSummary(
        resolved_store_dir,
        db_path,
        clip_dir,
        speakers,
        dry_run,
        sample_count,
        manifest.project_id,
        decisions,
        only_needed,
        min_samples,
    )


def plan_voiceprint_capture(
    project_dir: Path,
    *,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None = None,
    speaker_ids: set[int] | None = None,
    only_needed: bool = False,
    min_samples: int = 10,
) -> VoiceprintCaptureSummary:
    """
    Plan voiceprint clips without writing WAV files or SQLite rows.

    Args:
        project_dir: Project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional XDG-style voiceprint store directory.
        speaker_ids: Optional project speaker ids to capture.
        only_needed: Skip people who already have enough samples.
        min_samples: Existing-sample threshold used by ``only_needed``.

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
        speaker_ids=speaker_ids,
        only_needed=only_needed,
        min_samples=min_samples,
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
    source = resolve_project_audio_path(paths.root, manifest)
    speakers = _filter_voiceprint_speakers(planned.speakers, selected_clip_rel_paths)
    if not speakers:
        raise ValueError("No selected voiceprint clips matched the capture plan.")
    persisted = _persist_voiceprint_capture(
        paths.root,
        manifest,
        source,
        speakers,
        planned.store_dir,
        planned.db_path,
        progress,
        target_sample_count=planned.target_sample_count,
    )
    speakers = persisted.speakers
    decisions = (
        _apply_capture_results(planned.decisions, persisted, planned.db_path)
        if planned.decisions
        else []
    )
    return VoiceprintCaptureSummary(
        planned.store_dir,
        planned.db_path,
        planned.clip_dir,
        speakers,
        False,
        planned.target_sample_count,
        manifest.project_id,
        decisions,
        planned.only_needed,
        planned.min_samples,
    )


def _filter_voiceprint_speakers(
    speakers: list[VoiceprintSpeaker],
    selected_clip_rel_paths: set[str] | frozenset[str],
) -> list[VoiceprintSpeaker]:
    """Return speakers containing only selected clips."""
    filtered: list[VoiceprintSpeaker] = []
    for speaker in speakers:
        clips = [
            clip for clip in speaker.clips if clip.rel_path in selected_clip_rel_paths
        ]
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


@dataclass(frozen=True, slots=True)
class _PersistedCapture:
    """Internal per-speaker persistence result."""

    speakers: list[VoiceprintSpeaker]
    rows_by_speaker: dict[int, list[VoiceprintSampleRow]]
    failures: dict[int, str]


@dataclass(frozen=True, slots=True)
class _CaptureFileBackup:
    """Rollback information for one deterministic capture output."""

    path: Path
    backup_path: Path
    existed: bool


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
    continue_on_error: bool = False,
) -> _PersistedCapture:
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
    captured: list[VoiceprintSpeaker] = []
    rows_by_speaker: dict[int, list[VoiceprintSampleRow]] = {}
    failures: dict[int, str] = {}
    for speaker in speakers:
        try:
            persisted_speaker, rows = _persist_voiceprint_speaker(
                project_root,
                manifest.project_id,
                source,
                speaker,
                db_path,
                progress,
                target_sample_count=target_sample_count,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures[speaker.speaker_id] = str(exc)
            continue
        captured.append(persisted_speaker)
        rows_by_speaker[speaker.speaker_id] = rows
    sample_count = sum(len(rows) for rows in rows_by_speaker.values())
    if not captured and not failures:
        raise RuntimeError("No voiceprint clips passed audio quality checks.")
    if not captured:
        return _PersistedCapture([], {}, failures)
    manifest.speakers["voiceprints"] = {
        "store_dir": str(store_dir),
        "db_path": str(db_path),
        "sample_count": sample_count,
    }
    manifest.status = "voiceprinted"
    save_manifest(project_root, manifest)
    emit_progress(progress, "Voiceprint capture complete")
    return _PersistedCapture(captured, rows_by_speaker, failures)


def _persist_voiceprint_speaker(
    project_root: Path,
    project_id: str,
    source: Path,
    speaker: VoiceprintSpeaker,
    db_path: Path,
    progress: CliProgressReporter | None,
    *,
    target_sample_count: int,
) -> tuple[VoiceprintSpeaker, list[VoiceprintSampleRow]]:
    """Capture and index one speaker as an atomic unit.

    SQLite already rolls a failed batch back.  This wrapper also snapshots every
    deterministic clip target so extraction or database failures cannot leave
    orphaned files or overwrite a previously valid sample.
    """
    with tempfile.TemporaryDirectory(prefix="meeting-asr-voiceprint-capture-") as raw:
        backup_root = Path(raw)
        backups = _backup_capture_files(speaker, backup_root)
        try:
            written = _write_voiceprint_clips(source, [speaker], progress)
            selected = _select_central_voiceprint_clips(written, target_sample_count)
            if not selected:
                raise RuntimeError(
                    f"No voiceprint clips passed audio quality checks for speaker {speaker.speaker_id}."
                )
            emit_progress(progress, f"Indexing {speaker.name} voiceprint samples")
            samples = _stored_samples(project_root, project_id, source, selected)
            _database_path, rows = store_voiceprint_samples_with_rows(samples, db_path)
            selected_speaker = selected[0]
            if not rows:
                raise RuntimeError(
                    f"No new voiceprint samples were indexed for speaker {speaker.speaker_id}."
                )
            persisted = selected_speaker
        except Exception:
            _restore_capture_files(backups, keep=set())
            raise
        _restore_capture_files(
            backups,
            # Preserve the historical capture contract: candidate WAVs remain
            # available for review even when centrality or duplicate detection
            # indexes only a subset.  The snapshot is still restored in full on
            # any failure before the batch commits.
            keep={clip.path.expanduser().resolve() for clip in speaker.clips},
            strict=False,
        )
        return persisted, rows


def _backup_capture_files(
    speaker: VoiceprintSpeaker, backup_root: Path
) -> list[_CaptureFileBackup]:
    """Snapshot clip and temporary embedding targets for one speaker."""
    backups: list[_CaptureFileBackup] = []
    paths: list[Path] = []
    for clip in speaker.clips:
        paths.append(clip.path.expanduser().resolve())
        paths.append(
            clip.path.with_name(f"{clip.path.stem}_embedding.wav")
            .expanduser()
            .resolve()
        )
    for index, path in enumerate(dict.fromkeys(paths)):
        backup_path = backup_root / f"{index:04d}-{path.name}"
        existed = path.is_file()
        if existed:
            shutil.copy2(path, backup_path)
        backups.append(_CaptureFileBackup(path, backup_path, existed))
    return backups


def _restore_capture_files(
    backups: list[_CaptureFileBackup], *, keep: set[Path], strict: bool = True
) -> None:
    """Restore or remove every capture target not committed by the speaker batch."""
    for item in backups:
        resolved = item.path.expanduser().resolve()
        if resolved in keep:
            continue
        try:
            if item.existed:
                item.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.backup_path, item.path)
            else:
                item.path.unlink(missing_ok=True)
        except OSError:
            if strict:
                raise


def _apply_capture_results(
    decisions: list[VoiceprintCaptureDecision],
    persisted: _PersistedCapture,
    db_path: Path,
) -> list[VoiceprintCaptureDecision]:
    """Merge actual sample ids and failures into a capture plan."""
    embedded_ids = list_any_embedded_sample_ids(db_path)
    updated: list[VoiceprintCaptureDecision] = []
    for decision in decisions:
        if decision.decision == "skip":
            updated.append(decision)
            continue
        failure = persisted.failures.get(decision.speaker_id)
        if failure is not None:
            updated.append(
                replace(
                    decision,
                    decision="failed",
                    reason="capture_failed",
                    error=failure,
                )
            )
            continue
        rows = persisted.rows_by_speaker.get(decision.speaker_id, [])
        if not rows:
            updated.append(
                replace(
                    decision,
                    decision="failed",
                    reason="no_samples_indexed",
                    error="No voiceprint samples were indexed.",
                )
            )
            continue
        first = rows[0]
        samples = [
            VoiceprintCapturedSample(
                sample_id=row.sample_id,
                public_id=row.public_id,
                clip_path=row.clip_path,
                embedded=row.sample_id in embedded_ids,
            )
            for row in rows
        ]
        updated.append(
            replace(
                decision,
                name=first.speaker_name,
                person_id=first.speaker_id,
                person_public_id=first.speaker_public_id,
                decision="captured",
                samples=samples,
            )
        )
    return updated


def _validate_capture_options(
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    *,
    min_samples: int = 10,
) -> None:
    """
    Validate voiceprint capture options.

    Args:
        sample_count: Requested clips per speaker.
        max_seconds: Requested maximum clip length.
        padding_seconds: Requested context padding.
        min_samples: Required existing sample count for ``only_needed``.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1.")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0.")
    if padding_seconds < 0:
        raise ValueError("padding_seconds must be >= 0.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")


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
    sample_count: int = 0


def _load_required_speaker_identities(
    speakers_dir: Path,
    db_path: Path,
    *,
    speaker_ids: set[int] | None = None,
) -> dict[int, _SpeakerIdentity]:
    """
    Load named speaker identities required for voiceprint references.

    Args:
        speakers_dir: Project speakers directory.
        db_path: Voiceprint SQLite path.
        speaker_ids: Optional explicit ids; unrelated mappings are not resolved.

    Returns:
        Project speaker id to identity mapping.
    """
    path = speakers_dir / "speaker_map.json"
    if not path.exists():
        raise FileNotFoundError(
            "Speaker mapping does not exist. Run meeting-asr project review first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {
        int(key): str(value)
        for key, value in payload.items()
        if speaker_ids is None or int(key) in speaker_ids
    }
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
    return {
        int(key): value
        for key, value in payload.items()
        if isinstance(value, int | str)
    }


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
            return _SpeakerIdentity(
                person.name,
                person.speaker_id,
                person.public_id,
                person.sample_count,
            )
        return _SpeakerIdentity(name, None, None)
    person = get_voiceprint_person(person_ref, db_path)
    if person is None:
        raise LookupError(
            f"speaker_person_map.json points to missing voiceprint person id {person_ref} for {name}."
        )
    if _normalize_identity_name(name) != _normalize_identity_name(person.name):
        raise ValueError(
            f"Project speaker {speaker_id} name {name!r} conflicts with canonical "
            f"voiceprint person name {person.name!r} ({person.public_id})."
        )
    return _SpeakerIdentity(
        person.name,
        person.speaker_id,
        person.public_id,
        person.sample_count,
    )


def _normalize_identity_name(name: str) -> str:
    """Normalize a display name using the registry's identity semantics."""
    return " ".join(name.strip().split()).casefold()


def _plan_voiceprint_speakers(
    clip_dir: Path,
    project_id: str,
    result: TranscriptResult,
    identities: dict[int, _SpeakerIdentity],
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    *,
    speaker_ids: set[int] | None,
    only_needed: bool,
    min_samples: int,
) -> tuple[list[VoiceprintSpeaker], list[VoiceprintCaptureDecision]]:
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
        Selected voiceprint speakers plus per-speaker decisions.
    """
    speakers: list[VoiceprintSpeaker] = []
    decisions: list[VoiceprintCaptureDecision] = []
    grouped = _speaker_segments_by_id(result)
    requested = _resolve_requested_speaker_ids(grouped, identities, speaker_ids)
    for speaker_id in requested:
        identity = _identified_speaker_identity(speaker_id, identities)
        if identity is None:
            continue
        decision, reason = _capture_decision(identity, only_needed, min_samples)
        decisions.append(
            VoiceprintCaptureDecision(
                speaker_id=speaker_id,
                name=identity.name,
                person_id=identity.person_id,
                person_public_id=identity.person_public_id,
                existing_sample_count=identity.sample_count,
                decision=decision,
                reason=reason,
            )
        )
        if decision == "skip":
            continue
        selected = select_voiceprint_segments(
            grouped[speaker_id], result.sentences, sample_count
        )
        clips = _build_clips(
            clip_dir, project_id, speaker_id, selected, max_seconds, padding_seconds
        )
        if clips:
            speakers.append(
                VoiceprintSpeaker(
                    speaker_id,
                    identity.name,
                    identity.person_id,
                    identity.person_public_id,
                    clips,
                )
            )
    return speakers, decisions


def _resolve_requested_speaker_ids(
    grouped: dict[int, list[SentenceSegment]],
    identities: dict[int, _SpeakerIdentity],
    speaker_ids: set[int] | None,
) -> list[int]:
    """Validate explicit ids or return every named speaker with usable audio."""
    if speaker_ids is None:
        return [
            speaker_id
            for speaker_id in sorted(grouped)
            if _identified_speaker_identity(speaker_id, identities) is not None
        ]
    if not speaker_ids:
        raise ValueError("At least one speaker id must be selected.")
    if any(speaker_id < 0 for speaker_id in speaker_ids):
        raise ValueError("speaker_id must be >= 0.")
    missing = sorted(speaker_ids.difference(grouped))
    if missing:
        joined = ", ".join(str(value) for value in missing)
        raise ValueError(
            f"Project speaker id(s) have no usable transcript segments: {joined}."
        )
    unnamed = [
        speaker_id
        for speaker_id in sorted(speaker_ids)
        if _identified_speaker_identity(speaker_id, identities) is None
    ]
    if unnamed:
        joined = ", ".join(str(value) for value in unnamed)
        raise ValueError(
            f"Project speaker id(s) are not confirmed and named: {joined}."
        )
    return sorted(speaker_ids)


def _capture_decision(
    identity: _SpeakerIdentity, only_needed: bool, min_samples: int
) -> tuple[str, str]:
    """Return selection decision and stable reason for one named speaker."""
    if not only_needed:
        return "capture", "selected"
    if identity.sample_count == 0:
        return "capture", "no_samples"
    if identity.sample_count < min_samples:
        return "capture", "below_min_samples"
    return "skip", "enough_samples"


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


def _speaker_segments_by_id(
    result: TranscriptResult,
) -> dict[int, list[SentenceSegment]]:
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


def _build_clips(
    clip_dir: Path,
    project_id: str,
    speaker_id: int,
    segments: list[ScoredVoiceprintSegment],
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
        clip = _build_clip(
            clip_dir,
            project_id,
            speaker_id,
            index,
            candidate,
            max_seconds,
            padding_seconds,
        )
        if clip.duration_seconds > 0:
            clips.append(clip)
    return clips


def _build_clip(
    clip_dir: Path,
    project_id: str,
    speaker_id: int,
    index: int,
    candidate: ScoredVoiceprintSegment,
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
    emit_progress(
        progress, "Writing voiceprint clips", total=_clip_count(speakers), completed=0
    )
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
            emit_progress(
                progress, f"Captured {speaker.name} voiceprint clip", advance=1
            )
        if captured_clips:
            captured_speakers.append(
                VoiceprintSpeaker(
                    speaker.speaker_id,
                    speaker.name,
                    speaker.person_id,
                    speaker.person_public_id,
                    captured_clips,
                )
            )
    return captured_speakers


def _with_audio_quality(
    clip: VoiceprintClip, quality: tuple[float | None, str]
) -> VoiceprintClip:
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
    except wave.Error, OSError, EOFError:
        return None, "audio=unknown"
    if sample_width != 2 or not frames:
        return None, "audio=unknown"
    samples = [
        value[0]
        for value in struct.iter_unpack("<h", frames[: len(frames) - len(frames) % 2])
    ]
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
                VoiceprintSpeaker(
                    speaker.speaker_id,
                    speaker.name,
                    speaker.person_id,
                    speaker.person_public_id,
                    clips,
                )
            )
    return selected_speakers


def _select_central_clips_for_speaker(
    clips: list[VoiceprintClip], target_sample_count: int
) -> list[VoiceprintClip]:
    """Return final clips closest to the candidate embedding centroid."""
    if len(clips) <= target_sample_count:
        return clips
    if len(clips) < 3:
        recommended = [clip for clip in clips if clip.recommended]
        return (recommended or clips)[:target_sample_count]
    ranked = _rank_clips_by_embedding_centrality(clips)
    if not ranked:
        ranked = [(clip, clip.selection_score) for clip in clips]
    selected = [
        clip
        for clip, _score in sorted(ranked, key=lambda item: item[1], reverse=True)[
            :target_sample_count
        ]
    ]
    return sorted(selected, key=lambda clip: clip.source_begin_time_ms)


def _rank_clips_by_embedding_centrality(
    clips: list[VoiceprintClip],
) -> list[tuple[VoiceprintClip, float]]:
    """Rank clips by distance to the candidate embedding centroid."""
    vectors: list[tuple[VoiceprintClip, list[float]]] = []
    for clip in clips:
        try:
            embedding_path = clip.path.with_name(f"{clip.path.stem}_embedding.wav")
            trim_embedding_audio_silence(clip.path, embedding_path)
            vectors.append(
                (
                    clip,
                    _normalize_vector(embed_audio_file(embedding_path, provider=None)),
                )
            )
        except Exception:
            return []
    centroid = _normalize_vector(
        [
            sum(values) / len(vectors)
            for values in zip(*(vector for _clip, vector in vectors))
        ]
    )
    return [
        (clip, _cosine(vector, centroid) + 0.05 * clip.selection_score)
        for clip, vector in vectors
    ]


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
            samples.append(
                _stored_sample(project_root, project_id, source, speaker, clip)
            )
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
