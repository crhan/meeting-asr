"""Project-level orchestration for per-sentence speaker reassignments.

When a Project Review user reassigns sentences in the timeline view, several
artifacts go stale:

1. ``asr/sentences.json`` and ``asr/sentences_corrected.json`` —
   ground-truth per-sentence speaker labels.
2. ``exports/transcript_speakers.txt`` — anonymous transcript with speaker
   labels.
3. Voiceprint samples in ``voiceprints.sqlite`` that were captured from a
   sentence whose attribution just changed.
4. ``speakers/speaker_matches.json`` — voiceprint match scores aggregated
   per speaker.

This module exposes a single entry point that applies the reassignments and
refreshes every dependent artifact in one pass so the save flow can call it
without scattering the rebuild logic across command code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.project_models import ProjectManifest
from app.postprocess import merge_adjacent_sentences, render_speaker_text
from app.models import TranscriptResult
from app.project_manager import (
    load_manifest,
    project_paths,
    save_manifest,
)
from app.speaker_labeling import (
    EmptySpeakerDeletionResult,
    SentenceReassignmentSpec,
    apply_sentence_reassignments,
    delete_empty_speaker_segments,
    load_transcript_result,
)
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers
from app.utils import safe_write_text
from app.voiceprint_models import DeletedVoiceprintSample, VoiceprintSampleRow
from app.voiceprint_store import (
    delete_voiceprint_samples_by_ids,
    get_voiceprint_db_path,
    list_voiceprint_samples_for_project,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_REMATCH_THRESHOLD = 0.75
DEFAULT_REMATCH_SAMPLE_COUNT = 2
DEFAULT_REMATCH_MAX_SECONDS = 12.0
DEFAULT_REMATCH_PADDING_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class SentenceReassignmentApplyResult:
    """Outcome of applying sentence reassignments and rebuilding downstream artifacts."""

    sentence_files: tuple[Path, ...]
    anonymous_transcript_path: Path | None
    deleted_samples: tuple[DeletedVoiceprintSample, ...]
    match_summary: SpeakerMatchSummary | None
    rematch_skipped_reason: str | None


@dataclass(frozen=True, slots=True)
class EmptySpeakerDeletionApplyResult:
    """Outcome of deleting empty speaker tracks and rebuilding transcript artifacts."""

    sentence_files: tuple[Path, ...]
    anonymous_transcript_path: Path | None
    deleted_sentence_count: int


def apply_project_sentence_reassignments(
    project_dir: Path,
    reassignments: Sequence[SentenceReassignmentSpec],
    *,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    rematch: bool = True,
) -> SentenceReassignmentApplyResult:
    """Persist sentence reassignments and refresh every dependent artifact.

    Args:
        project_dir: Project root.
        reassignments: Sentence reassignments to persist.
        store_dir: Voiceprint store directory; ``None`` resolves to the default.
        provider: Optional voiceprint embedding provider for the rematch.
        model: Optional voiceprint embedding model key for the rematch.
        rematch: Whether to rerun voiceprint matching after invalidation.

    Returns:
        Description of which files were rewritten, which voiceprint samples
        were dropped, and the new match summary (when rematch ran).
    """
    if not reassignments:
        return SentenceReassignmentApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_samples=(),
            match_summary=None,
            rematch_skipped_reason=None,
        )
    paths = project_paths(project_dir)
    sentence_files = apply_sentence_reassignments(paths.asr_dir, reassignments)
    manifest = load_manifest(paths.root)
    transcript_path = _refresh_anonymous_speaker_outputs(
        paths.asr_dir, paths.exports_dir
    )
    deleted_samples = _invalidate_overlapping_voiceprint_samples(
        manifest=manifest,
        reassignments=reassignments,
        store_dir=store_dir,
    )
    match_summary, rematch_skipped_reason = _maybe_rematch_speakers(
        project_dir=paths.root,
        store_dir=store_dir,
        provider=provider,
        model=model,
        rematch=rematch,
        manifest=manifest,
    )
    return SentenceReassignmentApplyResult(
        sentence_files=sentence_files,
        anonymous_transcript_path=transcript_path,
        deleted_samples=tuple(deleted_samples),
        match_summary=match_summary,
        rematch_skipped_reason=rematch_skipped_reason,
    )


def apply_project_empty_speaker_deletions(
    project_dir: Path,
    speaker_ids: Sequence[int],
) -> EmptySpeakerDeletionApplyResult:
    """Delete empty speaker tracks and refresh anonymous transcript output."""
    if not speaker_ids:
        return EmptySpeakerDeletionApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_sentence_count=0,
        )
    paths = project_paths(project_dir)
    result: EmptySpeakerDeletionResult = delete_empty_speaker_segments(
        paths.asr_dir, speaker_ids
    )
    transcript_path = (
        _refresh_anonymous_speaker_outputs(paths.asr_dir, paths.exports_dir)
        if result.sentence_files
        else None
    )
    return EmptySpeakerDeletionApplyResult(
        sentence_files=result.sentence_files,
        anonymous_transcript_path=transcript_path,
        deleted_sentence_count=result.deleted_sentence_count,
    )


def _refresh_anonymous_speaker_outputs(asr_dir: Path, exports_dir: Path) -> Path:
    """Rewrite ``transcript_speakers.txt`` from the updated sentence file.

    Args:
        asr_dir: Project ``asr/`` directory.
        exports_dir: Project ``exports/`` directory.

    Returns:
        Path of the rewritten transcript.
    """
    result = load_transcript_result(asr_dir / "sentences.json")
    merged = TranscriptResult(
        result.full_text,
        merge_adjacent_sentences(result.sentences),
        result.detected_speakers,
    )
    return safe_write_text(
        exports_dir / "transcript_speakers.txt", render_speaker_text(merged)
    )


def _invalidate_overlapping_voiceprint_samples(
    *,
    manifest: ProjectManifest,
    reassignments: Sequence[SentenceReassignmentSpec],
    store_dir: Path | None,
) -> list[DeletedVoiceprintSample]:
    """Delete voiceprint samples whose audio now belongs to another speaker.

    A sample is considered stale when its source time range overlaps a
    reassigned sentence and its ``project_speaker_id`` matches the sentence's
    original speaker. Samples on other speakers, projects, or time ranges are
    untouched.

    Args:
        manifest: Loaded project manifest (used for ``project_id``).
        reassignments: Reassignments being applied.
        store_dir: Voiceprint store directory.

    Returns:
        Deleted sample summaries (empty when nothing matched).
    """
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_voiceprint_samples_for_project(manifest.project_id, db_path)
    if not samples:
        return []
    stale_ids = _stale_sample_ids(samples, reassignments)
    if not stale_ids:
        return []
    return delete_voiceprint_samples_by_ids(
        stale_ids, db_path=db_path, delete_clips=True
    )


def _stale_sample_ids(
    samples: Sequence[VoiceprintSampleRow],
    reassignments: Sequence[SentenceReassignmentSpec],
) -> list[int]:
    """Return sample row ids whose audio falls inside a reassignment.

    Args:
        samples: Project voiceprint samples.
        reassignments: Reassignments being applied.

    Returns:
        Stale sample row ids in registry order.
    """
    by_speaker: dict[int, list[tuple[int, int]]] = {}
    for item in reassignments:
        if item.original_speaker_id is None:
            continue
        by_speaker.setdefault(int(item.original_speaker_id), []).append(
            (int(item.begin_time_ms), int(item.end_time_ms)),
        )
    stale: list[int] = []
    for sample in samples:
        ranges = by_speaker.get(int(sample.project_speaker_id))
        if not ranges:
            continue
        if any(
            _ranges_overlap(
                sample.source_begin_time_ms, sample.source_end_time_ms, begin, end
            )
            for begin, end in ranges
        ):
            stale.append(int(sample.sample_id))
    return stale


def _ranges_overlap(
    left_begin: int, left_end: int, right_begin: int, right_end: int
) -> bool:
    """Return whether two half-open time ranges overlap."""
    return left_begin < right_end and right_begin < left_end


def _maybe_rematch_speakers(
    *,
    project_dir: Path,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
    rematch: bool,
    manifest: ProjectManifest,
) -> tuple[SpeakerMatchSummary | None, str | None]:
    """Rerun voiceprint matching when reassignments changed speaker grouping.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Optional voiceprint embedding provider.
        model: Optional voiceprint embedding model key.
        rematch: Whether the caller requested a rematch.
        manifest: Loaded project manifest (used for source-path metadata).

    Returns:
        ``(summary, None)`` after a successful rematch, or ``(None, reason)``
        when rematch was skipped or failed. The reason is surfaced to the user
        via the save modal.
    """
    if not rematch:
        return None, "rematch disabled by caller"
    try:
        summary = match_project_speakers(
            project_dir,
            store_dir=store_dir,
            provider=provider,
            model=model,
            threshold=DEFAULT_REMATCH_THRESHOLD,
            sample_count=DEFAULT_REMATCH_SAMPLE_COUNT,
            max_seconds=DEFAULT_REMATCH_MAX_SECONDS,
            padding_seconds=DEFAULT_REMATCH_PADDING_SECONDS,
            progress=None,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Voiceprint rematch after reassignment failed: %s", exc)
        return None, str(exc)
    # Refresh manifest reference so callers reading it next see speaker_matches.
    manifest.speakers["matches"] = "speakers/speaker_matches.json"
    save_manifest(project_dir, manifest)
    return summary, None


__all__ = [
    "DEFAULT_REMATCH_MAX_SECONDS",
    "DEFAULT_REMATCH_PADDING_SECONDS",
    "DEFAULT_REMATCH_SAMPLE_COUNT",
    "DEFAULT_REMATCH_THRESHOLD",
    "SentenceReassignmentApplyResult",
    "apply_project_sentence_reassignments",
]
