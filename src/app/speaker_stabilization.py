"""Automatic speaker assignment stabilization for project runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.sentence_reassignment import SentenceReassignmentApplyResult, apply_project_sentence_reassignments
from app.speaker_cluster_quality import (
    SpeakerClusterQualitySummary,
    SpeakerClusterSampleScore,
    analyze_project_speaker_clusters,
)
from app.speaker_labeling import SentenceReassignmentSpec
from app.speaker_matching import SpeakerMatchSummary
from app.speaker_sample_matching import (
    DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
    DEFAULT_IDENTITY_CONFLICT_MARGIN,
    DEFAULT_SAMPLE_IDENTITY_THRESHOLD,
    SpeakerSampleMatchSummary,
    match_project_speaker_samples,
)
from app.project_manager import apply_project_speakers
from app.voiceprint_quality import DEFAULT_CRITICAL_SCORE, DEFAULT_WARNING_SCORE

DEFAULT_STABILIZATION_ITERATIONS = 2
DEFAULT_STABILIZATION_SAMPLE_WORKERS = 4
DEFAULT_STABILIZATION_CLUSTER_SAMPLE_COUNT = 40
DEFAULT_STABILIZATION_MATCH_MAX_SECONDS = 12.0
DEFAULT_STABILIZATION_MATCH_PADDING_SECONDS = 0.5
DEFAULT_STABILIZATION_CLUSTER_SAME_SPEAKER_THRESHOLD = 0.60
DEFAULT_STABILIZATION_CLUSTER_MERGE_THRESHOLD = 0.62


@dataclass(frozen=True, slots=True)
class SpeakerStabilizationIteration:
    """Result of one automatic sentence-reassignment stabilization pass."""

    index: int
    reassignments: tuple[SentenceReassignmentSpec, ...]
    apply_result: SentenceReassignmentApplyResult | None
    cluster_summary: SpeakerClusterQualitySummary
    sample_summary: SpeakerSampleMatchSummary


@dataclass(frozen=True, slots=True)
class SpeakerStabilizationSummary:
    """Full stabilization result after repeated diagnostics and reassignment."""

    iterations: tuple[SpeakerStabilizationIteration, ...]

    @property
    def reassignment_count(self) -> int:
        """Return the total number of persisted sentence reassignments."""
        return sum(len(iteration.reassignments) for iteration in self.iterations)

    @property
    def final_match_summary(self) -> SpeakerMatchSummary | None:
        """Return the latest aggregate match summary produced by reassignment."""
        for iteration in reversed(self.iterations):
            if iteration.apply_result and iteration.apply_result.match_summary:
                return iteration.apply_result.match_summary
        return None


def stabilize_project_speakers(
    project_dir: Path,
    *,
    store_dir: Path | None,
    model: str | None,
    iterations: int = DEFAULT_STABILIZATION_ITERATIONS,
    sample_workers: int = DEFAULT_STABILIZATION_SAMPLE_WORKERS,
    progress: CliProgressReporter | None = None,
) -> SpeakerStabilizationSummary:
    """
    Repeatedly reassign sentence-level identity conflicts and refresh scores.

    Args:
        project_dir: Project root directory.
        store_dir: Optional voiceprint store directory.
        model: Optional voiceprint embedding model key.
        iterations: Number of check/apply/refresh passes.
        sample_workers: Parallel workers for per-sentence sample matching.
        progress: Optional progress reporter.

    Returns:
        Stabilization summary for every executed pass.
    """
    results: list[SpeakerStabilizationIteration] = []
    total = max(0, iterations)
    for index in range(1, total + 1):
        emit_progress(
            progress,
            f"Speaker stabilization pass {index}/{total}",
            total=total,
            completed=index - 1,
        )
        cluster_summary, sample_summary = _refresh_diagnostics(
            project_dir,
            store_dir=store_dir,
            model=model,
            sample_workers=sample_workers,
            progress=progress,
        )
        reassignments = tuple(_sentence_reassignments(sample_summary, cluster_summary))
        apply_result = None
        if reassignments:
            emit_progress(progress, f"Applying {len(reassignments)} sentence speaker reassignment(s)")
            apply_result = apply_project_sentence_reassignments(
                project_dir,
                reassignments,
                store_dir=store_dir,
                provider=None,
                model=model,
                rematch=True,
            )
            _apply_latest_match_names(project_dir, apply_result.match_summary)
            cluster_summary, sample_summary = _refresh_diagnostics(
                project_dir,
                store_dir=store_dir,
                model=model,
                sample_workers=sample_workers,
                progress=progress,
            )
        results.append(
            SpeakerStabilizationIteration(
                index,
                reassignments,
                apply_result,
                cluster_summary,
                sample_summary,
            )
        )
        emit_progress(progress, f"Speaker stabilization pass {index}/{total} complete", completed=index, total=total)
    return SpeakerStabilizationSummary(tuple(results))


def _refresh_diagnostics(
    project_dir: Path,
    *,
    store_dir: Path | None,
    model: str | None,
    sample_workers: int,
    progress: CliProgressReporter | None,
) -> tuple[SpeakerClusterQualitySummary, SpeakerSampleMatchSummary]:
    """Refresh speaker cluster and per-sentence identity reports."""
    cluster_summary = analyze_project_speaker_clusters(
        project_dir,
        provider=None,
        model=model,
        sample_count=DEFAULT_STABILIZATION_CLUSTER_SAMPLE_COUNT,
        max_seconds=DEFAULT_STABILIZATION_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_STABILIZATION_MATCH_PADDING_SECONDS,
        score_all_segments=True,
        same_speaker_threshold=DEFAULT_STABILIZATION_CLUSTER_SAME_SPEAKER_THRESHOLD,
        merge_speaker_threshold=DEFAULT_STABILIZATION_CLUSTER_MERGE_THRESHOLD,
        warning_score=DEFAULT_WARNING_SCORE,
        critical_score=DEFAULT_CRITICAL_SCORE,
        write_report=True,
        progress=progress,
    )
    sample_summary = match_project_speaker_samples(
        project_dir,
        store_dir=store_dir,
        provider=None,
        model=model,
        threshold=DEFAULT_SAMPLE_IDENTITY_THRESHOLD,
        conflict_margin=DEFAULT_IDENTITY_CONFLICT_MARGIN,
        ambiguous_margin=DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
        max_seconds=DEFAULT_STABILIZATION_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_STABILIZATION_MATCH_PADDING_SECONDS,
        write_report=True,
        workers=sample_workers,
        progress=progress,
    )
    return cluster_summary, sample_summary


def _sentence_reassignments(
    sample_summary: SpeakerSampleMatchSummary,
    cluster_summary: SpeakerClusterQualitySummary,
) -> list[SentenceReassignmentSpec]:
    """Convert strong per-sentence conflicts into project speaker reassignments."""
    target_by_person = _target_speaker_by_person(sample_summary)
    cluster_samples = _cluster_sample_index(cluster_summary)
    seen: set[tuple[int | None, int, int]] = set()
    reassignments: list[SentenceReassignmentSpec] = []
    for report in sample_summary.reports:
        for sample in report.samples:
            if sample.status != "identity-conflict" or sample.best_other_person_id is None:
                continue
            target_speaker_id = target_by_person.get(sample.best_other_person_id)
            if target_speaker_id is None or target_speaker_id == sample.speaker_id:
                continue
            identity = (sample.sentence_id, sample.begin_time_ms, sample.end_time_ms)
            if identity in seen:
                continue
            cluster_sample = cluster_samples.get((sample.speaker_id, *identity))
            if cluster_sample is not None and not _cluster_allows_target(cluster_sample, target_speaker_id):
                continue
            seen.add(identity)
            reassignments.append(
                SentenceReassignmentSpec(
                    sentence_id=sample.sentence_id,
                    begin_time_ms=sample.begin_time_ms,
                    end_time_ms=sample.end_time_ms,
                    new_speaker_id=target_speaker_id,
                    original_speaker_id=sample.speaker_id,
                )
            )
    return reassignments


def _target_speaker_by_person(summary: SpeakerSampleMatchSummary) -> dict[int, int]:
    """Return voiceprint person id to project speaker id for assigned speakers."""
    targets: dict[int, int] = {}
    for report in summary.reports:
        if report.assigned_person_id is not None:
            targets.setdefault(report.assigned_person_id, report.speaker_id)
    return targets


def _cluster_sample_index(
    summary: SpeakerClusterQualitySummary,
) -> dict[tuple[int, int | None, int, int], SpeakerClusterSampleScore]:
    """Index cluster sample scores by current speaker and sentence identity."""
    indexed: dict[tuple[int, int | None, int, int], SpeakerClusterSampleScore] = {}
    for report in summary.reports:
        for sample in report.samples:
            indexed[(report.speaker_id, sample.sentence_id, sample.begin_time_ms, sample.end_time_ms)] = sample
    return indexed


def _cluster_allows_target(sample: SpeakerClusterSampleScore, target_speaker_id: int) -> bool:
    """Return whether cluster diagnostics do not contradict the identity target."""
    if sample.nearest_speaker_id == target_speaker_id:
        return True
    return sample.status not in {"conflict", "ambiguous"}


def _apply_latest_match_names(project_dir: Path, summary: SpeakerMatchSummary | None) -> None:
    """Refresh named transcript outputs from the latest accepted aggregate matches."""
    if summary is None or not summary.accepted_mapping:
        return
    apply_project_speakers(
        project_dir,
        summary.accepted_mapping,
        person_mapping=summary.accepted_person_mapping,
        person_public_mapping=summary.accepted_person_public_mapping,
    )


__all__ = [
    "DEFAULT_STABILIZATION_ITERATIONS",
    "DEFAULT_STABILIZATION_SAMPLE_WORKERS",
    "SpeakerStabilizationIteration",
    "SpeakerStabilizationSummary",
    "stabilize_project_speakers",
]
