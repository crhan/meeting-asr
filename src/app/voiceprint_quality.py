"""Voiceprint sample quality diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings

VOICEPRINT_SAMPLE_STATUS_ACTIVE = "active"
VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE = "verified-active"
VOICEPRINT_SAMPLE_STATUS_QUARANTINED = "quarantined"
VOICEPRINT_SAMPLE_STATUS_VERIFIED_QUARANTINED = "verified-quarantined"
VOICEPRINT_SAMPLE_STATUS_REJECTED = "rejected"
VOICEPRINT_SAMPLE_STATUSES = (
    VOICEPRINT_SAMPLE_STATUS_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_QUARANTINED,
    VOICEPRINT_SAMPLE_STATUS_VERIFIED_QUARANTINED,
    VOICEPRINT_SAMPLE_STATUS_REJECTED,
)
VOICEPRINT_MATCHING_SAMPLE_STATUSES = frozenset(
    {
        VOICEPRINT_SAMPLE_STATUS_ACTIVE,
        VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE,
    }
)
DEFAULT_CRITICAL_SCORE = 0.60
DEFAULT_WARNING_SCORE = 0.70
DEFAULT_MIN_CLUSTER_SIZE = 3


@dataclass(frozen=True, slots=True)
class VoiceprintQualitySample:
    """One sample quality score inside a person's embedding cluster."""

    sample_id: int
    sample_public_id: str
    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    clip_path: Path
    project_id: str
    source_begin_time_ms: int
    source_end_time_ms: int
    transcript_text: str
    status: str
    score: float | None
    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class VoiceprintQualityProject:
    """Quality diagnostics grouped by source project."""

    project_id: str
    sample_count: int
    matching_sample_count: int
    suspicious_count: int
    critical_count: int
    mean_score: float | None
    min_score: float | None


@dataclass(frozen=True, slots=True)
class VoiceprintQualityNeighbor:
    """Closest other person by voiceprint centroid."""

    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    score: float


@dataclass(frozen=True, slots=True)
class VoiceprintQualityPerson:
    """Quality diagnostics for one voiceprint person."""

    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    sample_count: int
    active_sample_count: int
    mean_score: float | None
    stdev_score: float | None
    samples: tuple[VoiceprintQualitySample, ...]
    projects: tuple[VoiceprintQualityProject, ...] = ()
    closest_people: tuple[VoiceprintQualityNeighbor, ...] = ()

    @property
    def suspicious_count(self) -> int:
        """Return number of active samples marked warning or critical."""
        return sum(
            1
            for item in self.samples
            if item.status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
            and item.label in {"warning", "critical"}
        )

    @property
    def critical_count(self) -> int:
        """Return number of active samples marked critical."""
        return sum(
            1
            for item in self.samples
            if item.status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
            and item.label == "critical"
        )


@dataclass(frozen=True, slots=True)
class VoiceprintQualityReport:
    """Quality diagnostics for the voiceprint library."""

    db_path: Path
    model: str
    people: tuple[VoiceprintQualityPerson, ...]

    @property
    def sample_count(self) -> int:
        """Return total sample count in the report."""
        return sum(person.sample_count for person in self.people)

    @property
    def suspicious_count(self) -> int:
        """Return total active suspicious sample count."""
        return sum(person.suspicious_count for person in self.people)

    @property
    def critical_count(self) -> int:
        """Return total active critical sample count."""
        return sum(person.critical_count for person in self.people)


def analyze_voiceprint_quality(
    *,
    store_dir: Path | None,
    speaker: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    critical_score: float = DEFAULT_CRITICAL_SCORE,
    warning_score: float = DEFAULT_WARNING_SCORE,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> VoiceprintQualityReport:
    """
    Analyze sample-level quality from stored voiceprint embeddings.

    Args:
        store_dir: Optional voiceprint store directory.
        speaker: Optional person name or public id filter.
        provider: Optional embedding provider override.
        model: Optional embedding model key.
        critical_score: Absolute score below which a sample is critical.
        warning_score: Absolute score below which a sample is warning.
        min_cluster_size: Minimum active sample count for outlier scoring.

    Returns:
        Voiceprint quality report.
    """
    _provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    db_path = get_voiceprint_db_path(store_dir)
    rows = list_voiceprint_embeddings(resolved_model, db_path, include_inactive=True)
    grouped = _group_rows(rows, speaker)
    centroids = {
        speaker_id: _matching_centroid(rows, min_cluster_size=min_cluster_size)
        for speaker_id, rows in grouped.items()
    }
    people = tuple(
        replace(
            _person_quality(
                rows,
                critical_score=critical_score,
                warning_score=warning_score,
                min_cluster_size=min_cluster_size,
            ),
            closest_people=_closest_people(
                rows[0].speaker_id, centroids=centroids, grouped=grouped
            ),
        )
        for rows in grouped.values()
    )
    return VoiceprintQualityReport(
        db_path, resolved_model, tuple(sorted(people, key=_person_sort_key))
    )


def _group_rows(rows: list[object], speaker: str | None) -> dict[int, list[object]]:
    """Group embedding rows by speaker id, optionally filtering one person."""
    grouped: dict[int, list[object]] = defaultdict(list)
    normalized_filter = speaker.strip().casefold() if speaker else None
    for row in rows:
        if normalized_filter and normalized_filter not in {
            row.speaker_public_id.casefold(),
            row.speaker_name.casefold(),
        }:
            continue
        grouped[row.speaker_id].append(row)
    return grouped


def _person_quality(
    rows: list[object],
    *,
    critical_score: float,
    warning_score: float,
    min_cluster_size: int,
) -> VoiceprintQualityPerson:
    """Build quality diagnostics for one person."""
    first = rows[0]
    active_rows = _matching_rows(rows)
    centroid = _matching_centroid(rows, min_cluster_size=min_cluster_size)
    scores = (
        [_cosine(_normalize(row.vector), centroid) for row in active_rows]
        if centroid is not None
        else []
    )
    mean_score = statistics.mean(scores) if scores else None
    stdev_score = statistics.pstdev(scores) if len(scores) > 1 else None
    samples = tuple(
        _sample_quality(
            row,
            centroid=centroid,
            mean_score=mean_score,
            stdev_score=stdev_score,
            critical_score=critical_score,
            warning_score=warning_score,
        )
        for row in rows
    )
    return VoiceprintQualityPerson(
        first.speaker_id,
        first.speaker_public_id,
        first.speaker_name,
        len(rows),
        len(active_rows),
        mean_score,
        stdev_score,
        tuple(sorted(samples, key=_sample_sort_key)),
        _project_quality(samples),
    )


def _sample_quality(
    row: object,
    *,
    centroid: list[float] | None,
    mean_score: float | None,
    stdev_score: float | None,
    critical_score: float,
    warning_score: float,
) -> VoiceprintQualitySample:
    """Score one sample against the active cluster centroid."""
    if row.sample_status not in VOICEPRINT_MATCHING_SAMPLE_STATUSES:
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_QUARANTINED:
            return _quality_sample(
                row,
                None,
                "verified-disabled",
                "identity confirmed; excluded from matching",
            )
        return _quality_sample(
            row, None, row.sample_status, f"status={row.sample_status}"
        )
    if centroid is None:
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
            return _quality_sample(
                row,
                None,
                "unknown",
                "identity confirmed; need at least 3 matching samples",
            )
        return _quality_sample(row, None, "unknown", "need at least 3 active samples")
    score = _cosine(_normalize(row.vector), centroid)
    statistical_limit = (
        None
        if mean_score is None or stdev_score is None
        else mean_score - 2 * stdev_score
    )
    if score < critical_score:
        reason = f"score<{critical_score:.2f}"
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
            reason = f"identity confirmed; {reason}"
        return _quality_sample(row, score, "critical", reason)
    if statistical_limit is not None and score < statistical_limit:
        reason = "statistical outlier"
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
            reason = f"identity confirmed; {reason}"
        return _quality_sample(row, score, "critical", reason)
    if score < warning_score:
        reason = f"score<{warning_score:.2f}"
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
            reason = f"identity confirmed; {reason}"
        return _quality_sample(row, score, "warning", reason)
    if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
        return _quality_sample(
            row, score, "ok", "identity confirmed; cluster-consistent"
        )
    return _quality_sample(row, score, "ok", "cluster-consistent")


def _quality_sample(
    row: object, score: float | None, label: str, reason: str
) -> VoiceprintQualitySample:
    """Build one quality sample row."""
    return VoiceprintQualitySample(
        row.sample_id,
        row.sample_public_id,
        row.speaker_id,
        row.speaker_public_id,
        row.speaker_name,
        row.clip_path,
        row.project_id,
        row.source_begin_time_ms,
        row.source_end_time_ms,
        row.transcript_text,
        row.sample_status,
        score,
        label,
        reason,
    )


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Return normalized mean vector."""
    return _normalize([sum(values) / len(vectors) for values in zip(*vectors)])


def _matching_rows(rows: list[object]) -> list[object]:
    """Return samples that participate in voiceprint matching."""
    return [
        row for row in rows if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
    ]


def _matching_centroid(
    rows: list[object], *, min_cluster_size: int
) -> list[float] | None:
    """Return a person's matching centroid when enough samples exist."""
    active_rows = _matching_rows(rows)
    if len(active_rows) < min_cluster_size:
        return None
    return _centroid([row.vector for row in active_rows])


def _project_quality(
    samples: tuple[VoiceprintQualitySample, ...],
) -> tuple[VoiceprintQualityProject, ...]:
    """Aggregate sample diagnostics by source project."""
    grouped: dict[str, list[VoiceprintQualitySample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.project_id].append(sample)
    projects: list[VoiceprintQualityProject] = []
    for project_id, rows in grouped.items():
        matching = [
            row for row in rows if row.status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
        ]
        scores = [row.score for row in matching if row.score is not None]
        projects.append(
            VoiceprintQualityProject(
                project_id=project_id,
                sample_count=len(rows),
                matching_sample_count=len(matching),
                suspicious_count=sum(
                    1 for row in matching if row.label in {"warning", "critical"}
                ),
                critical_count=sum(1 for row in matching if row.label == "critical"),
                mean_score=statistics.mean(scores) if scores else None,
                min_score=min(scores) if scores else None,
            )
        )
    return tuple(sorted(projects, key=_project_sort_key))


def _closest_people(
    speaker_id: int,
    *,
    centroids: dict[int, list[float] | None],
    grouped: dict[int, list[object]],
    limit: int = 3,
) -> tuple[VoiceprintQualityNeighbor, ...]:
    """Return closest other matching centroids."""
    centroid = centroids.get(speaker_id)
    if centroid is None:
        return ()
    scored: list[VoiceprintQualityNeighbor] = []
    for other_id, other_centroid in centroids.items():
        if other_id == speaker_id or other_centroid is None:
            continue
        first = grouped[other_id][0]
        scored.append(
            VoiceprintQualityNeighbor(
                speaker_id=first.speaker_id,
                speaker_public_id=first.speaker_public_id,
                speaker_name=first.speaker_name,
                score=_cosine(centroid, other_centroid),
            )
        )
    return tuple(sorted(scored, key=lambda item: item.score, reverse=True)[:limit])


def _normalize(vector: list[float]) -> list[float]:
    """Return a unit vector, preserving zero vectors."""
    magnitude = math.sqrt(sum(item * item for item in vector))
    if magnitude == 0:
        return vector
    return [item / magnitude for item in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    return sum(a * b for a, b in zip(left, right))


def _person_sort_key(person: VoiceprintQualityPerson) -> tuple[int, str]:
    """Sort people with suspicious samples first."""
    return (
        -person.critical_count,
        -person.suspicious_count,
        person.speaker_name.casefold(),
    )


def _sample_sort_key(sample: VoiceprintQualitySample) -> tuple[int, float, str]:
    """Sort suspicious samples first, then by score."""
    severity = {"critical": 0, "warning": 1, "ok": 2, "verified": 3, "unknown": 4}.get(
        sample.label, 5
    )
    score = sample.score if sample.score is not None else 999.0
    return (severity, score, sample.sample_public_id)


def _project_sort_key(project: VoiceprintQualityProject) -> tuple[int, int, float, str]:
    """Sort projects by actionable quality risk."""
    min_score = project.min_score if project.min_score is not None else 999.0
    return (
        -project.critical_count,
        -project.suspicious_count,
        min_score,
        project.project_id,
    )
