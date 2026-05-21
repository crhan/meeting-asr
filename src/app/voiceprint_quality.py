"""Voiceprint sample quality diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings

VOICEPRINT_SAMPLE_STATUS_ACTIVE = "active"
VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE = "verified-active"
VOICEPRINT_SAMPLE_STATUS_QUARANTINED = "quarantined"
VOICEPRINT_SAMPLE_STATUS_REJECTED = "rejected"
VOICEPRINT_SAMPLE_STATUSES = (
    VOICEPRINT_SAMPLE_STATUS_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_QUARANTINED,
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

    @property
    def suspicious_count(self) -> int:
        """Return number of active samples marked warning or critical."""
        return sum(
            1
            for item in self.samples
            if item.status == VOICEPRINT_SAMPLE_STATUS_ACTIVE
            and item.label in {"warning", "critical"}
        )

    @property
    def critical_count(self) -> int:
        """Return number of active samples marked critical."""
        return sum(
            1
            for item in self.samples
            if item.status == VOICEPRINT_SAMPLE_STATUS_ACTIVE
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
    people = tuple(
        _person_quality(
            rows,
            critical_score=critical_score,
            warning_score=warning_score,
            min_cluster_size=min_cluster_size,
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
    active_rows = [
        row for row in rows if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
    ]
    centroid = (
        _centroid([row.vector for row in active_rows])
        if len(active_rows) >= min_cluster_size
        else None
    )
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
        return _quality_sample(
            row, None, row.sample_status, f"status={row.sample_status}"
        )
    if centroid is None:
        if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
            return _quality_sample(row, None, "verified", "human verified active")
        return _quality_sample(row, None, "unknown", "need at least 3 active samples")
    score = _cosine(_normalize(row.vector), centroid)
    if row.sample_status == VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE:
        return _quality_sample(row, score, "verified", "human verified active")
    statistical_limit = (
        None
        if mean_score is None or stdev_score is None
        else mean_score - 2 * stdev_score
    )
    if score < critical_score:
        return _quality_sample(row, score, "critical", f"score<{critical_score:.2f}")
    if statistical_limit is not None and score < statistical_limit:
        return _quality_sample(row, score, "critical", "statistical outlier")
    if score < warning_score:
        return _quality_sample(row, score, "warning", f"score<{warning_score:.2f}")
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
