"""Store-driven calibration evidence for speaker match thresholds.

The acceptance thresholds in :mod:`app.speaker_pipeline_params` were tuned on
early real-world projects; as the voiceprint library grows they can be
re-checked against the store itself. This module computes, from the embedded
matching-pool samples only:

- the **genuine** score distribution — each sample against its own person's
  leave-one-out centroid (what a correct match looks like), and
- the **impostor** score distribution — each sample against its best OTHER
  person centroid (what a wrong match looks like),

then sweeps candidate thresholds to report the equal-error point and the
lowest threshold holding impostor acceptance under 1%. Read-only; nothing is
tuned automatically — the numbers are evidence for a human deciding whether
``DEFAULT_MATCH_THRESHOLD`` still fits their library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from app.speaker_pipeline_params import DEFAULT_MATCH_THRESHOLD
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings

MIN_PERSON_SAMPLES = 2
SWEEP_START = 0.30
SWEEP_STOP = 0.95
SWEEP_STEP = 0.005
IMPOSTOR_RATE_TARGET = 0.01


@dataclass(frozen=True, slots=True)
class ScoreDistribution:
    """Summary statistics for one calibration score population."""

    count: int
    minimum: float
    p5: float
    median: float
    p95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class VoiceprintCalibrationReport:
    """Calibration evidence computed from the voiceprint store."""

    model: str
    person_count: int
    scored_person_count: int
    sample_count: int
    genuine: ScoreDistribution | None
    impostor: ScoreDistribution | None
    eer_threshold: float | None
    eer_rate: float | None
    low_impostor_threshold: float | None
    current_threshold: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready payload."""
        return {
            "model": self.model,
            "person_count": self.person_count,
            "scored_person_count": self.scored_person_count,
            "sample_count": self.sample_count,
            "genuine": _distribution_payload(self.genuine),
            "impostor": _distribution_payload(self.impostor),
            "eer_threshold": self.eer_threshold,
            "eer_rate": self.eer_rate,
            "low_impostor_threshold": self.low_impostor_threshold,
            "current_threshold": self.current_threshold,
            "warnings": list(self.warnings),
        }


def calibrate_voiceprint_thresholds(
    *,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> VoiceprintCalibrationReport:
    """
    Compute threshold calibration evidence from the voiceprint store.

    Args:
        store_dir: Optional voiceprint store directory.
        provider: Optional embedding provider override.
        model: Optional embedding model override.

    Returns:
        Calibration report (read-only; the store is never modified).
    """
    _resolved_provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    db_path = get_voiceprint_db_path(store_dir)
    rows = list_voiceprint_embeddings(resolved_model, db_path)
    vectors_by_person: dict[int, list[list[float]]] = {}
    names_by_person: dict[int, str] = {}
    for row in rows:
        vectors_by_person.setdefault(row.speaker_id, []).append(
            _normalize(row.vector)
        )
        names_by_person[row.speaker_id] = row.speaker_name
    warnings: list[str] = []
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    centroids = {
        person_id: _normalize(_mean(vectors))
        for person_id, vectors in vectors_by_person.items()
    }
    scored_people = 0
    for person_id, vectors in vectors_by_person.items():
        other_centroids = [
            centroid for key, centroid in centroids.items() if key != person_id
        ]
        for index, vector in enumerate(vectors):
            if other_centroids:
                impostor_scores.append(
                    max(_cosine(vector, centroid) for centroid in other_centroids)
                )
            if len(vectors) >= MIN_PERSON_SAMPLES:
                rest = vectors[:index] + vectors[index + 1 :]
                genuine_scores.append(_cosine(vector, _normalize(_mean(rest))))
        if len(vectors) >= MIN_PERSON_SAMPLES:
            scored_people += 1
        else:
            warnings.append(
                f"{names_by_person[person_id]}: only {len(vectors)} embedded "
                "sample(s); excluded from the genuine distribution"
            )
    if len(vectors_by_person) < 2:
        warnings.append(
            "fewer than 2 people have embedded samples; impostor distribution "
            "is unavailable"
        )
    eer_threshold, eer_rate = _equal_error_threshold(genuine_scores, impostor_scores)
    return VoiceprintCalibrationReport(
        model=resolved_model,
        person_count=len(vectors_by_person),
        scored_person_count=scored_people,
        sample_count=len(rows),
        genuine=_distribution(genuine_scores),
        impostor=_distribution(impostor_scores),
        eer_threshold=eer_threshold,
        eer_rate=eer_rate,
        low_impostor_threshold=_low_impostor_threshold(impostor_scores),
        current_threshold=DEFAULT_MATCH_THRESHOLD,
        warnings=tuple(warnings),
    )


def _equal_error_threshold(
    genuine: list[float], impostor: list[float]
) -> tuple[float | None, float | None]:
    """Sweep thresholds and return the equal-error point."""
    if not genuine or not impostor:
        return None, None
    best_threshold: float | None = None
    best_gap = math.inf
    best_rate: float | None = None
    threshold = SWEEP_START
    while threshold <= SWEEP_STOP + 1e-9:
        far = sum(1 for score in impostor if score >= threshold) / len(impostor)
        frr = sum(1 for score in genuine if score < threshold) / len(genuine)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_threshold = round(threshold, 3)
            best_rate = round((far + frr) / 2, 4)
        threshold += SWEEP_STEP
    return best_threshold, best_rate


def _low_impostor_threshold(impostor: list[float]) -> float | None:
    """Return the lowest threshold keeping impostor acceptance <= 1%."""
    if not impostor:
        return None
    threshold = SWEEP_START
    while threshold <= SWEEP_STOP + 1e-9:
        far = sum(1 for score in impostor if score >= threshold) / len(impostor)
        if far <= IMPOSTOR_RATE_TARGET:
            return round(threshold, 3)
        threshold += SWEEP_STEP
    return None


def _distribution(scores: list[float]) -> ScoreDistribution | None:
    """Summarize one score population."""
    if not scores:
        return None
    ordered = sorted(scores)
    return ScoreDistribution(
        count=len(ordered),
        minimum=round(ordered[0], 3),
        p5=round(_percentile(ordered, 0.05), 3),
        median=round(_percentile(ordered, 0.5), 3),
        p95=round(_percentile(ordered, 0.95), 3),
        maximum=round(ordered[-1], 3),
    )


def _distribution_payload(
    distribution: ScoreDistribution | None,
) -> dict[str, object] | None:
    """Return a JSON-ready distribution payload."""
    if distribution is None:
        return None
    return {
        "count": distribution.count,
        "min": distribution.minimum,
        "p5": distribution.p5,
        "median": distribution.median,
        "p95": distribution.p95,
        "max": distribution.maximum,
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    """Return an interpolated percentile from ascending scores."""
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean(vectors: list[list[float]]) -> list[float]:
    """Return the component-wise mean vector."""
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def _normalize(vector: list[float]) -> list[float]:
    """Return a unit vector, preserving zero vectors."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return list(vector)
    return [value / magnitude for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    return sum(a * b for a, b in zip(left, right))


__all__ = [
    "ScoreDistribution",
    "VoiceprintCalibrationReport",
    "calibrate_voiceprint_thresholds",
]
