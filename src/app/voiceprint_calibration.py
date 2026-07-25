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

from app.speaker_pipeline_params import resolve_match_threshold
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings

MIN_PERSON_SAMPLES = 2
SWEEP_START = 0.30
SWEEP_STOP = 0.95
SWEEP_STEP = 0.005
IMPOSTOR_RATE_TARGET = 0.01
# Raw score populations are shipped to the UI so a threshold slider can price
# each candidate value locally; cap the payload for large libraries.
MAX_EXPORTED_SCORES = 2000
# A suggested threshold must clear the worst impostor by this much, so a single
# unseen impostor slightly above today's maximum does not immediately match.
SUGGESTION_IMPOSTOR_MARGIN = 0.02
# Below this many genuine observations the sweep is anecdote, not statistics.
LOW_CONFIDENCE_SAMPLE_COUNT = 30


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
class ThresholdCost:
    """What one candidate threshold costs on the current library."""

    threshold: float
    false_reject_count: int
    false_reject_rate: float
    false_accept_count: int
    false_accept_rate: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready payload."""
        return {
            "threshold": self.threshold,
            "false_reject_count": self.false_reject_count,
            "false_reject_rate": self.false_reject_rate,
            "false_accept_count": self.false_accept_count,
            "false_accept_rate": self.false_accept_rate,
        }


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
    genuine_scores: tuple[float, ...] = ()
    impostor_scores: tuple[float, ...] = ()
    suggested_threshold: float | None = None
    suggested_reason: str = ""
    # Machine-readable counterpart of suggested_reason ("gap" | "single-person"
    # | "overlap" | "none") so a localized UI can restate it in its own words.
    suggested_kind: str = "none"
    low_confidence: bool = False

    @property
    def current_cost(self) -> ThresholdCost | None:
        """Return what the active threshold costs on this library."""
        return self.cost_at(self.current_threshold)

    @property
    def suggested_cost(self) -> ThresholdCost | None:
        """Return what the suggested threshold would cost on this library."""
        if self.suggested_threshold is None:
            return None
        return self.cost_at(self.suggested_threshold)

    def cost_at(self, threshold: float) -> ThresholdCost | None:
        """
        Price one candidate threshold against the stored score populations.

        Counts are scaled from the exported scores back to the true population.
        Above ``MAX_EXPORTED_SCORES`` the stored arrays are an evenly
        downsampled view, so counting them directly would report "2000 wrong
        matches" for any library large enough to be downsampled -- a number
        that is really the export cap wearing a cost's clothes. The rate is
        what the sample measures honestly; the count is that rate applied to
        ``ScoreDistribution.count``, which is the full population.

        The frontend prices the cursor locally from the same exported arrays
        and must scale identically, or dragging the slider would disagree with
        the value the backend put on the page.

        Args:
            threshold: Candidate acceptance threshold.

        Returns:
            Cost breakdown, or None when there is nothing to score against.
        """
        if not self.genuine_scores and not self.impostor_scores:
            return None
        reject_rate = _rate(self.genuine_scores, lambda score: score < threshold)
        accept_rate = _rate(self.impostor_scores, lambda score: score >= threshold)
        return ThresholdCost(
            threshold=threshold,
            false_reject_count=round(reject_rate * _population(self.genuine)),
            false_reject_rate=reject_rate,
            false_accept_count=round(accept_rate * _population(self.impostor)),
            false_accept_rate=accept_rate,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready payload."""
        current = self.current_cost
        suggested = self.suggested_cost
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
            "genuine_scores": list(self.genuine_scores),
            "impostor_scores": list(self.impostor_scores),
            "suggested_threshold": self.suggested_threshold,
            "suggested_reason": self.suggested_reason,
            "suggested_kind": self.suggested_kind,
            "low_confidence": self.low_confidence,
            "current_cost": current.to_dict() if current else None,
            "suggested_cost": suggested.to_dict() if suggested else None,
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
    suggested, reason, suggested_kind = _suggested_threshold(
        genuine_scores, impostor_scores, eer_threshold
    )
    low_confidence = len(genuine_scores) < LOW_CONFIDENCE_SAMPLE_COUNT
    if low_confidence and genuine_scores:
        warnings.append(
            f"only {len(genuine_scores)} genuine observations; treat the sweep as "
            "a direction, not a precise operating point"
        )
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
        current_threshold=resolve_match_threshold(),
        warnings=tuple(warnings),
        genuine_scores=_exported_scores(genuine_scores),
        impostor_scores=_exported_scores(impostor_scores),
        suggested_threshold=suggested,
        suggested_reason=reason,
        suggested_kind=suggested_kind,
        low_confidence=low_confidence,
    )


def _suggested_threshold(
    genuine: list[float], impostor: list[float], eer_threshold: float | None
) -> tuple[float | None, str, str]:
    """
    Suggest an operating threshold from the separation between populations.

    Prefers the midpoint of the gap between the worst impostor (plus a safety
    margin) and the 5th-percentile genuine score: that centers the threshold in
    the empty band between the two populations, so both an unusually strong
    impostor and an unusually weak genuine sample have room before they flip a
    decision. When the populations overlap there is no such band, and the
    equal-error point is the least-bad compromise.

    Args:
        genuine: Same-person leave-one-out scores.
        impostor: Best-other-person scores.
        eer_threshold: Equal-error point, when computable.

    Returns:
        Suggested threshold, a human-readable rationale, and a stable kind.
    """
    if not genuine:
        return (
            None,
            "no genuine observations; add more samples per person first",
            "none",
        )
    ceiling = _percentile(sorted(genuine), 0.05)
    if not impostor:
        return (
            round(max(ceiling, 0.0), 3),
            "only one person has embeddings, so there is no impostor evidence; "
            "this only protects against rejecting that person",
            "single-person",
        )
    floor = max(impostor) + SUGGESTION_IMPOSTOR_MARGIN
    if floor < ceiling:
        return (
            round((floor + ceiling) / 2, 3),
            f"centered in the gap between the worst impostor ({max(impostor):.3f}) "
            f"and the 5th-percentile genuine score ({ceiling:.3f})",
            "gap",
        )
    if eer_threshold is not None:
        return (
            eer_threshold,
            "genuine and impostor scores overlap, so no threshold separates them "
            "cleanly; this is the equal-error compromise. Fixing sample quality "
            "will help more than moving the threshold",
            "overlap",
        )
    return None, "not enough evidence to suggest a threshold", "none"


def _rate(scores: tuple[float, ...], predicate) -> float:
    """Return the share of scores satisfying predicate, 0.0 when empty."""
    if not scores:
        return 0.0
    return sum(1 for score in scores if predicate(score)) / len(scores)


def _population(distribution: ScoreDistribution | None) -> int:
    """Return a distribution's true observation count, 0 when absent."""
    return distribution.count if distribution else 0


def _exported_scores(scores: list[float]) -> tuple[float, ...]:
    """Return sorted scores, evenly downsampled when the population is large."""
    ordered = sorted(round(score, 4) for score in scores)
    if len(ordered) <= MAX_EXPORTED_SCORES:
        return tuple(ordered)
    step = len(ordered) / MAX_EXPORTED_SCORES
    return tuple(ordered[int(index * step)] for index in range(MAX_EXPORTED_SCORES))


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
    "ThresholdCost",
    "VoiceprintCalibrationReport",
    "calibrate_voiceprint_thresholds",
]
