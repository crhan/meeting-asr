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
from collections import defaultdict
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
    """What one candidate threshold costs on the current library.

    Strictly a *threshold-only* model: acceptance is ``score >= threshold`` and
    nothing else. Production matching also rescues a score down to
    ``STRONG_MARGIN_ACCEPT_SCORE`` when it leads the runner-up by
    ``STRONG_MARGIN_ACCEPT_MARGIN``, and pricing that would need each probe's
    best/runner-up pair, which this all-pairs sweep does not have. So these
    counts are an upper bound on false rejects and a lower bound on false
    accepts, and callers must present them as the cost of the threshold rule
    rather than as the decisions the pipeline will actually make.
    """

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
class ConfusablePair:
    """How close one person sits to the single other person they risk.

    The impostor sweep already scores every sample against every other
    person's centroid, then keeps only the maximum -- throwing away *whose*
    centroid it was. That discarded identity is the actionable half: a library
    told "3 wrong-person scores clear the threshold" can only answer by moving
    the threshold, whereas "3 of A's samples are accepted as B" points at two
    specific people whose audio can be fixed instead.

    Direction matters, so the pair is deliberately not symmetric: a stray
    sample of A may sit on B's centroid while every sample of B stays far from
    A's. The remedy then belongs to A alone.

    A high score against another centroid is *not* by itself a wrong match.
    Matching ranks every candidate and applies the threshold to the winner
    (``_acceptance_decision`` on ``candidates[0]``), so a sample scoring 0.82
    as B while scoring 0.95 as itself is still named correctly. The pair
    therefore carries the **competition**, not just the impostor score: each
    sample's score against the other centroid is compared with its own
    leave-one-out centroid score -- the same standing in for an unseen probe
    that the genuine distribution uses -- and only a sample the other person
    actually wins counts as a crossing.
    """

    person_public_id: str
    person_name: str
    other_public_id: str
    other_name: str
    # Best score any of this person's samples reaches against the other's
    # centroid -- the same quantity the impostor distribution is built from.
    best_score: float
    # Thinnest win over this other person across the samples: own
    # leave-one-out score minus the other-centroid score. Negative means the
    # other person already ranks first for that sample. None when the person
    # has too few samples for a leave-one-out centroid, in which case the
    # competition cannot be judged and no crossing is claimed.
    min_lead: float | None
    # Of this person's samples, those the other person both wins and clears
    # the active threshold with -- an automatic wrong name today.
    crossing_sample_public_ids: tuple[str, ...]
    sample_count: int

    @property
    def crossing_count(self) -> int:
        """Return how many samples already clear the threshold as the other."""
        return len(self.crossing_sample_public_ids)


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
    # Per person, the one other person their samples come closest to, most
    # dangerous first. Empty when fewer than two people have embeddings.
    neighbors: tuple[ConfusablePair, ...] = ()

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
            "neighbors": [
                {
                    "person_public_id": pair.person_public_id,
                    "person_name": pair.person_name,
                    "other_public_id": pair.other_public_id,
                    "other_name": pair.other_name,
                    "best_score": pair.best_score,
                    "min_lead": pair.min_lead,
                    "crossing_count": pair.crossing_count,
                    "crossing_sample_public_ids": list(pair.crossing_sample_public_ids),
                    "sample_count": pair.sample_count,
                }
                for pair in self.neighbors
            ],
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
    public_ids_by_person: dict[int, str] = {}
    # Sample public ids parallel to vectors_by_person, so an impostor score can
    # name the sample that produced it rather than just counting it.
    sample_ids_by_person: dict[int, list[str]] = {}
    for row in rows:
        vectors_by_person.setdefault(row.speaker_id, []).append(_normalize(row.vector))
        sample_ids_by_person.setdefault(row.speaker_id, []).append(row.sample_public_id)
        names_by_person[row.speaker_id] = row.speaker_name
        public_ids_by_person[row.speaker_id] = row.speaker_public_id
    warnings: list[str] = []
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    centroids = {
        person_id: _normalize(_mean(vectors))
        for person_id, vectors in vectors_by_person.items()
    }
    # Resolved once and reused for the report: two reads of the configured
    # threshold could disagree, which would price the sweep against one value
    # while reporting another.
    threshold = resolve_match_threshold()
    pair_best: dict[tuple[int, int], float] = {}
    pair_leads: dict[tuple[int, int], float] = {}
    pair_crossings: dict[tuple[int, int], list[str]] = defaultdict(list)
    scored_people = 0
    for person_id, vectors in vectors_by_person.items():
        others = [
            (key, centroid) for key, centroid in centroids.items() if key != person_id
        ]
        has_leave_one_out = len(vectors) >= MIN_PERSON_SAMPLES
        for index, vector in enumerate(vectors):
            own: float | None = None
            if has_leave_one_out:
                rest = vectors[:index] + vectors[index + 1 :]
                own = _cosine(vector, _normalize(_mean(rest)))
                genuine_scores.append(own)
            if others:
                scored = [
                    (other_id, _cosine(vector, centroid))
                    for other_id, centroid in others
                ]
                impostor_scores.append(max(score for _, score in scored))
                sample_public_id = sample_ids_by_person[person_id][index]
                for other_id, score in scored:
                    key = (person_id, other_id)
                    if score > pair_best.get(key, -1.0):
                        pair_best[key] = score
                    if own is None:
                        # Without a leave-one-out centroid there is nothing to
                        # run the other person against, so the competition is
                        # simply unknown -- claiming a swap here would be the
                        # very over-reach this comparison exists to avoid.
                        continue
                    lead = own - score
                    if lead < pair_leads.get(key, math.inf):
                        pair_leads[key] = lead
                    # Strictly greater: matching ranks candidates and applies
                    # the threshold to the winner, so the other person has to
                    # actually take first place before this is a wrong name.
                    if score > own and score >= threshold:
                        pair_crossings[key].append(sample_public_id)
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
        current_threshold=threshold,
        warnings=tuple(warnings),
        genuine_scores=_exported_scores(genuine_scores),
        impostor_scores=_exported_scores(impostor_scores),
        suggested_threshold=suggested,
        suggested_reason=reason,
        suggested_kind=suggested_kind,
        low_confidence=low_confidence,
        neighbors=_confusable_pairs(
            pair_best,
            pair_leads,
            pair_crossings,
            names_by_person,
            public_ids_by_person,
            vectors_by_person,
        ),
    )


def _confusable_pairs(
    pair_best: dict[tuple[int, int], float],
    pair_leads: dict[tuple[int, int], float],
    pair_crossings: dict[tuple[int, int], list[str]],
    names: dict[int, str],
    public_ids: dict[int, str],
    vectors_by_person: dict[int, list[list[float]]],
) -> tuple[ConfusablePair, ...]:
    """
    Reduce the all-pairs evidence to the one other person each one risks.

    Ranking is by danger, not by similarity: a pair the other person already
    wins and gets accepted on beats one they merely score well against, and a
    thin winning margin beats a comfortable one. Sorting by raw score instead
    would promote people who are simply similar over people who are actually
    swappable. Only the top pair is kept, because the remedy -- fix this
    person's audio -- is the same whoever else is nearby.

    Args:
        pair_best: Best other-centroid score per ordered (person, other) pair.
        pair_leads: Thinnest own-minus-other margin per pair, where known.
        pair_crossings: Sample public ids the other person wins and clears the
            threshold with, per pair.
        names: Person display name by speaker id.
        public_ids: Person public id by speaker id.
        vectors_by_person: Embedded vectors by speaker id.

    Returns:
        One pair per person that has any other person to be confused with,
        most dangerous first.
    """
    riskiest: dict[int, tuple[tuple[int, float, float], int]] = {}
    # Walk the pairs in a fixed order and only displace on a strictly better
    # rank, so an exact tie between two others resolves to the same one on
    # every run rather than to whichever row the store happened to return.
    for person_id, other_id in sorted(
        pair_best, key=lambda key: (key[0], names[key[1]].casefold(), key[1])
    ):
        rank = (
            len(pair_crossings.get((person_id, other_id), ())),
            # Negated so a thinner margin ranks higher. An unknown margin
            # sorts last rather than first: not having been able to judge the
            # competition is not evidence of danger.
            -pair_leads.get((person_id, other_id), math.inf),
            pair_best[(person_id, other_id)],
        )
        current = riskiest.get(person_id)
        if current is None or rank > current[0]:
            riskiest[person_id] = (rank, other_id)
    pairs = [
        ConfusablePair(
            person_public_id=public_ids[person_id],
            person_name=names[person_id],
            other_public_id=public_ids[other_id],
            other_name=names[other_id],
            best_score=pair_best[(person_id, other_id)],
            min_lead=pair_leads.get((person_id, other_id)),
            crossing_sample_public_ids=tuple(
                pair_crossings.get((person_id, other_id), ())
            ),
            sample_count=len(vectors_by_person[person_id]),
        )
        for person_id, (_rank, other_id) in riskiest.items()
    ]
    return tuple(
        sorted(
            pairs,
            key=lambda item: (
                -item.crossing_count,
                item.min_lead if item.min_lead is not None else math.inf,
                -item.best_score,
                item.person_name.casefold(),
            ),
        )
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
    """
    Return sorted scores, evenly downsampled when the population is large.

    Both endpoints are kept. Striding by ``int(index * step)`` never lands on
    the last element, which drops the maximum every time -- and the maximum is
    the one score that matters most here: a handful of impostors above the
    threshold is exactly the tail that makes a threshold unsafe, and losing it
    would let ``cost_at`` price that threshold at zero false accepts.
    """
    ordered = sorted(round(score, 4) for score in scores)
    if len(ordered) <= MAX_EXPORTED_SCORES:
        return tuple(ordered)
    last = len(ordered) - 1
    return tuple(
        ordered[round(index * last / (MAX_EXPORTED_SCORES - 1))]
        for index in range(MAX_EXPORTED_SCORES)
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
    "ConfusablePair",
    "ScoreDistribution",
    "ThresholdCost",
    "VoiceprintCalibrationReport",
    "calibrate_voiceprint_thresholds",
]
