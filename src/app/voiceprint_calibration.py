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

from app.speaker_matching import (
    _KnownProjectVector,
    _KnownSpeakerVector,
    _acceptance_decision,
    _ranked_matches,
    _score_known_vector,
)
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
    Matching ranks every candidate and applies the acceptance rule to the
    winner, so a sample scoring 0.82 as B while scoring 0.95 as itself is
    still named correctly. The pair therefore carries the **competition**, and
    that competition is replayed through the production decision path rather
    than re-derived: the same project-aware candidate scoring, the same
    ranking, and the same acceptance rule -- which also attaches a
    sub-threshold winner that runs away from the runner-up. See
    :func:`_confusable_pairs`.

    Unlike the genuine/impostor distributions, which stay a deliberately
    simple threshold-rule model over whole-person centroids, this is meant to
    predict the decision itself. The two therefore need not agree, and
    ``threshold-too-low`` counting more wrong-person scores than there are
    crossings here is expected.
    """

    person_public_id: str
    person_name: str
    other_public_id: str
    other_name: str
    # Best score any of this person's samples reaches against the other, as
    # matching would score them -- so possibly from one of the other's
    # per-project centroids rather than their whole-person centroid.
    best_score: float
    # Thinnest win over this other person across the samples: own score minus
    # theirs, both from the production scorer. Negative means the other person
    # already ranks first for that sample. None when the person has too few
    # samples to stand in for an unseen probe, in which case the competition
    # cannot be judged and no crossing is claimed.
    min_lead: float | None
    # Of this person's samples, those this other person actually wins and is
    # accepted on -- an automatic wrong name today.
    crossing_sample_public_ids: tuple[str, ...]
    sample_count: int
    # Why matching accepted those winners, straight from
    # ``_acceptance_decision``: "threshold", "strong-margin", or both. The
    # report has to say which, because a strong-margin acceptance happens
    # *below* the bar and describing it as clearing the bar is simply untrue.
    accept_reasons: tuple[str, ...] = ()

    @property
    def crossing_count(self) -> int:
        """Return how many of this person's samples the other person takes."""
        return len(self.crossing_sample_public_ids)

    @property
    def accept_reason(self) -> str | None:
        """Return "threshold", "strong-margin", "mixed", or None."""
        if not self.accept_reasons:
            return None
        if len(self.accept_reasons) > 1:
            return "mixed"
        return self.accept_reasons[0]


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
                    "accept_reason": pair.accept_reason,
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
    people = _library_people(rows)
    vectors_by_person = {person.person_id: person.vectors for person in people.values()}
    names_by_person = {person.person_id: person.name for person in people.values()}
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
        current_threshold=threshold,
        warnings=tuple(warnings),
        genuine_scores=_exported_scores(genuine_scores),
        impostor_scores=_exported_scores(impostor_scores),
        suggested_threshold=suggested,
        suggested_reason=reason,
        suggested_kind=suggested_kind,
        low_confidence=low_confidence,
        neighbors=_confusable_pairs(people, threshold),
    )


@dataclass(frozen=True, slots=True)
class _LibraryPerson:
    """One person's embedded samples, kept with the facts matching needs."""

    person_id: int
    name: str
    public_id: str
    vectors: list[list[float]]
    sample_public_ids: list[str]
    project_ids: list[str]


def _library_people(rows: list) -> dict[int, _LibraryPerson]:
    """Group embedding rows into per-person sample sets."""
    people: dict[int, _LibraryPerson] = {}
    for row in rows:
        person = people.get(row.speaker_id)
        if person is None:
            person = _LibraryPerson(
                person_id=row.speaker_id,
                name=row.speaker_name,
                public_id=row.speaker_public_id,
                vectors=[],
                sample_public_ids=[],
                project_ids=[],
            )
            people[row.speaker_id] = person
        person.vectors.append(_normalize(row.vector))
        person.sample_public_ids.append(row.sample_public_id)
        person.project_ids.append(row.project_id)
    return people


def _known_speaker_vector(
    person: _LibraryPerson, *, exclude: int | None = None
) -> _KnownSpeakerVector | None:
    """
    Build the candidate data matching would hold for this person.

    Mirrors ``_known_speaker_vectors``: a whole-person centroid plus one
    centroid per source project, because ``_score_known_vector`` scores a
    probe against the best of them. Rebuilding only the whole-person centroid
    would score a probe against something production never uses on its own.

    Args:
        person: The person's embedded samples.
        exclude: Index of a sample to leave out, so it can play an unseen
            probe against the library the rest of it forms.

    Returns:
        Candidate data, or None when nothing is left to build it from.
    """
    kept = [index for index in range(len(person.vectors)) if index != exclude]
    if not kept:
        return None
    by_project: dict[str, list[list[float]]] = defaultdict(list)
    for index in kept:
        by_project[person.project_ids[index]].append(person.vectors[index])
    return _KnownSpeakerVector(
        person.person_id,
        person.name,
        _normalize(_mean([person.vectors[index] for index in kept])),
        person.public_id,
        tuple(
            _KnownProjectVector(project_id, _normalize(_mean(vectors)), len(vectors))
            for project_id, vectors in sorted(by_project.items())
        ),
        len(kept),
        len(by_project),
    )


def _confusable_pairs(
    people: dict[int, _LibraryPerson], threshold: float
) -> tuple[ConfusablePair, ...]:
    """
    Replay matching on each sample and report who it would be named.

    Every sample is run through the *production* decision path rather than a
    re-derived approximation of it: ``_ranked_matches`` over the same
    project-aware candidate data, then ``_acceptance_decision`` on the winner
    -- which also accepts a sub-threshold winner that leads the runner-up by
    ``STRONG_MARGIN_ACCEPT_MARGIN``. Re-deriving any part of that invites the
    report to claim wrong names matching would not make, and to miss ones it
    would.

    The sample under test is excluded from its own person's candidate data,
    which is the same leave-one-out standing-in-for-an-unseen-probe the
    genuine distribution uses. A person with a single sample has nothing left
    to stand for them, so their competition is simply not judged.

    Args:
        people: Embedded samples grouped by person.
        threshold: Acceptance threshold in force.

    Returns:
        One pair per person whose samples reach another person at all, most
        dangerous first.
    """
    full = {
        person_id: _known_speaker_vector(person) for person_id, person in people.items()
    }
    # owner -> other -> evidence
    crossings: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    reasons: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    leads: dict[int, dict[int, float]] = defaultdict(dict)
    bests: dict[int, dict[int, float]] = defaultdict(dict)
    for person_id, person in people.items():
        for index in range(len(person.vectors)):
            probe = person.vectors[index]
            own = _known_speaker_vector(person, exclude=index)
            if own is None:
                continue
            # Candidate order is part of the decision, not a detail: `sorted`
            # is stable, so an exact tie is settled by the order
            # `_known_speaker_vectors` hands matching, which is each person's
            # first appearance in the embedding rows. `people` is built from
            # the same rows in the same order, so substituting the owner in
            # place -- rather than appending them -- reproduces which name a
            # tie actually attaches instead of inventing a different one.
            known = {
                other_id: (own if other_id == person_id else candidate)
                for other_id, candidate in full.items()
                if candidate is not None
            }
            if len(known) < 2:
                continue
            candidates = _ranked_matches(probe, known, limit=3)
            accepted, reason = _acceptance_decision(
                candidates[0] if candidates else None, tuple(candidates), threshold
            )
            own_score, _source = _score_known_vector(probe, own)
            top_other = next(
                (item for item in candidates if item.person_id != person_id), None
            )
            if top_other is None:
                continue
            lead = own_score - top_other.score
            if lead < leads[person_id].get(top_other.person_id, math.inf):
                leads[person_id][top_other.person_id] = lead
            if top_other.score > bests[person_id].get(top_other.person_id, -1.0):
                bests[person_id][top_other.person_id] = top_other.score
            winner = candidates[0]
            # Only the candidate that actually wins is blamed. A third person
            # outscoring the owner does not make *this* other the name that
            # would be attached.
            if accepted and winner.person_id != person_id:
                crossings[person_id][winner.person_id].append(
                    person.sample_public_ids[index]
                )
                reasons[person_id][winner.person_id].add(reason or "threshold")
                if winner.score > bests[person_id].get(winner.person_id, -1.0):
                    bests[person_id][winner.person_id] = winner.score
                leads[person_id].setdefault(winner.person_id, own_score - winner.score)
    pairs = [
        pair
        for person_id, person in people.items()
        if (
            pair := _riskiest_pair(
                person,
                people,
                crossings.get(person_id, {}),
                reasons.get(person_id, {}),
                leads.get(person_id, {}),
                bests.get(person_id, {}),
            )
        )
        is not None
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


def _riskiest_pair(
    person: _LibraryPerson,
    people: dict[int, _LibraryPerson],
    crossings: dict[int, list[str]],
    reasons: dict[int, set[str]],
    leads: dict[int, float],
    bests: dict[int, float],
) -> ConfusablePair | None:
    """
    Pick the one other person this person is most at risk from.

    Ranking is by danger, not by similarity: someone who already takes the
    name beats someone merely scored well against, and a thin win beats a
    comfortable one. Only the top pair is kept, because the remedy -- fix this
    person's audio -- is the same whoever else is nearby.
    """
    if not bests:
        return None
    # A fixed walk order with strict displacement keeps an exact tie resolving
    # to the same person on every run, rather than to whichever row the store
    # happened to return first.
    ordered = sorted(bests, key=lambda other: (people[other].name.casefold(), other))
    best_other = max(
        ordered,
        key=lambda other: (
            len(crossings.get(other, ())),
            -leads.get(other, math.inf),
            bests[other],
        ),
    )
    other = people[best_other]
    return ConfusablePair(
        person_public_id=person.public_id,
        person_name=person.name,
        other_public_id=other.public_id,
        other_name=other.name,
        best_score=bests[best_other],
        min_lead=leads.get(best_other),
        crossing_sample_public_ids=tuple(crossings.get(best_other, ())),
        sample_count=len(person.vectors),
        accept_reasons=tuple(sorted(reasons.get(best_other, set()))),
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
