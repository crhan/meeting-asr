"""Low-confidence crosstalk/noise classification for voiceprint matches.

Meeting tails often pick up a different group's fragmentary crosstalk: a handful
of short utterances, a very low best voiceprint score, and candidates that do not
converge on any one library person. Such clusters used to stall at
``below-threshold`` and force downstream tools to either guess a name or skip the
whole meeting.

This module classifies those clusters as *suspected crosstalk/noise*. It is a
non-destructive advisory label only: the speaker stays an anonymous ``Speaker N``
with every sentence intact, but the cluster is flagged so the main flow does not
block on it and downstream can choose to let it through. The thresholds are
configurable; the defaults are deliberately conservative because the worst case
of a false positive is a harmless "suspected crosstalk" badge on a real but very
quiet attendee, never dropped audio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_CROSSTALK_MAX_SAMPLES = 3
DEFAULT_CROSSTALK_SCORE_FLOOR = 0.5
DEFAULT_CROSSTALK_CONCENTRATION_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class CrosstalkParams:
    """Tunable thresholds for the crosstalk/noise tier.

    Attributes:
        enabled: Whether crosstalk classification runs at all.
        max_samples: A cluster with more sentences than this is never crosstalk.
        score_floor: The best candidate score must be below this (and above 0) to
            qualify; a clearly-matched person scores at or above it.
        concentration_margin: When the top two candidates lead by at least this
            margin the cluster has a clear (if weak) identity and is left as a
            real below-threshold person rather than crosstalk.
    """

    enabled: bool = True
    max_samples: int = DEFAULT_CROSSTALK_MAX_SAMPLES
    score_floor: float = DEFAULT_CROSSTALK_SCORE_FLOOR
    concentration_margin: float = DEFAULT_CROSSTALK_CONCENTRATION_MARGIN


def is_crosstalk(match: object, params: CrosstalkParams) -> bool:
    """Return whether a match row looks like low-confidence crosstalk/noise.

    A cluster is crosstalk when it is unmatched, has few samples, its best
    candidate is weak (present but below ``score_floor``), and the candidates do
    not concentrate on a single identity. An empty-library row (no candidate at
    all) stays ``no-candidate`` rather than crosstalk, so a missing voiceprint
    library never mass-flags real speakers.

    Args:
        match: A ``SpeakerMatch`` dataclass or JSON-like mapping.
        params: Crosstalk thresholds.

    Returns:
        ``True`` when the row should be flagged as suspected crosstalk.
    """
    if not params.enabled:
        return False
    if bool(_field(match, "accepted")):
        return False
    sample_count = _int_field(match, "sample_count")
    if sample_count is None or sample_count > params.max_samples:
        return False
    best_score = _best_score(match)
    if best_score is None or best_score <= 0.0 or best_score >= params.score_floor:
        return False
    candidate_scores = _candidate_scores(match)
    if len(candidate_scores) >= 2:
        lead = candidate_scores[0] - candidate_scores[1]
        if lead >= params.concentration_margin:
            # A weak but clear front-runner: treat as a real below-threshold
            # person, not ambiguous noise.
            return False
    return True


def _best_score(match: object) -> float | None:
    """Return the best candidate score, falling back to the raw score."""
    best = _float_field(match, "best_score")
    return best if best is not None else _float_field(match, "score")


def _candidate_scores(match: object) -> list[float]:
    """Return candidate scores in ranked order."""
    candidates = _field(match, "candidates") or ()
    scores: list[float] = []
    for candidate in candidates:
        score = _float_field(candidate, "score")
        if score is not None:
            scores.append(score)
    return scores


def _field(item: object, key: str) -> Any:
    """Read a field from either a mapping or an object."""
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _float_field(item: object, key: str) -> float | None:
    """Read a float field from either a mapping or an object."""
    value = _field(item, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_field(item: object, key: str) -> int | None:
    """Read an int field from either a mapping or an object."""
    value = _field(item, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CrosstalkParams",
    "DEFAULT_CROSSTALK_MAX_SAMPLES",
    "DEFAULT_CROSSTALK_SCORE_FLOOR",
    "DEFAULT_CROSSTALK_CONCENTRATION_MARGIN",
    "is_crosstalk",
]
