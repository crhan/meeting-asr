"""Voiceprint match loading for the speaker review TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.speaker_match_status import (
    MATCH_STATUS_MATCHED,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    match_threshold,
    voiceprint_match_status,
)
from app.presentation.tui.speaker_people import optional_person_id


@dataclass(frozen=True, slots=True)
class SpeakerMatchPerson:
    """One scored voiceprint person candidate."""

    person_id: int | None
    name: str
    score: float | None
    person_public_id: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakerMatchCandidate:
    """One voiceprint match candidate for a project speaker."""

    name: str
    score: float | None
    accepted: bool
    person_id: int | None = None
    best_name: str | None = None
    best_score: float | None = None
    best_person_id: int | None = None
    best_person_public_id: str | None = None
    accepted_person_id: int | None = None
    accepted_person_public_id: str | None = None
    threshold: float | None = None
    status: str = ""
    candidates: tuple[SpeakerMatchPerson, ...] = ()


def load_match_candidates(path: Path) -> dict[int, SpeakerMatchCandidate]:
    """
    Load voiceprint match candidates if they exist.

    Args:
        path: speaker_matches.json path.

    Returns:
        Project speaker id to match candidate.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: dict[int, SpeakerMatchCandidate] = {}
    for item in payload.get("matches", []):
        if isinstance(item, dict) and "speaker_id" in item:
            candidates[int(item["speaker_id"])] = _match_candidate(item)
    return candidates


def accepted_review_name(match: SpeakerMatchCandidate | None) -> str | None:
    """
    Return a usable accepted match name.

    Args:
        match: Optional TUI match candidate.

    Returns:
        Accepted display name or ``None``.
    """
    if match is None or voiceprint_match_status(match) != MATCH_STATUS_MATCHED:
        return None
    return accepted_match_name(match)


def accepted_review_person_id(match: SpeakerMatchCandidate | None) -> int | None:
    """
    Return an accepted match person id.

    Args:
        match: Optional TUI match candidate.

    Returns:
        Accepted person id or ``None``.
    """
    if match is None or voiceprint_match_status(match) != MATCH_STATUS_MATCHED:
        return None
    return match.accepted_person_id or match.person_id


def accepted_review_person_public_id(match: SpeakerMatchCandidate | None) -> str | None:
    """
    Return an accepted match person public id.

    Args:
        match: Optional TUI match candidate.

    Returns:
        Accepted person public id or ``None``.
    """
    if match is None or voiceprint_match_status(match) != MATCH_STATUS_MATCHED:
        return None
    return match.accepted_person_public_id or match.best_person_public_id


def _match_candidate(item: dict[str, object]) -> SpeakerMatchCandidate:
    """Convert one raw match row into a TUI candidate."""
    status = voiceprint_match_status(item)
    name = accepted_match_name(item) if status == MATCH_STATUS_MATCHED else best_candidate_name(item)
    score = best_candidate_score(item)
    return SpeakerMatchCandidate(
        name=name or "unknown",
        score=score,
        accepted=bool(item.get("accepted")),
        person_id=optional_person_id(item.get("accepted_person_id") or item.get("person_id")),
        best_name=best_candidate_name(item),
        best_score=best_candidate_score(item),
        best_person_id=optional_person_id(item.get("best_person_id")),
        best_person_public_id=_optional_person_public_id(item.get("best_person_public_id")),
        accepted_person_id=optional_person_id(item.get("accepted_person_id")),
        accepted_person_public_id=_optional_person_public_id(item.get("accepted_person_public_id")),
        threshold=match_threshold(item),
        status=status,
        candidates=_person_candidates(item),
    )


def _person_candidates(item: dict[str, object]) -> tuple[SpeakerMatchPerson, ...]:
    """Parse top-k person candidates from one match row."""
    rows = []
    raw_candidates = item.get("candidates")
    if isinstance(raw_candidates, list):
        for raw in raw_candidates:
            if isinstance(raw, dict):
                rows.append(
                    SpeakerMatchPerson(
                        optional_person_id(raw.get("person_id")),
                        str(raw.get("name") or "unknown"),
                        _optional_score(raw.get("score")),
                        _optional_person_public_id(raw.get("person_public_id")),
                    )
                )
    best_name = best_candidate_name(item)
    best_person_id = optional_person_id(item.get("best_person_id"))
    best_person_public_id = _optional_person_public_id(item.get("best_person_public_id"))
    best_score = best_candidate_score(item)
    if best_name and not _has_candidate(rows, best_person_id, best_name):
        rows.append(SpeakerMatchPerson(best_person_id, best_name, best_score, best_person_public_id))
    return tuple(sorted(rows, key=_candidate_sort_key))


def _has_candidate(rows: list[SpeakerMatchPerson], person_id: int | None, name: str) -> bool:
    """Return whether rows already contain the best candidate."""
    if person_id is not None:
        return any(row.person_id == person_id for row in rows)
    normalized = " ".join(name.strip().split()).casefold()
    return any(" ".join(row.name.strip().split()).casefold() == normalized for row in rows)


def _candidate_sort_key(candidate: SpeakerMatchPerson) -> tuple[int, float]:
    """Sort scored candidates before unscored candidates."""
    if candidate.score is None:
        return (1, 0.0)
    return (0, -candidate.score)


def _optional_score(value: object) -> float | None:
    """Parse one optional score value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_person_public_id(value: object) -> str | None:
    """Parse an optional voiceprint person public id."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
