"""Shared voiceprint match status policy."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

MATCH_STATUS_MATCHED = "matched"
MATCH_STATUS_BELOW_THRESHOLD = "below-threshold"
MATCH_STATUS_NO_CANDIDATE = "no-candidate"
MATCH_STATUS_IGNORED = "ignored"
MATCH_STATUS_CROSSTALK = "crosstalk"


def voiceprint_match_status(match: object) -> str:
    """
    Classify a voiceprint match row.

    Args:
        match: Match dataclass or JSON-like mapping.

    Returns:
        One of matched, crosstalk, below-threshold, or no-candidate.
    """
    if bool(_field(match, "accepted")) and accepted_match_name(match):
        return MATCH_STATUS_MATCHED
    if bool(_field(match, "crosstalk")):
        # A persisted low-confidence crosstalk/noise flag (see speaker_crosstalk).
        # Advisory only: the speaker stays anonymous; it just must not block.
        return MATCH_STATUS_CROSSTALK
    if best_candidate_name(match):
        return MATCH_STATUS_BELOW_THRESHOLD
    score = (
        _float_field(match, "best_score")
        if _field(match, "best_score") is not None
        else _float_field(match, "score")
    )
    if score is not None and score > 0:
        return MATCH_STATUS_BELOW_THRESHOLD
    return MATCH_STATUS_NO_CANDIDATE


def project_has_unresolved_match(
    project_dir: Path,
    *,
    ignored_speaker_ids: Collection[int] | None = None,
) -> bool:
    """
    Return whether a project has any unresolved non-ignored speaker match.

    Shared by CLI next-step hints and the web project list ("needs review" badge).

    Args:
        project_dir: Project root.
        ignored_speaker_ids: Speaker ids the user marked as ignored; ``None`` loads
            ``speakers/speaker_ignore.json`` from the project.

    Returns:
        True when at least one non-ignored match row is neither matched nor
        crosstalk-flagged (crosstalk is advisory-only and must not block).
    """
    match_path = project_dir / "speakers" / "speaker_matches.json"
    if not match_path.exists():
        return False
    try:
        payload = json.loads(match_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # One truncated/corrupt match file (killed run, hand edit) must not take the
        # whole web project list down; treat it as "nothing unresolved to report".
        return False
    if not isinstance(payload, dict):
        return False
    if ignored_speaker_ids is None:
        # Local import: speaker_labeling imports app.models/postprocess; keeping this
        # module import-light avoids pulling that stack in for pure status checks.
        from app.speaker_labeling import load_project_ignored_speakers

        ignored_speaker_ids = load_project_ignored_speakers(project_dir)
    ignored = set(ignored_speaker_ids)
    for item in payload.get("matches", []):
        if not isinstance(item, dict):
            continue
        speaker_id = speaker_id_from_match(item)
        if speaker_id is not None and speaker_id in ignored:
            continue
        if voiceprint_match_status(item) not in (
            MATCH_STATUS_MATCHED,
            MATCH_STATUS_CROSSTALK,
        ):
            return True
    return False


def effective_match_status(
    match: object, *, ignored_speaker_ids: Collection[int] | None = None
) -> str:
    """
    Return the user-facing match status after merging explicit ignore state.

    Args:
        match: Match dataclass or JSON-like mapping.
        ignored_speaker_ids: Speaker ids the user has marked as ignored.

    Returns:
        ``"ignored"`` when the row belongs to an ignored speaker, otherwise the
        voiceprint match status.
    """
    if ignored_speaker_ids and speaker_id_from_match(match) in set(ignored_speaker_ids):
        return MATCH_STATUS_IGNORED
    return voiceprint_match_status(match)


def speaker_id_from_match(match: object) -> int | None:
    """
    Return the integer speaker id stored on a match row.

    Args:
        match: Match dataclass or JSON-like mapping.

    Returns:
        Integer speaker id, or ``None`` when the row lacks one.
    """
    value = _field(match, "speaker_id")
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def accepted_match_name(match: object) -> str | None:
    """
    Return the automatically accepted speaker name.

    Args:
        match: Match dataclass or JSON-like mapping.

    Returns:
        Accepted name, or None when the match was not accepted.
    """
    if not bool(_field(match, "accepted")):
        return None
    return _clean_name(_field(match, "accepted_name")) or _clean_name(
        _field(match, "name")
    )


def best_candidate_name(match: object) -> str | None:
    """
    Return the best candidate name without applying it automatically.

    Args:
        match: Match dataclass or JSON-like mapping.

    Returns:
        Best candidate name, or None when no candidate exists.
    """
    best_name = _clean_name(_field(match, "best_name"))
    if best_name:
        return best_name
    if bool(_field(match, "accepted")):
        return accepted_match_name(match)
    return _clean_name(_field(match, "name"))


def best_candidate_score(match: object) -> float | None:
    """
    Return the best candidate score.

    Args:
        match: Match dataclass or JSON-like mapping.

    Returns:
        Best score, or None when no candidate exists.
    """
    best_name = best_candidate_name(match)
    if best_name is None:
        score = (
            _float_field(match, "best_score")
            if _field(match, "best_score") is not None
            else _float_field(match, "score")
        )
        if score is not None and score > 0 and not bool(_field(match, "accepted")):
            return score
        return None
    return (
        _float_field(match, "best_score")
        if _field(match, "best_score") is not None
        else _float_field(match, "score")
    )


def match_threshold(match: object, default: float | None = None) -> float | None:
    """
    Return the threshold attached to a match row.

    Args:
        match: Match dataclass or JSON-like mapping.
        default: Fallback threshold when the row does not carry one.

    Returns:
        Threshold value, or the supplied default.
    """
    value = _float_field(match, "threshold")
    return default if value is None else value


def _field(match: object, key: str) -> Any:
    """Read a field from either a mapping or an object."""
    if isinstance(match, Mapping):
        return match.get(key)
    return getattr(match, key, None)


def _float_field(match: object, key: str) -> float | None:
    """Read a float field from either a mapping or an object."""
    value = _field(match, key)
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _clean_name(value: object) -> str | None:
    """Return a displayable non-placeholder speaker name."""
    if value is None:
        return None
    name = str(value).strip()
    if not name or name.lower() == "unknown":
        return None
    return name
