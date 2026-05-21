"""Reusable Rich table for voiceprint match explanations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Iterable

from rich import box
from rich.table import Table

from app.postprocess import speaker_id_to_label
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_IGNORED,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    effective_match_status,
    match_threshold,
)


@dataclass(frozen=True, slots=True)
class SpeakerMatchRow:
    """Human-facing summary of one voiceprint match row."""

    label: str
    status: str
    candidate: str | None
    score: float | None
    threshold: float | None


def speaker_match_rows(
    matches: Iterable[object],
    *,
    default_threshold: float | None = None,
    ignored_speaker_ids: Collection[int] | None = None,
) -> tuple[SpeakerMatchRow, ...]:
    """
    Convert raw match objects into presentation rows.

    Args:
        matches: Dataclass or JSON-like match rows.
        default_threshold: Fallback threshold from the match payload.
        ignored_speaker_ids: Speaker ids the user has marked as ignored.

    Returns:
        Display rows in input order.
    """
    ignored_set = set(ignored_speaker_ids or ())
    return tuple(
        _speaker_match_row(
            match, default_threshold=default_threshold, ignored_speaker_ids=ignored_set
        )
        for match in matches
    )


def render_speaker_match_table(rows: tuple[SpeakerMatchRow, ...]) -> Table | None:
    """
    Build a Rich voiceprint explanation table.

    Args:
        rows: Prepared match rows.

    Returns:
        Rich table, or ``None`` when no match rows exist.
    """
    if not rows:
        return None
    table = Table(
        title=_table_title(rows), box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False
    )
    table.add_column("Speaker", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Threshold", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(
            row.label,
            _status_text(row.status),
            _candidate_text(row),
            _score_text(row.score),
            _score_text(row.threshold),
        )
    return table


def voiceprint_threshold_text(rows: tuple[SpeakerMatchRow, ...]) -> str:
    """
    Return a compact auto-accept threshold explanation.

    Args:
        rows: Match rows.

    Returns:
        Human-facing threshold text.
    """
    thresholds = {row.threshold for row in rows if row.threshold is not None}
    if not thresholds:
        return "-"
    if len(thresholds) == 1:
        return f"auto accept >= {thresholds.pop():.3f}"
    ordered = ", ".join(f"{threshold:.3f}" for threshold in sorted(thresholds))
    return f"per speaker: {ordered}"


def _speaker_match_row(
    match: object,
    *,
    default_threshold: float | None,
    ignored_speaker_ids: Collection[int] | None = None,
) -> SpeakerMatchRow:
    """Convert one raw match object into a display row."""
    status = effective_match_status(match, ignored_speaker_ids=ignored_speaker_ids)
    if status == MATCH_STATUS_MATCHED:
        candidate = accepted_match_name(match)
    elif status == MATCH_STATUS_IGNORED:
        candidate = None
    else:
        candidate = best_candidate_name(match)
    threshold = match_threshold(match, default_threshold)
    return SpeakerMatchRow(
        label=_match_label(match),
        status=status,
        candidate=candidate,
        score=best_candidate_score(match),
        threshold=threshold,
    )


def _match_label(match: object) -> str:
    """Return the best available speaker label."""
    label = _field(match, "label")
    if label:
        return str(label)
    speaker_id = _field(match, "speaker_id")
    try:
        return speaker_id_to_label(int(speaker_id))
    except TypeError, ValueError:
        return "Speaker"


def _field(match: object, name: str) -> object:
    """Read a field from either a mapping or an object."""
    if isinstance(match, dict):
        return match.get(name)
    return getattr(match, name, None)


def _table_title(rows: tuple[SpeakerMatchRow, ...]) -> str:
    """Return table title with threshold context."""
    threshold = voiceprint_threshold_text(rows)
    if threshold == "-":
        return "Voiceprint candidates"
    return f"Voiceprint candidates ({threshold})"


def _status_text(status: str) -> str:
    """Return styled match status."""
    styles = {
        MATCH_STATUS_MATCHED: "green",
        MATCH_STATUS_BELOW_THRESHOLD: "yellow",
        MATCH_STATUS_NO_CANDIDATE: "red",
        MATCH_STATUS_IGNORED: "cyan",
    }
    return f"[{styles.get(status, 'white')}]{status}[/]"


def _candidate_text(row: SpeakerMatchRow) -> str:
    """Return a candidate explanation for one row."""
    if row.status == MATCH_STATUS_MATCHED:
        return f"accepted: {row.candidate or 'unknown'}"
    if row.status == MATCH_STATUS_BELOW_THRESHOLD:
        return f"best: {row.candidate or 'unknown'}"
    if row.status == MATCH_STATUS_IGNORED:
        return "ignored"
    return "-"


def _score_text(value: float | None) -> str:
    """Format score-like values."""
    return "-" if value is None else f"{value:.3f}"
