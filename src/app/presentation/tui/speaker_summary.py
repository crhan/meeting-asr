"""Plain summary rendering for speaker review sessions."""

from __future__ import annotations

import shlex
from typing import Any

from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    match_threshold,
    voiceprint_match_status,
)
from app.presentation.tui.speaker_status import speaker_status


def render_speaker_review_summary(session: Any, *, speaker_only: bool = False) -> str:
    """
    Render a non-interactive summary of the review queue.

    Args:
        session: Speaker review inputs.
        speaker_only: Prefer the speaker-only review entrypoint.

    Returns:
        Plain terminal text.
    """
    lines = [
        f"Speaker review queue: {session.project_dir}",
        f"Known people: {len(session.people_names)}",
    ]
    unresolved_speakers = [
        speaker
        for speaker in session.speakers
        if speaker.match is not None and voiceprint_match_status(speaker.match) != MATCH_STATUS_MATCHED
    ]
    if unresolved_speakers:
        project_ref = shlex.quote(session.overview.project_id)
        review_command = (
            f"meeting-asr project speakers review {project_ref}"
            if speaker_only
            else f"meeting-asr project review {project_ref}"
        )
        lines.extend(
            [
                "",
                f"Recommended next step: {review_command}",
                "This opens the human review workflow for unresolved speakers.",
                "",
                "Voiceprint status:",
            ]
        )
        lines.extend(_voiceprint_status_line(speaker) for speaker in unresolved_speakers)
        lines.extend(
            [
                "",
                "Advanced/scripted alternative (not the recommended human path):",
                f"  meeting-asr project speakers apply {project_ref} --map 0=Name",
                "",
                "After saving names:",
                f"  meeting-asr voiceprint capture {project_ref} --review",
                "  meeting-asr voiceprint embed",
                "",
                "Review queue:",
            ]
        )
    for speaker in session.speakers:
        lines.append(_summary_line(speaker))
    return "\n".join(lines)


def _voiceprint_status_line(speaker: Any) -> str:
    """
    Render a concise voiceprint status line for summary output.

    Args:
        speaker: Speaker review row.

    Returns:
        Human-readable voiceprint status.
    """
    if speaker.match is None:
        return f"{speaker.label}: no-match"
    status = voiceprint_match_status(speaker.match)
    if status == MATCH_STATUS_BELOW_THRESHOLD:
        name = best_candidate_name(speaker.match) or "unrecorded"
        score = best_candidate_score(speaker.match)
        score_text = "" if score is None else f" score={score:.3f}"
        threshold = match_threshold(speaker.match)
        threshold_text = "" if threshold is None else f" threshold={threshold:.3f}"
        return f"{speaker.label}: below-threshold best={name}{score_text}{threshold_text}"
    if status == MATCH_STATUS_MATCHED:
        name = accepted_match_name(speaker.match) or speaker.match.name
        return f"{speaker.label}: matched name={name}"
    return f"{speaker.label}: no-candidate"


def _summary_line(speaker: Any) -> str:
    """Render one plain summary row."""
    status = speaker_status(speaker)
    if speaker.match is None:
        match = "-"
    elif voiceprint_match_status(speaker.match) == MATCH_STATUS_NO_CANDIDATE:
        match = "no-candidate"
    elif speaker.match.accepted:
        match = f"matched:{accepted_match_name(speaker.match) or speaker.match.name}"
    else:
        match = _below_threshold_text(speaker.match)
    return (
        f"{speaker.label} speaker_id={speaker.speaker_id} "
        f"status={status} name={speaker.current_name} match={match}"
    )


def _below_threshold_text(match: Any) -> str:
    """Render below-threshold match text."""
    score = best_candidate_score(match)
    score_text = "" if score is None else f" score={score:.3f}"
    threshold = match_threshold(match)
    threshold_text = "" if threshold is None else f" threshold={threshold:.3f}"
    candidate = best_candidate_name(match)
    if candidate is None and match.name != "unknown":
        candidate = match.name
    return f"below-threshold:{candidate or 'unrecorded'}{score_text}{threshold_text}"
