"""Status rendering helpers for the speaker review TUI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rich.markup import escape

from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    voiceprint_match_status,
)
from app.utils import format_ms_timestamp


class SpeakerMatchLike(Protocol):
    """Match fields needed by status rendering."""

    name: str
    score: float | None
    accepted: bool
    best_name: str | None
    best_score: float | None
    threshold: float | None
    status: str


class ReviewSpeakerLike(Protocol):
    """Speaker fields needed by status rendering."""

    speaker_id: int
    label: str
    current_name: str
    match: SpeakerMatchLike | None
    ignored: bool


@dataclass(frozen=True, slots=True)
class VoiceprintReviewProgress:
    """Project-scoped voiceprint progress for the review TUI."""

    captured_names_by_speaker: dict[int, frozenset[str]]
    captured_sample_ids: frozenset[int]
    embed_model: str | None
    embedded_sample_ids: frozenset[int] | None
    embed_error: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakerReviewOverview:
    """Project and workflow state shown above the review panes."""

    project_id: str
    title: str
    project_status: str
    source_name: str
    duration_ms: int
    match_file_exists: bool
    saved_names_by_speaker: dict[int, str]
    voiceprint: VoiceprintReviewProgress


def render_overview_pane(
    speakers: Sequence[ReviewSpeakerLike],
    overview: SpeakerReviewOverview,
    selected: ReviewSpeakerLike,
) -> str:
    """
    Render stable project and workflow state.

    Args:
        speakers: Current review speakers.
        overview: Immutable project overview.
        selected: Currently selected speaker.

    Returns:
        Rich markup for the overview pane.
    """
    lines = [
        _page_overview_line(),
        _project_overview_line(overview, len(speakers)),
        _workflow_overview_line(speakers, overview),
        _match_overview_line(speakers),
        _risk_overview_line(speakers, selected),
        _output_overview_line(),
        _next_action_line(speakers, overview),
    ]
    return "\n".join(lines)


def speaker_status(speaker: ReviewSpeakerLike) -> str:
    """
    Return the review status for one speaker.

    Args:
        speaker: Speaker row.

    Returns:
        Status key used by color and icon helpers.
    """
    if has_conflict(speaker):
        return "conflict"
    if has_mismatch(speaker):
        return "mismatch"
    if is_ignored(speaker):
        return "ignored"
    if speaker.current_name == speaker.label:
        return "review"
    if speaker.match and speaker.current_name == accepted_match_name(speaker.match):
        return "matched"
    return "confirmed"


def is_ignored(speaker: ReviewSpeakerLike) -> bool:
    """
    Return whether a speaker has been deliberately kept anonymous.

    Args:
        speaker: Speaker row.

    Returns:
        True when the anonymous label was explicitly confirmed.
    """
    return speaker.ignored and speaker.current_name == speaker.label


def has_conflict(speaker: ReviewSpeakerLike) -> bool:
    """
    Return whether the current name conflicts with an accepted match.

    Args:
        speaker: Speaker row.

    Returns:
        True when an accepted match disagrees with the current name.
    """
    if speaker.match is None or not speaker.match.accepted:
        return False
    match_name = accepted_match_name(speaker.match)
    if not match_name or speaker.current_name == speaker.label:
        return False
    return speaker.current_name != match_name


def has_mismatch(speaker: ReviewSpeakerLike) -> bool:
    """
    Return whether a review-only match disagrees with the current name.

    Args:
        speaker: Speaker row.

    Returns:
        True when a non-accepted match points to another name.
    """
    if speaker.match is None or speaker.match.accepted:
        return False
    match_name = best_candidate_name(speaker.match)
    if not match_name or speaker.current_name == speaker.label:
        return False
    return speaker.current_name != match_name


def status_style(status: str) -> str:
    """
    Map a status to a Rich style.

    Args:
        status: Speaker status key.

    Returns:
        Rich style string.
    """
    styles = {
        "conflict": "bold red",
        "mismatch": "orange1",
        "ignored": "cyan",
        "review": "yellow",
        "matched": "green",
    }
    return styles.get(status, "bold green")


def status_icon(speaker: ReviewSpeakerLike) -> str:
    """
    Return a compact status marker.

    Args:
        speaker: Speaker row.

    Returns:
        Single-character marker.
    """
    status = speaker_status(speaker)
    return {"conflict": "!", "mismatch": "*", "ignored": "-", "review": "?", "matched": "~"}.get(status, "+")


def match_badge(speaker: ReviewSpeakerLike) -> str:
    """
    Render one compact match badge for a speaker.

    Args:
        speaker: Speaker row.

    Returns:
        Human-readable match summary.
    """
    if speaker.match is None:
        return "match=- ignored" if is_ignored(speaker) else "match=-"
    state = voiceprint_match_status(speaker.match)
    display_name = accepted_match_name(speaker.match) if state == MATCH_STATUS_MATCHED else best_candidate_name(speaker.match)
    display = "match=-" if state == MATCH_STATUS_NO_CANDIDATE else f"match={escape(display_name or 'unrecorded')}"
    score = best_candidate_score(speaker.match)
    score_text = "-" if score is None else f"{score:.3f}"
    if is_ignored(speaker):
        state = "ignored"
    elif has_conflict(speaker):
        state = "CONFLICT"
    elif has_mismatch(speaker):
        state = "mismatch"
    return f"{display} score={score_text} {state}"


def render_match_lines(match: SpeakerMatchLike | None) -> list[str]:
    """
    Render the selected speaker's voiceprint match.

    Args:
        match: Optional voiceprint match.

    Returns:
        Lines for the identity pane.
    """
    if match is None:
        return ["Match: -"]
    state = voiceprint_match_status(match)
    if state == MATCH_STATUS_NO_CANDIDATE:
        return ["Match: -", "Status: no-candidate"]
    name = accepted_match_name(match) if state == MATCH_STATUS_MATCHED else best_candidate_name(match)
    score = best_candidate_score(match)
    score_text = "-" if score is None else f"{score:.3f}"
    prefix = "Match" if state == MATCH_STATUS_MATCHED else "Best candidate"
    return [f"{prefix}: {escape(name or 'unrecorded')}", f"Score: {score_text} status={state}"]


def render_selected_speaker_line(speaker: ReviewSpeakerLike) -> str:
    """
    Render selected speaker identity state inside the sample pane.

    Args:
        speaker: Selected speaker.

    Returns:
        Rich markup line.
    """
    status = speaker_status(speaker)
    return f"Name: [b]{escape(speaker.current_name)}[/] | status={status} | {match_badge(speaker)}"


def _page_overview_line() -> str:
    """Render the active top-level TUI page."""
    return (
        "[reverse][b] PROJECT REVIEW [/b][/]  "
        "p: switch project | v: Voiceprint Review | /: identity | e: edit text | s: save | q: quit"
    )


def _project_overview_line(overview: SpeakerReviewOverview, speaker_count: int) -> str:
    """Render the project identity line."""
    duration = format_ms_timestamp(overview.duration_ms)
    return (
        f"[b]Project[/b]  {escape(overview.title)} [dim]({escape(overview.project_id)})[/] | "
        f"{duration} | {speaker_count} speakers | project={escape(overview.project_status)}"
    )


def _workflow_overview_line(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render project workflow progress."""
    voiceprint = overview.voiceprint
    match_state = _badge("Match", "done" if overview.match_file_exists else "pending", overview.match_file_exists)
    manual_state = _manual_state(speakers, overview)
    capture_todo = _capture_todo_count(speakers, voiceprint)
    capture_state = (
        f"Capture=[{'green' if capture_todo == 0 else 'yellow'}]todo {capture_todo}[/], "
        f"{len(voiceprint.captured_names_by_speaker)} speakers/{len(voiceprint.captured_sample_ids)} clips"
    )
    return (
        f"[b]Steps[/b]    1 {match_state} | 2 Names={manual_state} | "
        f"3 {capture_state} | 4 {_embed_state(voiceprint)}"
    )


def _match_overview_line(speakers: Sequence[ReviewSpeakerLike]) -> str:
    """Render aggregate voiceprint match scores."""
    matches = [speaker.match for speaker in speakers if speaker.match is not None]
    matched = sum(1 for match in matches if voiceprint_match_status(match) == MATCH_STATUS_MATCHED)
    below = sum(1 for match in matches if voiceprint_match_status(match) == MATCH_STATUS_BELOW_THRESHOLD)
    no_candidate = sum(1 for match in matches if voiceprint_match_status(match) == MATCH_STATUS_NO_CANDIDATE)
    scores = [score for match in matches if (score := best_candidate_score(match)) is not None]
    score_summary = _score_summary(scores)
    return (
        f"[b]Auto[/b]     matched {matched}/{len(speakers)} | below-threshold {below} | "
        f"no-candidate {no_candidate} | {score_summary}"
    )


def _risk_overview_line(speakers: Sequence[ReviewSpeakerLike], selected: ReviewSpeakerLike) -> str:
    """Render conflict, mismatch, and selected speaker state."""
    conflicts = sum(1 for speaker in speakers if has_conflict(speaker))
    mismatches = sum(1 for speaker in speakers if has_mismatch(speaker))
    ignored = sum(1 for speaker in speakers if is_ignored(speaker))
    risk_style = "bold red" if conflicts else "yellow" if mismatches else "green"
    return (
        f"[b]Check[/b]    [{risk_style}]conflict {conflicts} | mismatch {mismatches}[/] | "
        f"ignored {ignored} | "
        f"selected {escape(selected.label)}: {speaker_status(selected)} | {match_badge(selected)}"
    )


def _output_overview_line() -> str:
    """Render the final project artifacts users should look for."""
    return "[b]Output[/b]   final: exports/transcript_named.txt + exports/subtitle_named.srt"


def _next_action_line(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render the most useful next action from current state."""
    if any(has_conflict(speaker) for speaker in speakers):
        return "[bold red]Next[/]     resolve conflicts before saving."
    if not overview.match_file_exists:
        return "[yellow]Next[/]     run `meeting-asr project speakers match`."
    if _manual_saved_count(speakers, overview) < len(speakers):
        return "[yellow]Next[/]     review speakers, then press `s` to save."
    if _has_unsaved_names(speakers, overview):
        return "[yellow]Next[/]     press `s` to write the updated speaker map."
    if _capture_todo_count(speakers, overview.voiceprint):
        return "[green]Done[/]     project outputs ready. Optional voiceprint: `meeting-asr voiceprint review PROJECT_ID`."
    embed_todo = _embed_todo_count(overview.voiceprint)
    if embed_todo is None and overview.voiceprint.captured_sample_ids:
        return (
            "[green]Done[/]     project outputs ready. Optional voiceprint: "
            "fix voiceprint embedding config, then `meeting-asr voiceprint embed`."
        )
    if embed_todo:
        return "[green]Done[/]     project outputs ready. Optional voiceprint: `meeting-asr voiceprint embed`."
    return (
        "[green]Done[/]     preview: `meeting-asr project speakers preview`; "
        "read: `meeting-asr project transcript show`."
    )


def _manual_state(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render manual review save and naming progress."""
    saved = _manual_saved_count(speakers, overview)
    named = sum(1 for speaker in speakers if speaker.current_name != speaker.label)
    ignored = sum(1 for speaker in speakers if is_ignored(speaker))
    total = len(speakers)
    state = "saved" if saved == total else "partial" if saved else "pending"
    style = "green" if saved == total else "yellow" if saved else "red"
    return f"[{style}]{state} {saved}/{total}[/], named {named}/{total}, ignored {ignored}"


def _manual_saved_count(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> int:
    """Return how many project speakers have been saved in speaker_map.json."""
    speaker_ids = {speaker.speaker_id for speaker in speakers}
    return len(speaker_ids & set(overview.saved_names_by_speaker))


def _capture_todo_count(speakers: Sequence[ReviewSpeakerLike], progress: VoiceprintReviewProgress) -> int:
    """Return how many named speakers still need project voiceprint capture."""
    return sum(1 for speaker in speakers if _needs_capture(speaker, progress))


def _needs_capture(speaker: ReviewSpeakerLike, progress: VoiceprintReviewProgress) -> bool:
    """Return whether a speaker's current name has no captured clip yet."""
    if speaker.current_name == speaker.label:
        return False
    captured_names = progress.captured_names_by_speaker.get(speaker.speaker_id, frozenset())
    return speaker.current_name not in captured_names


def _embed_state(progress: VoiceprintReviewProgress) -> str:
    """Render project-scoped voiceprint embedding progress."""
    embed_todo = _embed_todo_count(progress)
    if embed_todo is None:
        reason = _trim_status_text(progress.embed_error or "unknown config", limit=48)
        return f"Embed=[yellow]unknown[/] {escape(reason)}"
    embedded = len(progress.captured_sample_ids & (progress.embedded_sample_ids or frozenset()))
    style = "green" if embed_todo == 0 else "yellow"
    return f"Embed=[{style}]todo {embed_todo}[/], embedded {embedded}/{len(progress.captured_sample_ids)}"


def _embed_todo_count(progress: VoiceprintReviewProgress) -> int | None:
    """Return how many captured project samples need embedding."""
    if progress.embedded_sample_ids is None:
        return None
    return len(progress.captured_sample_ids - progress.embedded_sample_ids)


def _has_unsaved_names(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> bool:
    """Return whether the current TUI names differ from saved names."""
    saved_names = overview.saved_names_by_speaker
    return any(saved_names.get(speaker.speaker_id) != speaker.current_name for speaker in speakers)


def _score_summary(scores: list[float]) -> str:
    """Render average and best match score."""
    if not scores:
        return "score avg -, best -"
    average = sum(scores) / len(scores)
    return f"score avg {average:.3f}, best {max(scores):.3f}"


def _badge(label: str, value: str, good: bool) -> str:
    """Render a colored key/value workflow badge."""
    style = "green" if good else "yellow"
    return f"{label}=[{style}]{escape(value)}[/]"


def _trim_status_text(text: str, *, limit: int) -> str:
    """Trim status text so the overview pane stays stable."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
