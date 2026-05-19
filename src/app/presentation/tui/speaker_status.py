"""Status rendering helpers for the speaker review TUI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rich.markup import escape

from app.presentation.tui.i18n import tr
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
    saved_ignored_speaker_ids: frozenset[int] = frozenset()


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
        return tr("match=- ignored", "匹配=- 已忽略") if is_ignored(speaker) else tr("match=-", "匹配=-")
    state = voiceprint_match_status(speaker.match)
    display_name = accepted_match_name(speaker.match) if state == MATCH_STATUS_MATCHED else best_candidate_name(speaker.match)
    display = (
        tr("match=-", "匹配=-")
        if state == MATCH_STATUS_NO_CANDIDATE
        else tr(f"match={escape(display_name or 'unrecorded')}", f"匹配={escape(display_name or '未录入')}")
    )
    score = best_candidate_score(speaker.match)
    score_text = "-" if score is None else f"{score:.3f}"
    if is_ignored(speaker):
        state = "ignored"
    elif has_conflict(speaker):
        state = "CONFLICT"
    elif has_mismatch(speaker):
        state = "mismatch"
    return tr(f"{display} score={score_text} {state}", f"{display} 分数={score_text} {state}")


def render_match_lines(match: SpeakerMatchLike | None) -> list[str]:
    """
    Render the selected speaker's voiceprint match.

    Args:
        match: Optional voiceprint match.

    Returns:
        Lines for the identity pane.
    """
    if match is None:
        return [tr("Match: -", "匹配：-")]
    state = voiceprint_match_status(match)
    if state == MATCH_STATUS_NO_CANDIDATE:
        return [tr("Match: -", "匹配：-"), tr("Status: no-candidate", "状态：无候选")]
    name = accepted_match_name(match) if state == MATCH_STATUS_MATCHED else best_candidate_name(match)
    score = best_candidate_score(match)
    score_text = "-" if score is None else f"{score:.3f}"
    prefix = tr("Match", "匹配") if state == MATCH_STATUS_MATCHED else tr("Best candidate", "最佳候选")
    return [f"{prefix}: {escape(name or tr('unrecorded', '未录入'))}", tr(f"Score: {score_text} status={state}", f"分数：{score_text} 状态={state}")]


def render_selected_speaker_line(speaker: ReviewSpeakerLike) -> str:
    """
    Render selected speaker identity state inside the sample pane.

    Args:
        speaker: Selected speaker.

    Returns:
        Rich markup line.
    """
    status = speaker_status(speaker)
    return tr(
        f"Name: [b]{escape(speaker.current_name)}[/] | status={status} | {match_badge(speaker)}",
        f"姓名：[b]{escape(speaker.current_name)}[/] | 状态={status} | {match_badge(speaker)}",
    )


def _page_overview_line() -> str:
    """Render the active top-level TUI page."""
    return (
        "[reverse][b] PROJECT REVIEW [/b][/]  "
        + tr(
            "p: switch project | v: capture voiceprints | m: refresh diagnostics | b: embed | /: identity | e: edit text | s: save | q: quit",
            "p: 切项目 | v: 声纹采样 | m: 刷新诊断 | b: embedding | /: 身份 | e: 改文字 | s: 保存 | q: 退出",
        )
    )


def _project_overview_line(overview: SpeakerReviewOverview, speaker_count: int) -> str:
    """Render the project identity line."""
    duration = format_ms_timestamp(overview.duration_ms)
    return (
        tr("[b]Project[/b]", "[b]项目[/b]")
        + f"  {escape(overview.title)} [dim]({escape(overview.project_id)})[/] | "
        + tr(
            f"{duration} | {speaker_count} speakers | project={escape(overview.project_status)}",
            f"{duration} | {speaker_count} 个 speaker | 项目状态={escape(overview.project_status)}",
        )
    )


def _workflow_overview_line(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render project workflow progress."""
    voiceprint = overview.voiceprint
    match_state = _badge(tr("Match", "匹配"), tr("done", "完成") if overview.match_file_exists else tr("pending", "待处理"), overview.match_file_exists)
    manual_state = _manual_state(speakers, overview)
    capture_todo = _capture_todo_count(speakers, voiceprint)
    capture_state = (
        tr("Capture", "采样")
        + f"=[{'green' if capture_todo == 0 else 'yellow'}]"
        + tr(f"todo {capture_todo}", f"待处理 {capture_todo}")
        + "[/], "
        + tr(
            f"{len(voiceprint.captured_names_by_speaker)} speakers/{len(voiceprint.captured_sample_ids)} clips",
            f"{len(voiceprint.captured_names_by_speaker)} 个 speaker/{len(voiceprint.captured_sample_ids)} 个片段",
        )
    )
    return (
        tr("[b]Steps[/b]", "[b]步骤[/b]")
        + f"    1 {match_state} | 2 {tr('Names', '姓名')}={manual_state} | "
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
        tr("[b]Auto[/b]", "[b]自动[/b]")
        + tr(
            f"     matched {matched}/{len(speakers)} | below-threshold {below} | no-candidate {no_candidate} | {score_summary}",
            f"     已接受 {matched}/{len(speakers)} | 低于阈值 {below} | 无候选 {no_candidate} | {score_summary}",
        )
    )


def _risk_overview_line(speakers: Sequence[ReviewSpeakerLike], selected: ReviewSpeakerLike) -> str:
    """Render conflict, mismatch, and selected speaker state."""
    conflicts = sum(1 for speaker in speakers if has_conflict(speaker))
    mismatches = sum(1 for speaker in speakers if has_mismatch(speaker))
    ignored = sum(1 for speaker in speakers if is_ignored(speaker))
    risk_style = "bold red" if conflicts else "yellow" if mismatches else "green"
    return (
        tr("[b]Check[/b]", "[b]检查[/b]")
        + tr(
            f"    [{risk_style}]conflict {conflicts} | mismatch {mismatches}[/] | ignored {ignored} | "
            f"selected {escape(selected.label)}: {speaker_status(selected)} | {match_badge(selected)}",
            f"    [{risk_style}]冲突 {conflicts} | 不一致 {mismatches}[/] | 已忽略 {ignored} | "
            f"当前 {escape(selected.label)}: {speaker_status(selected)} | {match_badge(selected)}",
        )
    )


def _output_overview_line() -> str:
    """Render the final project artifacts users should look for."""
    return tr(
        "[b]Output[/b]   final: exports/transcript_named.txt + exports/subtitle_named.srt",
        "[b]产物[/b]     最终文本: exports/transcript_named.txt + exports/subtitle_named.srt",
    )


def _next_action_line(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render the most useful next action from current state."""
    if any(has_conflict(speaker) for speaker in speakers):
        return tr("[bold red]Next[/]     resolve conflicts before saving.", "[bold red]下一步[/]   保存前先解决冲突。")
    if not overview.match_file_exists:
        return tr("[yellow]Next[/]     run `meeting-asr project speakers match`.", "[yellow]下一步[/]   运行 `meeting-asr project speakers match`。")
    if _manual_saved_count(speakers, overview) < len(speakers):
        return tr("[yellow]Next[/]     review speakers, then press `s` to save.", "[yellow]下一步[/]   review speaker，然后按 `s` 保存。")
    if _has_unsaved_names(speakers, overview):
        return tr("[yellow]Next[/]     press `s` to write the updated speaker map.", "[yellow]下一步[/]   按 `s` 写入更新后的 speaker map。")
    if _capture_todo_count(speakers, overview.voiceprint):
        return tr(
            "[green]Done[/]     project outputs ready. Optional voiceprint: `meeting-asr voiceprint review PROJECT_ID`.",
            "[green]完成[/]     项目产物已就绪。可选声纹步骤：`meeting-asr voiceprint review PROJECT_ID`。",
        )
    embed_todo = _embed_todo_count(overview.voiceprint)
    if embed_todo is None and overview.voiceprint.captured_sample_ids:
        return (
            tr(
                "[green]Done[/]     project outputs ready. Optional voiceprint: "
                "fix voiceprint embedding config, then `meeting-asr voiceprint embed`.",
                "[green]完成[/]     项目产物已就绪。可选声纹步骤：先修复 embedding 配置，再运行 `meeting-asr voiceprint embed`。",
            )
        )
    if embed_todo:
        return tr(
            "[green]Done[/]     project outputs ready. Optional voiceprint: `meeting-asr voiceprint embed`.",
            "[green]完成[/]     项目产物已就绪。可选声纹步骤：`meeting-asr voiceprint embed`。",
        )
    return (
        tr(
            "[green]Done[/]     preview: `meeting-asr project speakers preview`; "
            "read: `meeting-asr project transcript show`.",
            "[green]完成[/]     预览：`meeting-asr project speakers preview`；查看：`meeting-asr project transcript show`。",
        )
    )


def _manual_state(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> str:
    """Render manual review save and naming progress."""
    saved = _manual_saved_count(speakers, overview)
    named = sum(1 for speaker in speakers if speaker.current_name != speaker.label)
    ignored = sum(1 for speaker in speakers if is_ignored(speaker))
    total = len(speakers)
    state = tr("saved", "已保存") if saved == total else tr("partial", "部分") if saved else tr("pending", "待处理")
    style = "green" if saved == total else "yellow" if saved else "red"
    return tr(
        f"[{style}]{state} {saved}/{total}[/], named {named}/{total}, ignored {ignored}",
        f"[{style}]{state} {saved}/{total}[/], 已命名 {named}/{total}, 已忽略 {ignored}",
    )


def _manual_saved_count(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> int:
    """Return how many project speakers have a saved name or ignore decision."""
    speaker_ids = {speaker.speaker_id for speaker in speakers}
    saved_ids = set(overview.saved_names_by_speaker) | set(overview.saved_ignored_speaker_ids)
    return len(speaker_ids & saved_ids)


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
        reason = _trim_status_text(progress.embed_error or tr("unknown config", "未知配置"), limit=48)
        return tr(f"Embed=[yellow]unknown[/] {escape(reason)}", f"Embedding=[yellow]未知[/] {escape(reason)}")
    embedded = len(progress.captured_sample_ids & (progress.embedded_sample_ids or frozenset()))
    style = "green" if embed_todo == 0 else "yellow"
    return tr(
        f"Embed=[{style}]todo {embed_todo}[/], embedded {embedded}/{len(progress.captured_sample_ids)}",
        f"Embedding=[{style}]待处理 {embed_todo}[/], 已生成 {embedded}/{len(progress.captured_sample_ids)}",
    )


def _embed_todo_count(progress: VoiceprintReviewProgress) -> int | None:
    """Return how many captured project samples need embedding."""
    if progress.embedded_sample_ids is None:
        return None
    return len(progress.captured_sample_ids - progress.embedded_sample_ids)


def _has_unsaved_names(speakers: Sequence[ReviewSpeakerLike], overview: SpeakerReviewOverview) -> bool:
    """Return whether current TUI name or ignore state differs from saved state."""
    saved_names = overview.saved_names_by_speaker
    saved_ignored = overview.saved_ignored_speaker_ids
    for speaker in speakers:
        current_ignored = is_ignored(speaker)
        if current_ignored != (speaker.speaker_id in saved_ignored):
            return True
        if current_ignored:
            continue
        if saved_names.get(speaker.speaker_id) != speaker.current_name:
            return True
    return False


def _score_summary(scores: list[float]) -> str:
    """Render average and best match score."""
    if not scores:
        return tr("score avg -, best -", "分数平均 -, 最高 -")
    average = sum(scores) / len(scores)
    return tr(f"score avg {average:.3f}, best {max(scores):.3f}", f"分数平均 {average:.3f}, 最高 {max(scores):.3f}")


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
