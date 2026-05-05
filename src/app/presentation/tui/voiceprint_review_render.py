"""Rendering helpers for Voiceprint Review panes."""

from __future__ import annotations

from app.presentation.tui.i18n import tr
from app.presentation.tui.voiceprint import VoiceprintSpeakerEntry
from app.presentation.tui.voiceprint_capture import VoiceprintCaptureClipEntry, VoiceprintCaptureSpeakerEntry
from app.utils import format_ms_timestamp
from app.voiceprint_store import VoiceprintSampleRow

PROJECT_MODE = "project"
LIBRARY_MODE = "library"


def mode_label(mode: str) -> str:
    """Return the human-readable mode label."""
    return tr("Project candidates", "项目候选样本") if mode == PROJECT_MODE else tr("Global library", "全局声纹库")


def next_view_label(mode: str, project_available: bool) -> str | None:
    """Return the view reached by pressing Tab, if there is one."""
    if mode == LIBRARY_MODE and not project_available:
        return None
    return mode_label(LIBRARY_MODE) if mode == PROJECT_MODE else mode_label(PROJECT_MODE)


def project_speaker_summary(speaker: VoiceprintCaptureSpeakerEntry | None) -> str:
    """Render selected project speaker summary."""
    if speaker is None:
        return "-"
    selected = sum(1 for clip in speaker.clips if clip.included)
    person = "" if speaker.person_public_id is None else f" | person {speaker.person_public_id}"
    return tr(
        f"{speaker.name} speaker {speaker.speaker_id}{person} | selected {selected}/{len(speaker.clips)}",
        f"{speaker.name} speaker {speaker.speaker_id}{person} | 已选 {selected}/{len(speaker.clips)}",
    )


def library_speaker_summary(speaker: VoiceprintSpeakerEntry | None) -> str:
    """Render selected library speaker summary."""
    if speaker is None:
        return "-"
    return tr(
        f"{speaker.name} id={speaker.public_id} | samples {speaker.sample_count} | "
        f"projects {speaker.project_count} | models {speaker.embedding_model_count}",
        f"{speaker.name} id={speaker.public_id} | 样本 {speaker.sample_count} | "
        f"项目 {speaker.project_count} | 模型 {speaker.embedding_model_count}",
    )


def project_sample_summary(sample: VoiceprintCaptureClipEntry | None) -> str:
    """Render selected project sample summary."""
    if sample is None:
        return "-"
    state = tr("included", "已选中") if sample.included else tr("excluded", "已排除")
    return f"{project_sample_time_range(sample)} | {state}"


def library_sample_summary(sample: VoiceprintSampleRow | None) -> str:
    """Render selected library sample summary."""
    if sample is None:
        return "-"
    return tr(f"sample_id {sample.public_id} | clip {sample.clip_path}", f"样本ID {sample.public_id} | 文件 {sample.clip_path}")


def project_sample_line(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one project capture sample row."""
    return f"{project_sample_time_range(sample)} {trim_text(sample.text)}"


def library_sample_line(sample: VoiceprintSampleRow) -> str:
    """Render one stored library sample row."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    return f"{sample.project_id} speaker {sample.project_speaker_id} {start}-{end} {trim_text(sample.transcript_text)}"


def project_sample_time_range(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one project sample time range."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    return f"{start}-{end}"


def page_footer(label: str, item_count: int, page_start: int, page_size: int) -> str:
    """Render pagination status for the active sample pane."""
    page_count = sample_page_count(item_count, page_size)
    page_number = page_start // page_size + 1
    start = page_start + 1 if item_count else 0
    end = min(page_start + page_size, item_count)
    return tr(
        f"Page {page_number}/{page_count}  {label} {start}-{end}/{item_count}",
        f"第 {page_number}/{page_count} 页  {label} {start}-{end}/{item_count}",
    )


def sample_page_start(selected_index: int, page_size: int) -> int:
    """Return the first sample index for the selected sample's page."""
    return selected_index // page_size * page_size


def last_sample_page_start(sample_count: int, page_size: int) -> int:
    """Return the first sample index of the last page."""
    return max(0, (sample_count - 1) // page_size * page_size)


def sample_page_count(sample_count: int, page_size: int) -> int:
    """Return the number of sample pages."""
    return max(1, (sample_count + page_size - 1) // page_size)


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into an inclusive range."""
    return min(max(value, minimum), maximum)


def trim_text(text: str, *, limit: int = 90) -> str:
    """Trim transcript text for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
