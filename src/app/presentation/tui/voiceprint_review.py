"""Unified Textual UI for project voiceprint capture and global library review."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from typing import Literal

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import Header, Static

from app.presentation.tui.voiceprint import (
    VoiceprintLibrarySession,
    VoiceprintSpeakerEntry,
    load_voiceprint_library_session,
)
from app.presentation.tui.voiceprint_capture import (
    VoiceprintCaptureClipEntry,
    VoiceprintCaptureReviewSession,
    VoiceprintCaptureSpeakerEntry,
    load_voiceprint_capture_review_session,
)
from app.presentation.tui.i18n import tr
from app.presentation.tui.voiceprint_review_context import (
    load_project_voiceprint_context,
)
from app.presentation.tui.voiceprint_review_render import (
    capture_count_markup,
    clamp,
    last_sample_page_start,
    library_sample_line,
    library_sample_summary,
    library_speaker_summary,
    mode_label,
    next_view_label,
    page_footer,
    project_match_score_markup,
    project_sample_line,
    project_sample_summary,
    project_sample_time_range,
    project_speaker_summary,
    sample_page_start,
    trim_text,
)
from app.presentation.tui.voiceprint_quality import (
    _changed_statuses as quality_changed_statuses,
    _initial_statuses as quality_initial_statuses,
    _person_style as quality_person_style,
    _person_index as quality_person_index,
    _sample_style as quality_sample_style,
)
from app.presentation.tui.speaker_matches import load_match_candidates
from app.presentation.tui.voiceprint_review_text import help_text, quality_reason_text, status_text
from app.presentation.tui.voiceprint_review_workflow import (
    VoiceprintReviewProcessingScreen,
    VoiceprintReviewResultScreen,
    VoiceprintReviewWorkflowSummary,
    compact_evaluation_line,
    compact_workflow_line,
    run_voiceprint_review_workflow,
)
from app.speaker_review import build_audio_preview_command
from app.utils import format_ms_timestamp
from app.voiceprint_audio import voiceprint_playback_clip_path
from app.voiceprint_evaluation import VoiceprintEvaluationSummary, evaluate_voiceprint_embedding
from app.voiceprint_playback import build_voiceprint_play_command
from app.voiceprint_quality import (
    VOICEPRINT_SAMPLE_STATUS_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_QUARANTINED,
    VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE,
    VoiceprintQualityPerson,
    VoiceprintQualityReport,
    VoiceprintQualitySample,
    analyze_voiceprint_quality,
)
from app.voiceprint_store import VoiceprintSampleRow, get_voiceprint_db_path, update_voiceprint_sample_status
from app.voiceprints import VoiceprintCaptureSummary, plan_voiceprint_capture

Mode = Literal["project", "library", "quality"]
Column = Literal["speakers", "samples"]

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS: tuple[Column, Column] = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"
PROJECT_MODE = "project"
LIBRARY_MODE = "library"
QUALITY_MODE = "quality"


@dataclass(frozen=True, slots=True)
class VoiceprintReviewSession:
    """Inputs used by the unified voiceprint review TUI."""

    capture: VoiceprintCaptureReviewSession | None
    library: VoiceprintLibrarySession
    quality: VoiceprintQualityReport
    store_dir: Path | None = None
    quality_model: str | None = None
    initial_mode: Mode = PROJECT_MODE
    return_hint: str = "quit"


@dataclass(frozen=True, slots=True)
class VoiceprintReviewDecision:
    """Result returned by the unified voiceprint review TUI."""

    saved: bool
    selected_clip_rel_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Current sample playback state for human-visible progress."""

    view: Mode
    key: str
    label: str
    started_at: float
    duration_seconds: float | None


class VoiceprintReviewHelpScreen(ModalScreen[None]):
    """Modal shortcut help for the unified voiceprint review TUI."""

    CSS = """
    VoiceprintReviewHelpScreen {
        align: center middle;
    }
    #voiceprint-review-help {
        width: 88;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("q", "close_help", "Close"),
        Binding("?", "close_help", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Build the help popup."""
        yield Static(help_text(), id="voiceprint-review-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class _VoiceprintReviewBase:
    """Shared controller for the voiceprint review app and embeddable screen."""

    CSS = """
    Screen {
        layout: vertical;
        background: #0f1117;
        color: #d9e2ec;
    }
    #overview {
        border: heavy #ffb000;
        background: #151922;
        color: #d9e2ec;
        height: 10;
        padding: 0 1;
    }
    #main {
        height: 1fr;
    }
    .pane {
        border: round #3d4758;
        background: #10151e;
        height: 100%;
        padding: 0 1;
    }
    .pane.focused-pane {
        border: heavy #00d1ff;
        background: #18202b;
    }
    .pane.unfocused-pane {
        border: round #3d4758;
        color: #9aa7b8;
    }
    #speakers {
        width: 34%;
    }
    #samples {
        width: 66%;
    }
    #status {
        height: 2;
        background: #0b0f14;
        color: #8fb3ff;
    }
    """

    BINDINGS = [
        Binding("tab", "switch_mode", "Switch view", priority=True),
        Binding("p", "project_mode", "Project"),
        Binding("g", "library_mode", "Library"),
        Binding("y", "quality_mode", "Quality"),
        Binding("j", "down", "Down"),
        Binding("k", "up", "Up"),
        Binding("down", "down", "Down", show=False),
        Binding("up", "up", "Up", show=False),
        Binding("h", "left", "Left"),
        Binding("l", "right", "Right"),
        Binding("left", "left", "Left", show=False),
        Binding("right", "right", "Right", show=False),
        Binding("pagedown", "next_sample_page", "Next page"),
        Binding("pageup", "previous_sample_page", "Previous page"),
        Binding("]", "next_sample_page", "Next page", show=False),
        Binding("[", "previous_sample_page", "Previous page", show=False),
        Binding("space", "play_sample", "Play/stop sample"),
        Binding("x", "toggle_sample", "Include/exclude"),
        Binding("a", "toggle_speaker", "Toggle speaker"),
        Binding("d", "exclude_speaker", "Exclude speaker"),
        Binding("r", "mark_quality_quarantined", "Quarantine", show=False),
        Binding("v", "mark_quality_verified", "Verify sample", show=False),
        Binding("s", "save", "Save selected"),
        Binding("u", "refresh_quality", "Refresh quality"),
        Binding("e", "evaluate", "Evaluate"),
        Binding("?", "show_shortcuts", "Help"),
        Binding("escape", "quit", "Back", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session: VoiceprintReviewSession) -> None:
        """
        Create the unified voiceprint review app.

        Args:
            session: Project capture and global library data.
        """
        super().__init__()
        self.session = session
        self.mode = self._initial_mode(session)
        self.project_selected_speaker_index = 0
        self.project_focused_column: Column = "speakers"
        self.library_selected_speaker_index = 0
        self.library_focused_column: Column = "speakers"
        self.quality_selected_person_index = 0
        self.quality_selected_sample_indices = {
            person.speaker_public_id: 0 for person in session.quality.people
        }
        self.quality_focused_column: Column = "speakers"
        self.quality_statuses = quality_initial_statuses(session.quality)
        self.playback_process: subprocess.Popen | None = None
        self.playback_state: PlaybackState | None = None
        self.playback_timer: Any | None = None
        self.workflow_summary: VoiceprintReviewWorkflowSummary | None = None
        self.evaluation_summary: VoiceprintEvaluationSummary | None = None

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(status_text(), id="status")

    def on_mount(self) -> None:
        """Render the initial view."""
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when the TUI closes."""
        self._stop_playback()

    def action_switch_mode(self) -> None:
        """Switch between project candidates, library, and quality review."""
        self._stop_playback()
        self.mode = self._next_mode()
        self._set_status(tr(f"Switched to {mode_label(self.mode)}.", f"已切换到{mode_label(self.mode)}。"))
        self._refresh()

    def action_project_mode(self) -> None:
        """Jump to the project capture view."""
        if self.session.capture is None:
            self._set_status(tr("No project was provided, so project candidates are unavailable.", "没有提供项目，因此项目候选样本不可用。"))
            return
        self._switch_to(PROJECT_MODE)

    def action_library_mode(self) -> None:
        """Jump to the global library view."""
        self._switch_to(LIBRARY_MODE)

    def action_quality_mode(self) -> None:
        """Jump to the global quality review view."""
        self._switch_to(QUALITY_MODE)

    def action_down(self) -> None:
        """Move down in the focused column."""
        self._move_focused_row(1)

    def action_up(self) -> None:
        """Move up in the focused column."""
        self._move_focused_row(-1)

    def action_left(self) -> None:
        """Focus the previous column."""
        self._move_column(-1)

    def action_right(self) -> None:
        """Focus the next column."""
        self._move_column(1)

    def action_next_sample_page(self) -> None:
        """Select the first sample on the next page."""
        self._move_sample_page(1)

    def action_previous_sample_page(self) -> None:
        """Select the first sample on the previous page."""
        self._move_sample_page(-1)

    def action_toggle_sample(self) -> None:
        """Include or exclude the selected project sample."""
        if self.mode == QUALITY_MODE:
            self._toggle_quality_sample()
            return
        if self.mode != PROJECT_MODE:
            self._set_status(tr("Sample selection only applies to project candidates.", "样本选择只适用于项目候选样本。"))
            return
        sample = self._project_selected_sample()
        if sample is None:
            return
        sample.included = not sample.included
        self._set_status(
            tr("Sample included.", "已选中样本。") if sample.included else tr("Sample excluded.", "已排除样本。")
        )
        self._refresh()

    def action_toggle_speaker(self) -> None:
        """Include or exclude all samples for the selected project speaker."""
        if self.mode == QUALITY_MODE:
            self._mark_quality_sample(VOICEPRINT_SAMPLE_STATUS_ACTIVE)
            return
        if self.mode != PROJECT_MODE:
            self._set_status(tr("Speaker selection only applies to project candidates.", "speaker 选择只适用于项目候选样本。"))
            return
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        include = not all(clip.included for clip in speaker.clips)
        for clip in speaker.clips:
            clip.included = include
        self._set_status(
            tr(
                f"{'Included' if include else 'Excluded'} all samples for {speaker.name}.",
                f"已{'选中' if include else '排除'} {speaker.name} 的全部样本。",
            )
        )
        self._refresh()

    def action_exclude_speaker(self) -> None:
        """Exclude all samples for the selected project speaker."""
        if self.mode != PROJECT_MODE:
            self._set_status(tr("Speaker exclusion only applies to project candidates.", "取消 speaker 只适用于项目候选样本。"))
            return
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        for clip in speaker.clips:
            clip.included = False
        self._set_status(tr(f"Excluded all samples for {speaker.name}.", f"已取消 {speaker.name} 的全部样本。"))
        self._refresh()

    def action_mark_quality_quarantined(self) -> None:
        """Mark the selected quality sample as quarantined."""
        if self.mode != QUALITY_MODE:
            self._set_status(tr("Quarantine applies only to Quality review.", "隔离只适用于质量检查视图。"))
            return
        self._mark_quality_sample(VOICEPRINT_SAMPLE_STATUS_QUARANTINED)

    def action_mark_quality_verified(self) -> None:
        """Mark the selected quality sample as human-verified active."""
        if self.mode != QUALITY_MODE:
            self._set_status(tr("Verification applies only to Quality review.", "人工确认只适用于质量检查视图。"))
            return
        self._mark_quality_sample(VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE)

    def action_play_sample(self) -> None:
        """Play or stop the selected sample for the active view."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止 sample 播放。"))
            self._refresh()
            return
        try:
            self._play_active_sample()
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Playback failed: {exc}", f"播放失败：{exc}"))

    def action_save(self) -> None:
        """Capture and embed selected project clips, or return them to the caller."""
        if self.mode == QUALITY_MODE:
            self._save_quality_changes()
            return
        if self.mode != PROJECT_MODE:
            self._set_status(tr("Switch to Project candidates before saving new samples.", "保存新样本前请先切换到项目候选样本。"))
            return
        if self.session.capture is None:
            self._set_status(tr("No project candidates to capture in this session.", "当前会话没有可采集的项目候选样本。"))
            return
        selected = self._selected_clip_rel_paths()
        if not selected:
            self._set_status(tr("No samples selected. Toggle at least one sample before capture.", "没有选中样本。采集前至少选中一个样本。"))
            return
        self._save_selected_clip_paths(frozenset(selected))

    def action_evaluate(self) -> None:
        """Evaluate current voiceprint matching impact."""
        self._set_status(tr("Evaluation is available after this screen is embedded in Project Review.", "评测能力需要从 Project Review 内进入后使用。"))

    def action_refresh_quality(self) -> None:
        """Refresh quality scores from SQLite."""
        if self.mode != QUALITY_MODE:
            self._set_status(tr("Switch to Quality review before refreshing quality scores.", "刷新质量评分前请先切换到质量检查。"))
            return
        if quality_changed_statuses(self.session.quality, self.quality_statuses):
            self._set_status(tr("Save staged quality changes before refreshing.", "刷新前请先保存暂存的质量变更。"))
            return
        self._reload_quality(status=tr("Quality scores refreshed.", "质量评分已刷新。"))

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.app.push_screen(VoiceprintReviewHelpScreen())

    def action_quit(self) -> None:
        """Exit without writing project samples."""
        self._finish(VoiceprintReviewDecision(False, frozenset()))

    def _finish(self, decision: VoiceprintReviewDecision) -> None:
        """Return the review decision to the active Textual host."""
        raise NotImplementedError

    def _save_selected_clip_paths(self, selected_clip_rel_paths: frozenset[str]) -> None:
        """Handle selected project clip paths."""
        self._finish(VoiceprintReviewDecision(True, selected_clip_rel_paths))

    def _switch_to(self, mode: Mode) -> None:
        """Switch to a specific mode and refresh the screen."""
        if self.mode == mode:
            return
        self._stop_playback()
        self.mode = mode
        self._set_status(tr(f"Switched to {mode_label(mode)}.", f"已切换到{mode_label(mode)}。"))
        self._refresh()

    def _next_mode(self) -> Mode:
        """Return the next visible voiceprint review mode."""
        if self.mode == PROJECT_MODE:
            return LIBRARY_MODE
        if self.mode == LIBRARY_MODE:
            return QUALITY_MODE
        if self.session.capture is not None:
            return PROJECT_MODE
        return LIBRARY_MODE

    def _move_focused_row(self, delta: int) -> None:
        """Move selection inside the focused column."""
        if self._focused_column() == "speakers":
            self._move_speaker(delta)
            return
        self._move_sample(delta)

    def _move_column(self, delta: int) -> None:
        """Move focus between speaker and sample panes."""
        index = COLUMNS.index(self._focused_column())
        self._set_focused_column(COLUMNS[clamp(index + delta, 0, len(COLUMNS) - 1)])
        self._refresh()

    def _move_speaker(self, delta: int) -> None:
        """Move the selected speaker in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_speaker(delta)
            return
        if self.mode == QUALITY_MODE:
            self._move_quality_person(delta)
            return
        self._move_library_speaker(delta)

    def _move_sample(self, delta: int) -> None:
        """Move the selected sample in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_sample(delta)
            return
        if self.mode == QUALITY_MODE:
            self._move_quality_sample(delta)
            return
        self._move_library_sample(delta)

    def _move_sample_page(self, delta: int) -> None:
        """Move the selected sample page in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_sample_page(delta)
            return
        if self.mode == QUALITY_MODE:
            self._move_quality_sample_page(delta)
            return
        self._move_library_sample_page(delta)

    def _refresh(self) -> None:
        """Refresh all visible panes from current state."""
        self._refresh_focus_styles()
        self.query_one("#overview", Static).update(self._overview_pane())
        self.query_one("#speakers", Static).update(self._speaker_pane())
        self.query_one("#samples", Static).update(self._sample_pane())

    def _refresh_focus_styles(self) -> None:
        """Make the focused pane visually obvious."""
        focused_column = self._focused_column()
        for column in COLUMNS:
            pane = self.query_one(f"#{column}", Static)
            focused = column == focused_column
            pane.set_class(focused, FOCUSED_PANE_CLASS)
            pane.set_class(not focused, UNFOCUSED_PANE_CLASS)

    def _overview_pane(self) -> str:
        """Render the overview for the active view."""
        if self.mode == PROJECT_MODE:
            return self._project_overview_pane()
        if self.mode == QUALITY_MODE:
            return self._quality_overview_pane()
        return self._library_overview_pane()

    def _speaker_pane(self) -> str:
        """Render speakers for the active view."""
        if self.mode == PROJECT_MODE:
            return self._project_speaker_pane()
        if self.mode == QUALITY_MODE:
            return self._quality_speaker_pane()
        return self._library_speaker_pane()

    def _sample_pane(self) -> str:
        """Render samples for the active view."""
        if self.mode == PROJECT_MODE:
            return self._project_sample_pane()
        if self.mode == QUALITY_MODE:
            return self._quality_sample_pane()
        return self._library_sample_pane()

    def _project_overview_pane(self) -> str:
        """Render project capture counts and selection state."""
        capture = self.session.capture
        if capture is None:
            return tr(
                "[b]Mode[/b]     Global library only\n[yellow]No project candidates loaded.[/]",
                "[b]模式[/b]     仅全局声纹库\n[yellow]未加载项目候选样本。[/]",
            )
        selected = len(self._selected_clip_rel_paths())
        total = sum(len(speaker.clips) for speaker in capture.speakers)
        speaker = self._project_speaker()
        sample = self._project_selected_sample()
        title = trim_text(capture.project_title or capture.project_id, limit=72)
        source_name = trim_text(capture.source_name or capture.source_path.name, limit=72)
        meeting_time = capture.meeting_time or "-"
        lines = [
            self._mode_line(),
            f"{tr('[b]Project[/b]', '[b]项目[/b]')}  {escape(title)} [dim]({escape(capture.project_id)})[/]",
            tr(
                f"[b]State[/b]    status={escape(capture.project_status or '-')} | time={escape(meeting_time)}",
                f"[b]状态[/b]     项目状态={escape(capture.project_status or '-')} | 时间={escape(meeting_time)}",
            ),
            f"{tr('[b]Source[/b]', '[b]来源[/b]')}   {escape(source_name)}",
            tr(
                "[b]Goal[/b]     verify samples, press s to capture, embed, and evaluate",
                "[b]目标[/b]     确认样本，按 s 采集、生成 embedding 并评测",
            ),
            tr(f"[b]Selected[/b] {selected}/{total} candidate sample(s)", f"[b]已选[/b]     {selected}/{total} 个候选样本"),
            tr(f"[b]Focus[/b]    {escape(project_speaker_summary(speaker))}", f"[b]当前[/b]     {escape(project_speaker_summary(speaker))}"),
            tr(f"[b]Sample[/b]   {escape(project_sample_summary(sample))}", f"[b]样本[/b]     {escape(project_sample_summary(sample))}"),
        ]
        if self.workflow_summary is not None:
            lines[-1] = compact_workflow_line(self.workflow_summary)
        elif self.evaluation_summary is not None:
            lines[-1] = compact_evaluation_line(self.evaluation_summary)
        return "\n".join(lines)

    def _library_overview_pane(self) -> str:
        """Render global library counts and selection state."""
        speaker = self._library_speaker()
        sample = self._library_selected_sample()
        sample_count = sum(item.sample_count for item in self.session.library.speakers)
        embedded = sum(item.embedded_sample_count for item in self.session.library.speakers)
        lines = [
            self._mode_line(),
            f"{tr('[b]Store[/b]', '[b]库路径[/b]')}    {escape(str(self.session.library.db_path))}",
            tr(
                f"[b]Library[/b]  speakers {len(self.session.library.speakers)} | samples {sample_count} | embedded {embedded}/{sample_count}",
                f"[b]声纹库[/b]   speaker {len(self.session.library.speakers)} | 样本 {sample_count} | embedding {embedded}/{sample_count}",
            ),
            tr(f"[b]Focus[/b]    {escape(library_speaker_summary(speaker))}", f"[b]当前[/b]     {escape(library_speaker_summary(speaker))}"),
            tr(f"[b]Sample[/b]   {escape(library_sample_summary(sample))}", f"[b]样本[/b]     {escape(library_sample_summary(sample))}"),
        ]
        return "\n".join(lines)

    def _quality_overview_pane(self) -> str:
        """Render global voiceprint quality state."""
        report = self.session.quality
        person = self._quality_person()
        sample = self._quality_selected_sample()
        changed = len(quality_changed_statuses(report, self.quality_statuses))
        lines = [
            self._mode_line(),
            f"{tr('[b]Store[/b]', '[b]库路径[/b]')}    {escape(str(report.db_path))}",
            f"{tr('[b]Model[/b]', '[b]模型[/b]')}    {escape(report.model)}",
            tr(
                f"[b]Quality[/b] people {len(report.people)} | samples {report.sample_count} | suspicious {report.suspicious_count} | critical {report.critical_count}",
                f"[b]质量[/b]     人员 {len(report.people)} | 样本 {report.sample_count} | 可疑 {report.suspicious_count} | 严重 {report.critical_count}",
            ),
            tr(f"[b]Focus[/b]    {escape(_quality_person_summary(person))}", f"[b]当前[/b]     {escape(_quality_person_summary(person))}"),
            tr(f"[b]Sample[/b]   {escape(_quality_sample_summary(sample, self.quality_statuses))}", f"[b]样本[/b]     {escape(_quality_sample_summary(sample, self.quality_statuses))}"),
            tr(f"[b]Changes[/b]  {changed} staged", f"[b]变更[/b]     已暂存 {changed} 个"),
        ]
        return "\n".join(lines)

    def _project_speaker_pane(self) -> str:
        """Render project speakers and selected sample counts."""
        lines = [self._pane_title(tr("Project candidates", "项目候选样本"), "speakers")]
        capture = self.session.capture
        if capture is None or not capture.speakers:
            lines.append(tr("[yellow]No capture candidates.[/]", "[yellow]没有可采集候选样本。[/]"))
            return "\n".join(lines)
        for index, speaker in enumerate(capture.speakers):
            marker = ">" if index == self.project_selected_speaker_index else " "
            selected = sum(1 for clip in speaker.clips if clip.included)
            person = "" if speaker.person_public_id is None else f" [dim]{escape(speaker.person_public_id)}[/]"
            name = f"[bold]{escape(speaker.name)}[/]"
            marker_text = "[bold yellow]>[/]" if marker == ">" else "[dim] [/]"
            label = tr(
                f"{marker_text} {name}{person}  "
                f"{project_match_score_markup(speaker.match_score, label='score')}  "
                f"{capture_count_markup(selected, len(speaker.clips), label='selected')}",
                f"{marker_text} {name}{person}  "
                f"{project_match_score_markup(speaker.match_score, label='分数')}  "
                f"{capture_count_markup(selected, len(speaker.clips), label='已选')}",
            )
            lines.append(self._current_row(label) if marker == ">" else label)
        return "\n".join(lines)

    def _library_speaker_pane(self) -> str:
        """Render global voiceprint people."""
        lines = [self._pane_title(tr("Global voiceprint people", "全局声纹人员"), "speakers")]
        if not self.session.library.speakers:
            lines.append(tr("[yellow]No voiceprints recorded.[/]", "[yellow]尚未录入声纹。[/]"))
            return "\n".join(lines)
        for index, speaker in enumerate(self.session.library.speakers):
            marker = "[bold yellow]>[/]" if index == self.library_selected_speaker_index else "[dim] [/]"
            embedded = f"{speaker.embedded_sample_count}/{speaker.sample_count}"
            label = tr(
                f"{marker} [bold]{escape(speaker.name)}[/] [dim]{escape(speaker.public_id)}[/]  "
                f"[cyan]samples {speaker.sample_count}[/]  [green]embedded {embedded}[/]",
                f"{marker} [bold]{escape(speaker.name)}[/] [dim]{escape(speaker.public_id)}[/]  "
                f"[cyan]样本 {speaker.sample_count}[/]  [green]embedding {embedded}[/]",
            )
            lines.append(self._current_row(label) if index == self.library_selected_speaker_index else label)
        return "\n".join(lines)

    def _quality_speaker_pane(self) -> str:
        """Render quality people sorted by risk."""
        lines = [self._pane_title(tr("Voiceprint quality", "声纹质量"), "speakers")]
        if not self.session.quality.people:
            lines.append(tr("[yellow]No embedded voiceprint samples.[/]", "[yellow]没有已 embedding 的声纹样本。[/]"))
            return "\n".join(lines)
        for index, person in enumerate(self.session.quality.people):
            marker = "[bold yellow]>[/]" if index == self.quality_selected_person_index else "[dim] [/]"
            style = quality_person_style(person)
            mean = "-" if person.mean_score is None else f"{person.mean_score:.3f}"
            label = (
                f"{marker} [bold]{escape(person.speaker_name)}[/] [dim]{escape(person.speaker_public_id)}[/]  "
                f"[cyan]mean={mean}[/]  suspicious={person.suspicious_count}/{person.sample_count}"
            )
            styled = f"[{style}]{label}[/]" if style else label
            lines.append(self._current_row(styled) if index == self.quality_selected_person_index else styled)
        return "\n".join(lines)

    def _project_sample_pane(self) -> str:
        """Render project capture samples for the selected speaker."""
        speaker = self._project_speaker()
        title = tr("Project samples", "项目样本") if speaker is None else tr(f"{speaker.name} project samples", f"{speaker.name} 项目样本")
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append(tr("[yellow]No speaker selected.[/]", "[yellow]未选择 speaker。[/]"))
            return "\n".join(lines)
        page_start, samples = self._visible_project_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_clip_index else " "
            checked = "[green]x[/]" if sample.included else "[dim] [/]"
            marker = "[bold yellow]>[/]" if prefix == ">" else "[dim] [/]"
            playing = "[bold magenta]PLAY[/]" if self._playing_key(PROJECT_MODE) == sample.rel_path else "[dim]    [/]"
            line = f"{marker} {checked} {playing} [cyan]#{index + 1}[/] {escape(project_sample_line(sample))}"
            lines.append(self._current_row(line) if prefix == ">" else line)
        if not samples:
            lines.append(tr("[yellow]No samples for this speaker.[/]", "[yellow]当前 speaker 没有样本。[/]"))
        lines.extend(["", self._project_sample_page_footer(speaker, page_start)])
        return "\n".join(lines)

    def _library_sample_pane(self) -> str:
        """Render stored WAV samples for the selected person."""
        speaker = self._library_speaker()
        title = tr("Library samples", "声纹库样本") if speaker is None else tr(f"{speaker.name} stored samples", f"{speaker.name} 已保存样本")
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append(tr("[yellow]No speaker selected.[/]", "[yellow]未选择 speaker。[/]"))
            return "\n".join(lines)
        page_start, samples = self._visible_library_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            marker = "[bold yellow]>[/]" if prefix == ">" else "[dim] [/]"
            playing = "[bold magenta]PLAY[/]" if self._playing_key(LIBRARY_MODE) == sample.public_id else "[dim]    [/]"
            line = f"{marker} {playing} [cyan]#{index + 1}[/] {escape(library_sample_line(sample))}"
            lines.append(self._current_row(line) if prefix == ">" else line)
        if not samples:
            lines.append(tr("[yellow]No samples for this person.[/]", "[yellow]当前人员没有样本。[/]"))
        lines.extend(["", self._library_sample_page_footer(speaker, page_start)])
        return "\n".join(lines)

    def _quality_sample_pane(self) -> str:
        """Render quality sample scores and staged statuses."""
        person = self._quality_person()
        title = tr("Quality samples", "质量样本") if person is None else tr(f"{person.speaker_name} quality samples", f"{person.speaker_name} 质量样本")
        lines = [self._pane_title(title, "samples")]
        if person is None:
            lines.append(tr("[yellow]No person selected.[/]", "[yellow]未选择人员。[/]"))
            return "\n".join(lines)
        selected_index = self.quality_selected_sample_indices[person.speaker_public_id]
        page_start, samples = self._visible_quality_samples(person)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == selected_index else " "
            marker = "[bold yellow]>[/]" if prefix == ">" else "[dim] [/]"
            status = self.quality_statuses[sample.sample_public_id]
            score = "-" if sample.score is None else f"{sample.score:.3f}"
            style = quality_sample_style(sample, status)
            playing = "[bold magenta]PLAY[/]" if self._playing_key(QUALITY_MODE) == sample.sample_public_id else "[dim]    [/]"
            line = (
                f"{marker} {playing} [cyan]#{index + 1}[/] {escape(sample.sample_public_id)} "
                f"score={escape(score)} {escape(sample.label)} -> {escape(status)} | "
                f"{escape(self._quality_sample_time(sample))}"
            )
            rendered = f"[{style}]{line}[/]" if style else line
            lines.append(self._current_row(rendered) if prefix == ">" else rendered)
            if prefix == ">":
                lines.append(tr(f"  [yellow]diagnosis[/]: {escape(quality_reason_text(sample.reason))}", f"  [yellow]诊断[/]：{escape(quality_reason_text(sample.reason))}"))
                lines.append(tr(f"  [bright_black]text[/]: {escape(trim_text(sample.transcript_text, limit=120))}", f"  [bright_black]文本[/]：{escape(trim_text(sample.transcript_text, limit=120))}"))
        if not samples:
            lines.append(tr("[yellow]No samples for this person.[/]", "[yellow]当前人员没有样本。[/]"))
        lines.extend(["", page_footer("Samples", len(person.samples), page_start, self._sample_page_size())])
        return "\n".join(lines)

    def _play_active_sample(self) -> None:
        """Start playback for the selected sample in the active view."""
        if self.mode == PROJECT_MODE:
            sample = self._project_selected_sample()
            if sample is None:
                self._set_status(tr("No project sample selected.", "未选择项目样本。"))
                return
            self._play_project_sample(sample)
            return
        if self.mode == QUALITY_MODE:
            sample = self._quality_selected_sample()
            if sample is None:
                self._set_status(tr("No quality sample selected.", "未选择质量样本。"))
                return
            self._play_quality_sample(sample)
            return
        sample = self._library_selected_sample()
        if sample is None:
            self._set_status(tr("No library sample selected.", "未选择声纹库样本。"))
            return
        self._play_library_sample(sample)

    def _play_project_sample(self, sample: VoiceprintCaptureClipEntry) -> None:
        """Start source-media playback for one planned project sample."""
        capture = self.session.capture
        if capture is None:
            return
        self._stop_playback()
        command = build_audio_preview_command(
            media=capture.source_path,
            start_seconds=sample.clip_begin_time_ms / 1000,
            duration_seconds=sample.duration_seconds,
        )
        self._start_playback(
            _start_player(command),
            view=PROJECT_MODE,
            key=sample.rel_path,
            label=tr(f"project sample {project_sample_time_range(sample)}", f"项目样本 {project_sample_time_range(sample)}"),
            duration_seconds=sample.duration_seconds,
        )

    def _play_library_sample(self, sample: VoiceprintSampleRow) -> None:
        """Start WAV playback for one stored library sample."""
        self._stop_playback()
        self._start_playback(
            _start_player(build_voiceprint_play_command(sample.clip_path)),
            view=LIBRARY_MODE,
            key=sample.public_id,
            label=tr(f"{sample.speaker_name} sample {sample.public_id}", f"{sample.speaker_name} 的样本 {sample.public_id}"),
            duration_seconds=(sample.source_end_time_ms - sample.source_begin_time_ms) / 1000,
        )

    def _play_quality_sample(self, sample: VoiceprintQualitySample) -> None:
        """Start WAV playback for one quality-review sample."""
        self._stop_playback()
        clip_path = voiceprint_playback_clip_path(sample.clip_path, store_dir=self.session.store_dir)
        self._start_playback(
            _start_player(build_voiceprint_play_command(clip_path)),
            view=QUALITY_MODE,
            key=sample.sample_public_id,
            label=tr(f"quality sample {sample.sample_public_id}", f"质量样本 {sample.sample_public_id}"),
            duration_seconds=(sample.source_end_time_ms - sample.source_begin_time_ms) / 1000,
        )

    def _start_playback(
        self,
        process: subprocess.Popen,
        *,
        view: Mode,
        key: str,
        label: str,
        duration_seconds: float | None,
    ) -> None:
        """Start tracking one playback process and refresh progress."""
        self.playback_process = process
        self.playback_state = PlaybackState(view, key, label, time.monotonic(), duration_seconds)
        self._restart_playback_timer()
        self._refresh_playback_status()
        self._refresh()

    def _restart_playback_timer(self) -> None:
        """Restart the UI timer used to update playback progress."""
        self._stop_playback_timer()
        self.playback_timer = self.set_interval(0.5, self._refresh_playback_status)

    def _stop_playback_timer(self) -> None:
        """Stop the playback progress timer if it exists."""
        timer = self.playback_timer
        self.playback_timer = None
        if timer is not None:
            timer.stop()

    def _refresh_playback_status(self) -> None:
        """Refresh playback progress in the status bar."""
        state = self.playback_state
        if state is None:
            return
        if not self._is_playing():
            label = state.label
            self.playback_state = None
            self._stop_playback_timer()
            self._set_status(tr(f"Finished playing {label}.", f"已播放完成：{label}。"))
            self._refresh()
            return
        self.query_one("#status", Static).update(escape(self._playback_status_text(state)))

    def _playback_status_text(self, state: PlaybackState) -> str:
        """Render a human-readable playback progress line."""
        elapsed = max(0.0, time.monotonic() - state.started_at)
        duration = state.duration_seconds if state.duration_seconds and state.duration_seconds > 0 else None
        if duration is None:
            return tr(
                f"Playing {state.label} | elapsed {_format_duration(elapsed)} | Space stops playback.",
                f"正在播放 {state.label} | 已播放 {_format_duration(elapsed)} | Space 停止播放。",
            )
        progress = min(1.0, elapsed / duration)
        return tr(
            f"Playing {state.label} | {_format_duration(elapsed)}/{_format_duration(duration)} | {progress:.0%} | Space stops playback.",
            f"正在播放 {state.label} | {_format_duration(elapsed)}/{_format_duration(duration)} | {progress:.0%} | Space 停止播放。",
        )

    def _playing_key(self, view: Mode) -> str | None:
        """Return the playing sample key for one view."""
        state = self.playback_state
        if state is None or not self._is_playing() or state.view != view:
            return None
        return state.key

    def _stop_playback(self) -> None:
        """Stop the current playback child process if it is still running."""
        self._stop_playback_timer()
        self.playback_state = None
        process = self.playback_process
        self.playback_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def _is_playing(self) -> bool:
        """Return whether a playback child process is still running."""
        process = self.playback_process
        return process is not None and process.poll() is None

    def _move_project_speaker(self, delta: int) -> None:
        """Move the selected project speaker index."""
        capture = self.session.capture
        if capture is None or not capture.speakers:
            return
        self.project_selected_speaker_index = (self.project_selected_speaker_index + delta) % len(capture.speakers)
        self._refresh()

    def _move_library_speaker(self, delta: int) -> None:
        """Move the selected library speaker index."""
        total = len(self.session.library.speakers)
        if total == 0:
            return
        self.library_selected_speaker_index = (self.library_selected_speaker_index + delta) % total
        self._refresh()

    def _move_quality_person(self, delta: int) -> None:
        """Move the selected quality person index."""
        total = len(self.session.quality.people)
        if total == 0:
            return
        self.quality_selected_person_index = (self.quality_selected_person_index + delta) % total
        self._refresh()

    def _move_project_sample(self, delta: int) -> None:
        """Move the selected project sample index."""
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        target = speaker.selected_clip_index + delta
        speaker.selected_clip_index = clamp(target, 0, len(speaker.clips) - 1)
        self._refresh()

    def _move_library_sample(self, delta: int) -> None:
        """Move the selected library sample index."""
        speaker = self._library_speaker()
        if speaker is None or not speaker.samples:
            return
        target = speaker.selected_sample_index + delta
        speaker.selected_sample_index = clamp(target, 0, len(speaker.samples) - 1)
        self._refresh()

    def _move_quality_sample(self, delta: int) -> None:
        """Move the selected quality sample index."""
        person = self._quality_person()
        if person is None or not person.samples:
            return
        current = self.quality_selected_sample_indices[person.speaker_public_id]
        self.quality_selected_sample_indices[person.speaker_public_id] = clamp(current + delta, 0, len(person.samples) - 1)
        self._refresh()

    def _move_project_sample_page(self, delta: int) -> None:
        """Move the selected project sample page."""
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        page_size = self._sample_page_size()
        current_start = sample_page_start(speaker.selected_clip_index, page_size)
        last_start = last_sample_page_start(len(speaker.clips), page_size)
        speaker.selected_clip_index = clamp(current_start + delta * page_size, 0, last_start)
        self._refresh()

    def _move_library_sample_page(self, delta: int) -> None:
        """Move the selected library sample page."""
        speaker = self._library_speaker()
        if speaker is None or not speaker.samples:
            return
        page_size = self._sample_page_size()
        current_start = sample_page_start(speaker.selected_sample_index, page_size)
        last_start = last_sample_page_start(len(speaker.samples), page_size)
        speaker.selected_sample_index = clamp(current_start + delta * page_size, 0, last_start)
        self._refresh()

    def _move_quality_sample_page(self, delta: int) -> None:
        """Move the selected quality sample page."""
        person = self._quality_person()
        if person is None or not person.samples:
            return
        page_size = self._sample_page_size()
        selected = self.quality_selected_sample_indices[person.speaker_public_id]
        current_start = sample_page_start(selected, page_size)
        last_start = last_sample_page_start(len(person.samples), page_size)
        self.quality_selected_sample_indices[person.speaker_public_id] = clamp(current_start + delta * page_size, 0, last_start)
        self._refresh()

    def _project_speaker(self) -> VoiceprintCaptureSpeakerEntry | None:
        """Return the selected project speaker, if any."""
        capture = self.session.capture
        if capture is None or not capture.speakers:
            return None
        return capture.speakers[self.project_selected_speaker_index]

    def _library_speaker(self) -> VoiceprintSpeakerEntry | None:
        """Return the selected library speaker, if any."""
        if not self.session.library.speakers:
            return None
        return self.session.library.speakers[self.library_selected_speaker_index]

    def _quality_person(self) -> VoiceprintQualityPerson | None:
        """Return the selected quality person, if any."""
        if not self.session.quality.people:
            return None
        return self.session.quality.people[self.quality_selected_person_index]

    def _project_selected_sample(self) -> VoiceprintCaptureClipEntry | None:
        """Return the selected project sample, if any."""
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return None
        return speaker.clips[speaker.selected_clip_index]

    def _library_selected_sample(self) -> VoiceprintSampleRow | None:
        """Return the selected library sample, if any."""
        speaker = self._library_speaker()
        if speaker is None or not speaker.samples:
            return None
        return speaker.samples[speaker.selected_sample_index]

    def _quality_selected_sample(self) -> VoiceprintQualitySample | None:
        """Return the selected quality sample, if any."""
        person = self._quality_person()
        if person is None or not person.samples:
            return None
        return person.samples[self.quality_selected_sample_indices[person.speaker_public_id]]

    def _visible_project_samples(
        self,
        speaker: VoiceprintCaptureSpeakerEntry,
    ) -> tuple[int, list[VoiceprintCaptureClipEntry]]:
        """Return the current project sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = sample_page_start(speaker.selected_clip_index, page_size)
        return page_start, speaker.clips[page_start : page_start + page_size]

    def _visible_library_samples(self, speaker: VoiceprintSpeakerEntry) -> tuple[int, list[VoiceprintSampleRow]]:
        """Return the current library sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = sample_page_start(speaker.selected_sample_index, page_size)
        return page_start, speaker.samples[page_start : page_start + page_size]

    def _visible_quality_samples(self, person: VoiceprintQualityPerson) -> tuple[int, list[VoiceprintQualitySample]]:
        """Return the current quality sample page start and rows."""
        page_size = self._sample_page_size()
        selected = self.quality_selected_sample_indices[person.speaker_public_id]
        page_start = sample_page_start(selected, page_size)
        return page_start, list(person.samples[page_start : page_start + page_size])

    def _toggle_quality_sample(self) -> None:
        """Toggle the selected quality sample active/quarantined."""
        sample = self._quality_selected_sample()
        if sample is None:
            return
        current = self.quality_statuses[sample.sample_public_id]
        status = VOICEPRINT_SAMPLE_STATUS_ACTIVE
        if current in {VOICEPRINT_SAMPLE_STATUS_ACTIVE, VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE}:
            status = VOICEPRINT_SAMPLE_STATUS_QUARANTINED
        self._set_quality_sample_status(sample, status)

    def _mark_quality_sample(self, status: str) -> None:
        """Set the selected quality sample lifecycle status."""
        sample = self._quality_selected_sample()
        if sample is None:
            return
        self._set_quality_sample_status(sample, status)

    def _set_quality_sample_status(self, sample: VoiceprintQualitySample, status: str) -> None:
        """Stage one quality sample status change."""
        self.quality_statuses[sample.sample_public_id] = status
        self._set_status(tr(f"Set {sample.sample_public_id} to {status}.", f"已把 {sample.sample_public_id} 标记为 {status}。"))
        self._refresh()

    def _save_quality_changes(self) -> None:
        """Persist staged quality status changes and refresh scores."""
        changes = quality_changed_statuses(self.session.quality, self.quality_statuses)
        if not changes:
            self._set_status(tr("No staged quality changes to save.", "没有待保存的声纹质量变更。"))
            return
        for sample_id, status in changes.items():
            update_voiceprint_sample_status(sample_id, status, get_voiceprint_db_path(self.session.store_dir))
        self._reload_quality(status=tr(f"Saved {len(changes)} quality change(s).", f"已保存 {len(changes)} 个质量变更。"))

    def _reload_quality(self, *, status: str) -> None:
        """Reload quality and library data while preserving the current person."""
        previous_person_id = self._quality_person().speaker_public_id if self._quality_person() is not None else None
        store_dir = self.session.store_dir
        quality = analyze_voiceprint_quality(store_dir=store_dir, model=self.session.quality_model)
        self.session = replace(
            self.session,
            quality=quality,
            library=load_voiceprint_library_session(store_dir=store_dir, page_size=self.session.library.page_size),
        )
        self.quality_selected_person_index = quality_person_index(self.session.quality, previous_person_id)
        self.quality_selected_sample_indices = {person.speaker_public_id: 0 for person in self.session.quality.people}
        self.quality_statuses = quality_initial_statuses(self.session.quality)
        self._refresh()
        self._set_status(status)

    def _sample_page_size(self) -> int:
        """Return the number of sample rows that fit in the sample pane."""
        configured = self._configured_page_size()
        if configured is not None:
            return max(1, configured)
        pane_height = self.query_one("#samples", Static).size.height
        if pane_height <= SAMPLE_PANE_RESERVED_ROWS:
            return DEFAULT_SAMPLE_PAGE_SIZE
        return max(1, pane_height - SAMPLE_PANE_RESERVED_ROWS)

    def _configured_page_size(self) -> int | None:
        """Return the active view page-size override, if configured."""
        if self.mode == PROJECT_MODE and self.session.capture is not None:
            return self.session.capture.page_size
        if self.mode == QUALITY_MODE:
            return None
        return self.session.library.page_size

    def _project_sample_page_footer(self, speaker: VoiceprintCaptureSpeakerEntry, page_start: int) -> str:
        """Render project sample pagination status."""
        return page_footer("Samples", len(speaker.clips), page_start, self._sample_page_size())

    def _library_sample_page_footer(self, speaker: VoiceprintSpeakerEntry, page_start: int) -> str:
        """Render library sample pagination status."""
        return page_footer("Samples", len(speaker.samples), page_start, self._sample_page_size())

    def _quality_sample_time(self, sample: VoiceprintQualitySample) -> str:
        """Render a quality sample's source time range."""
        start = format_ms_timestamp(sample.source_begin_time_ms)
        end = format_ms_timestamp(sample.source_end_time_ms)
        return f"{sample.project_id} {start}-{end}"

    def _pane_title(self, title: str, column: Column) -> str:
        """Render a pane title with focused-column state."""
        escaped = escape(title)
        if self._focused_column() == column:
            return f"[reverse][b] FOCUS [/b][/] [bold cyan]{escaped}[/]"
        return f"[dim]  {escaped}[/dim]"

    def _current_row(self, line: str) -> str:
        """Render the active row with a stable, visible background."""
        return f"[reverse]{line}[/]"

    def _mode_line(self) -> str:
        """Render active mode and switch affordance."""
        next_view = next_view_label(self.mode, self.session.capture is not None)
        next_text = tr("no alternate project view", "没有可切换的项目视图") if next_view is None else f"Tab -> {next_view}"
        return (
            "[reverse][b] VOICEPRINT REVIEW [/b][/]  "
            + tr("view=", "视图=")
            + f"[bold cyan]{mode_label(self.mode)}[/] | {next_text} | Esc/q: {escape(self.session.return_hint)}"
        )

    def _focused_column(self) -> Column:
        """Return the focused column for the active view."""
        if self.mode == PROJECT_MODE:
            return self.project_focused_column
        if self.mode == QUALITY_MODE:
            return self.quality_focused_column
        return self.library_focused_column

    def _set_focused_column(self, column: Column) -> None:
        """Set the focused column for the active view."""
        if self.mode == PROJECT_MODE:
            self.project_focused_column = column
            return
        if self.mode == QUALITY_MODE:
            self.quality_focused_column = column
            return
        self.library_focused_column = column

    def _set_status(self, message: str) -> None:
        """Show a short status message."""
        self.query_one("#status", Static).update(escape(message))

    def _selected_clip_rel_paths(self) -> set[str]:
        """Return selected project clip relative paths."""
        capture = self.session.capture
        if capture is None:
            return set()
        return {
            clip.rel_path
            for speaker in capture.speakers
            for clip in speaker.clips
            if clip.included
        }

    @staticmethod
    def _initial_mode(session: VoiceprintReviewSession) -> Mode:
        """Return a valid initial view for the session."""
        if session.initial_mode == PROJECT_MODE and session.capture is not None:
            return PROJECT_MODE
        if session.initial_mode == QUALITY_MODE:
            return QUALITY_MODE
        return LIBRARY_MODE


class VoiceprintReviewApp(_VoiceprintReviewBase, App[VoiceprintReviewDecision]):
    """Standalone voiceprint review app used by the CLI command."""

    CSS = _VoiceprintReviewBase.CSS
    BINDINGS = _VoiceprintReviewBase.BINDINGS

    def _finish(self, decision: VoiceprintReviewDecision) -> None:
        """Exit the standalone app with a review decision."""
        self.exit(decision)


class VoiceprintReviewScreen(_VoiceprintReviewBase, ModalScreen[VoiceprintReviewDecision]):
    """Embeddable voiceprint review screen used by project review."""

    CSS = _VoiceprintReviewBase.CSS
    BINDINGS = _VoiceprintReviewBase.BINDINGS

    def __init__(
        self,
        session: VoiceprintReviewSession,
        *,
        project_dir: Path | None = None,
        planned: VoiceprintCaptureSummary | None = None,
        store_dir: Path | None = None,
        on_complete: Callable[[VoiceprintReviewWorkflowSummary], None] | None = None,
    ) -> None:
        """
        Create an embeddable voiceprint review screen.

        Args:
            session: Unified voiceprint review inputs.
            project_dir: Project root when capture should run inside this screen.
            planned: Dry-run capture plan to persist on save.
            store_dir: Optional voiceprint store directory.
            on_complete: Callback after capture, embedding, and evaluation finish.
        """
        super().__init__(session)
        self.project_dir = project_dir
        self.planned = planned
        self.store_dir = store_dir
        self.on_complete = on_complete
        self.processing_screen: VoiceprintReviewProcessingScreen | None = None

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle embedded capture, embedding, and evaluation workers."""
        if event.worker.group == "voiceprint-review-workflow":
            self._handle_workflow_state(event)
            return
        if event.worker.group == "voiceprint-review-evaluate":
            self._handle_evaluation_state(event)
            return

    def _finish(self, decision: VoiceprintReviewDecision) -> None:
        """Dismiss the screen with a review decision."""
        self.dismiss(decision)

    def _save_selected_clip_paths(self, selected_clip_rel_paths: frozenset[str]) -> None:
        """Run capture, embedding, and evaluation inside Voiceprint Review."""
        if self.project_dir is None or self.planned is None:
            self._finish(VoiceprintReviewDecision(True, selected_clip_rel_paths))
            return
        self._show_processing()
        self.run_worker(
            lambda: run_voiceprint_review_workflow(
                project_dir=self.project_dir,
                planned=self.planned,
                selected_clip_rel_paths=selected_clip_rel_paths,
                store_dir=self.store_dir,
            ),
            group="voiceprint-review-workflow",
            name="capture-embed-evaluate",
            thread=True,
        )

    def action_evaluate(self) -> None:
        """Evaluate current project and historical speaker scores."""
        if self.project_dir is None:
            self._set_status(tr("No project is attached to this Voiceprint Review.", "当前 Voiceprint Review 没有关联项目。"))
            return
        self._show_processing()
        self.run_worker(
            lambda: evaluate_voiceprint_embedding(
                self.project_dir,
                store_dir=self.store_dir,
                provider=None,
                model=None,
            ),
            group="voiceprint-review-evaluate",
            name="evaluate",
            thread=True,
        )

    def _handle_workflow_state(self, event: Worker.StateChanged) -> None:
        """Update screen after the capture/embed/evaluate workflow finishes."""
        if event.state == WorkerState.SUCCESS:
            self._hide_processing()
            self._handle_workflow_success(event.worker.result)
            return
        if event.state == WorkerState.ERROR:
            self._hide_processing()
            self._set_status(
                tr(
                    f"Voiceprint capture/embed failed: {event.worker.error}. Run meeting-asr doctor --require-voiceprint-embedding.",
                    f"声纹采集或 embedding 失败：{event.worker.error}。请运行 meeting-asr doctor --require-voiceprint-embedding。",
                )
            )

    def _handle_evaluation_state(self, event: Worker.StateChanged) -> None:
        """Update screen after a standalone evaluation finishes."""
        if event.state == WorkerState.SUCCESS:
            self._hide_processing()
            self.evaluation_summary = event.worker.result
            self._set_status(compact_evaluation_line(self.evaluation_summary))
            self._refresh()
            return
        if event.state == WorkerState.ERROR:
            self._hide_processing()
            self._set_status(tr(f"Voiceprint evaluation failed: {event.worker.error}", f"声纹评测失败：{event.worker.error}"))

    def _handle_workflow_success(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """Ask the user whether to accept or roll back pending voiceprint changes."""
        self.app.push_screen(VoiceprintReviewResultScreen(summary), lambda accepted: self._handle_workflow_decision(summary, accepted))

    def _handle_workflow_decision(self, summary: VoiceprintReviewWorkflowSummary, accepted: bool | None) -> None:
        """Apply the TUI state for an accepted or rolled-back workflow."""
        if accepted:
            self._accept_workflow(summary)
            return
        self._reject_workflow(summary)

    def _accept_workflow(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """Refresh library state after the pending workflow is accepted."""
        self.workflow_summary = summary
        self.evaluation_summary = summary.evaluation
        self._reload_library_and_quality(summary.capture.store_dir)
        if self.on_complete is not None:
            self.on_complete(summary)
        self._set_status(compact_workflow_line(summary))
        self._refresh()

    def _reject_workflow(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """Refresh library state after the pending workflow is rolled back."""
        self.workflow_summary = None
        self.evaluation_summary = None
        self._reload_library_and_quality(summary.capture.store_dir)
        self._set_status(tr("Voiceprint changes rolled back; no new embeddings were kept.", "声纹变更已回滚；没有保留新的 embedding。"))
        self._refresh()

    def _reload_library_and_quality(self, store_dir: Path) -> None:
        """Reload the global library and quality report after workflow changes."""
        resolved_store_dir = self.store_dir or store_dir
        self.session = replace(
            self.session,
            library=load_voiceprint_library_session(
                store_dir=resolved_store_dir,
                page_size=self.session.library.page_size,
            ),
            quality=analyze_voiceprint_quality(store_dir=resolved_store_dir, model=self.session.quality_model),
            store_dir=resolved_store_dir,
        )
        self.quality_statuses = quality_initial_statuses(self.session.quality)

    def _show_processing(self) -> None:
        """Show a modal processing indicator for long-running work."""
        if self.processing_screen is not None:
            return
        self.processing_screen = VoiceprintReviewProcessingScreen()
        self.app.push_screen(self.processing_screen)

    def _hide_processing(self) -> None:
        """Dismiss the modal processing indicator if it is visible."""
        screen = self.processing_screen
        self.processing_screen = None
        if screen is None:
            return
        try:
            screen.dismiss(None)
        except Exception:  # noqa: BLE001
            pass


def load_voiceprint_review_session(
    *,
    project_dir: Path | None = None,
    sample_count: int = 3,
    max_seconds: float = 12.0,
    padding_seconds: float = 0.5,
    store_dir: Path | None = None,
    page_size: int | None = None,
    return_hint: str = "quit",
) -> tuple[VoiceprintReviewSession, VoiceprintCaptureSummary | None]:
    """
    Load project capture candidates and the global voiceprint library.

    Args:
        project_dir: Optional project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional voiceprint store directory.
        page_size: Optional TUI samples per page.
        return_hint: Human-readable q-key destination.

    Returns:
        Unified TUI session and the planned capture summary, if a project was loaded.
    """
    planned = None
    capture_session = None
    if project_dir is not None:
        context = load_project_voiceprint_context(project_dir)
        project_root = project_dir.expanduser().resolve()
        planned = plan_voiceprint_capture(
            project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
        )
        match_candidates = load_match_candidates(project_root / "speakers" / "speaker_matches.json")
        capture_session = load_voiceprint_capture_review_session(
            summary=planned,
            source_path=context.source_path,
            match_candidates=match_candidates,
            page_size=page_size,
            project_title=context.title,
            project_status=context.status,
            source_name=context.source_name,
            meeting_time=context.meeting_time,
        )
    library_session = load_voiceprint_library_session(store_dir=store_dir, page_size=page_size)
    quality_report = analyze_voiceprint_quality(store_dir=store_dir)
    return (
        VoiceprintReviewSession(
            capture=capture_session,
            library=library_session,
            quality=quality_report,
            store_dir=store_dir,
            return_hint=return_hint,
        ),
        planned,
    )


def run_voiceprint_review_tui(session: VoiceprintReviewSession) -> VoiceprintReviewDecision:
    """
    Run the unified voiceprint review app.

    Args:
        session: Unified voiceprint review inputs.

    Returns:
        User decision.
    """
    result = VoiceprintReviewApp(session).run()
    return result or VoiceprintReviewDecision(False, frozenset())


def _quality_person_summary(person: VoiceprintQualityPerson | None) -> str:
    """Return compact quality person summary."""
    if person is None:
        return "-"
    mean = "-" if person.mean_score is None else f"{person.mean_score:.3f}"
    return f"{person.speaker_name} {person.speaker_public_id} | mean {mean} | suspicious {person.suspicious_count}"


def _quality_sample_summary(sample: VoiceprintQualitySample | None, statuses: dict[str, str]) -> str:
    """Return compact quality sample summary."""
    if sample is None:
        return "-"
    score = "-" if sample.score is None else f"{sample.score:.3f}"
    return f"{sample.sample_public_id} | score {score} | {sample.label} | status {statuses[sample.sample_public_id]}"


def _start_player(command: list[str]) -> subprocess.Popen:
    """
    Start a detached sample player process.

    Args:
        command: Player command and arguments.

    Returns:
        Running child process.
    """
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _format_duration(seconds: float) -> str:
    """Render seconds as M:SS for playback progress."""
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes}:{remainder:02d}"
