"""Textual UI for reviewing project voiceprint capture candidates."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Header, Static

from app.presentation.tui.i18n import tr
from app.speaker_review import build_audio_preview_command
from app.utils import format_ms_timestamp
from app.voiceprints import VoiceprintCaptureSummary

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"
def capture_review_status() -> str:
    """Return localized voiceprint-capture review status text."""
    return tr(
        (
            "Capture review: h/l or left/right choose column | j/k or up/down move | "
            "x include/exclude | space play/stop | s capture selected | ? help | q cancel"
        ),
        "采样 Review：h/l 或 ←/→ 切列 | j/k 或 ↑/↓ 移动 | x 选中/排除 | Space 播放/停止 | s 采集已选 | ? 帮助 | q 取消",
    )


def shortcut_help() -> str:
    """Return localized voiceprint-capture shortcut help."""
    return tr(
        """\
[b]Voiceprint Capture Review Shortcuts[/b]

[b]What This Does[/b]
This review happens before WAV clips are written to the global voiceprint store.
Only checked samples are captured and later embedded.

[b]Navigation[/b]
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Actions[/b]
space                Play or stop selected sample from source media
x                    Include/exclude selected sample
a                    Include/exclude all samples for the selected speaker
s                    Capture checked samples into the global voiceprint store
?                    Show this help
q                    Cancel without writing voiceprints
""",
        """\
[b]声纹采样 Review 快捷键[/b]

[b]用途[/b]
这个 review 发生在 WAV 片段写入全局声纹库之前。
只有勾选的样本会被采集，后续才会生成 embedding。

[b]导航[/b]
h/l 或 ←/→           切换当前列
j/k 或 ↑/↓           在当前列内移动
PageUp/PageDown      上一页/下一页 sample
[ / ]                上一页/下一页 sample

[b]操作[/b]
space                从源媒体播放或停止当前 sample
x                    选中/排除当前 sample
a                    选中/排除当前 speaker 的全部 sample
s                    将勾选 sample 采集到全局声纹库
?                    显示帮助
q                    取消，不写入声纹
""",
    )


@dataclass(slots=True)
class VoiceprintCaptureClipEntry:
    """Mutable TUI row for one planned capture clip."""

    rel_path: str
    source_begin_time_ms: int
    source_end_time_ms: int
    clip_begin_time_ms: int
    clip_end_time_ms: int
    text: str
    included: bool = True

    @property
    def duration_seconds(self) -> float:
        """Return clip duration in seconds."""
        return (self.clip_end_time_ms - self.clip_begin_time_ms) / 1000


@dataclass(slots=True)
class VoiceprintCaptureSpeakerEntry:
    """Mutable TUI row for one speaker's capture candidates."""

    speaker_id: int
    name: str
    person_public_id: str | None
    clips: list[VoiceprintCaptureClipEntry]
    selected_clip_index: int = 0


@dataclass(frozen=True, slots=True)
class VoiceprintCaptureReviewSession:
    """Inputs needed by the voiceprint capture review TUI."""

    project_id: str
    project_title: str | None
    project_status: str | None
    source_name: str | None
    meeting_time: str | None
    source_path: Path
    store_dir: Path
    db_path: Path
    speakers: list[VoiceprintCaptureSpeakerEntry]
    page_size: int | None = None


@dataclass(frozen=True, slots=True)
class VoiceprintCaptureDecision:
    """Result returned by the voiceprint capture review TUI."""

    saved: bool
    selected_clip_rel_paths: frozenset[str]


class VoiceprintCaptureHelpScreen(ModalScreen[None]):
    """Modal shortcut help for voiceprint capture review."""

    CSS = """
    VoiceprintCaptureHelpScreen {
        align: center middle;
    }
    #voiceprint-capture-help {
        width: 84;
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
        yield Static(shortcut_help(), id="voiceprint-capture-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class VoiceprintCaptureReviewApp(App[VoiceprintCaptureDecision]):
    """Keyboard-first TUI for approving voiceprint capture samples."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #overview {
        border: round $accent;
        height: 7;
        padding: 0 1;
    }
    #main {
        height: 1fr;
    }
    .pane {
        border: round $accent;
        height: 100%;
        padding: 0 1;
    }
    .pane.focused-pane {
        border: heavy $accent;
        background: $boost;
    }
    .pane.unfocused-pane {
        border: round #555555;
    }
    #speakers {
        width: 34%;
    }
    #samples {
        width: 66%;
    }
    #status {
        height: 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
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
        Binding("s", "save", "Capture selected"),
        Binding("?", "show_shortcuts", "Help"),
        Binding("q", "cancel", "Cancel"),
    ]

    def __init__(self, session: VoiceprintCaptureReviewSession) -> None:
        """Create the capture review app."""
        super().__init__()
        self.session = session
        self.selected_speaker_index = 0
        self.focused_column = "speakers"
        self.playback_process: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(capture_review_status(), id="status")

    def on_mount(self) -> None:
        """Render the initial capture plan."""
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when the TUI closes."""
        self._stop_playback()

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
        """Include or exclude the selected sample."""
        sample = self._selected_sample()
        if sample is None:
            return
        sample.included = not sample.included
        self._set_status(
            tr("Sample included.", "已选中样本。") if sample.included else tr("Sample excluded.", "已排除样本。")
        )
        self._refresh()

    def action_toggle_speaker(self) -> None:
        """Include or exclude all samples for the selected speaker."""
        speaker = self._speaker()
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

    def action_play_sample(self) -> None:
        """Play or stop the selected sample."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止 sample 播放。"))
            return
        sample = self._selected_sample()
        if sample is None:
            self._set_status(tr("No sample selected.", "未选择样本。"))
            return
        try:
            self._play_sample(sample)
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Playback failed: {exc}", f"播放失败：{exc}"))

    def action_save(self) -> None:
        """Return selected clips for persistence."""
        selected = self._selected_clip_rel_paths()
        if not selected:
            self._set_status(tr("No samples selected. Toggle at least one sample before capture.", "没有选中样本。采集前至少选中一个样本。"))
            return
        self.exit(VoiceprintCaptureDecision(True, frozenset(selected)))

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.push_screen(VoiceprintCaptureHelpScreen())

    def action_cancel(self) -> None:
        """Cancel capture review without writing voiceprints."""
        self.exit(VoiceprintCaptureDecision(False, frozenset()))

    def _move_focused_row(self, delta: int) -> None:
        """Move selection inside the focused column."""
        if self.focused_column == "speakers":
            self._move_speaker(delta)
            return
        self._move_sample(delta)

    def _move_column(self, delta: int) -> None:
        """Move focus between the speaker and sample columns."""
        index = COLUMNS.index(self.focused_column)
        self.focused_column = COLUMNS[_clamp(index + delta, 0, len(COLUMNS) - 1)]
        self._refresh()

    def _move_speaker(self, delta: int) -> None:
        """Move the selected speaker index."""
        total = len(self.session.speakers)
        if total == 0:
            return
        self.selected_speaker_index = (self.selected_speaker_index + delta) % total
        self._refresh()

    def _move_sample(self, delta: int) -> None:
        """Move the selected sample index."""
        speaker = self._speaker()
        if speaker is None or not speaker.clips:
            return
        target = speaker.selected_clip_index + delta
        speaker.selected_clip_index = _clamp(target, 0, len(speaker.clips) - 1)
        self._refresh()

    def _move_sample_page(self, delta: int) -> None:
        """Move the selected sample page."""
        speaker = self._speaker()
        if speaker is None or not speaker.clips:
            return
        page_size = self._sample_page_size()
        current_start = _sample_page_start(speaker.selected_clip_index, page_size)
        last_start = _last_sample_page_start(len(speaker.clips), page_size)
        speaker.selected_clip_index = _clamp(current_start + delta * page_size, 0, last_start)
        self._refresh()

    def _refresh(self) -> None:
        """Refresh all panes from current state."""
        self._refresh_focus_styles()
        self.query_one("#overview", Static).update(self._overview_pane())
        self.query_one("#speakers", Static).update(self._speaker_pane())
        self.query_one("#samples", Static).update(self._sample_pane())

    def _refresh_focus_styles(self) -> None:
        """Make the focused pane visually obvious."""
        for column in COLUMNS:
            pane = self.query_one(f"#{column}", Static)
            focused = column == self.focused_column
            pane.set_class(focused, FOCUSED_PANE_CLASS)
            pane.set_class(not focused, UNFOCUSED_PANE_CLASS)

    def _overview_pane(self) -> str:
        """Render project, store, and selection counts."""
        selected = len(self._selected_clip_rel_paths())
        total = sum(len(speaker.clips) for speaker in self.session.speakers)
        speaker = self._speaker()
        sample = self._selected_sample()
        lines = [
            f"{tr('[b]Project[/b]', '[b]项目[/b]')}  {escape(self.session.project_id)}",
            f"{tr('[b]Source[/b]', '[b]来源[/b]')}   {escape(str(self.session.source_path))}",
            f"{tr('[b]Store[/b]', '[b]库路径[/b]')}    {escape(str(self.session.db_path))}",
            tr(f"[b]Selected[/b] {selected}/{total} sample(s)", f"[b]已选[/b]     {selected}/{total} 个样本"),
            tr(f"[b]Focus[/b]    {escape(_selected_speaker_summary(speaker))}", f"[b]当前[/b]     {escape(_selected_speaker_summary(speaker))}"),
            tr(f"[b]Sample[/b]   {escape(_selected_sample_summary(sample))}", f"[b]样本[/b]     {escape(_selected_sample_summary(sample))}"),
        ]
        return "\n".join(lines)

    def _speaker_pane(self) -> str:
        """Render speakers and selected sample counts."""
        lines = [self._pane_title(tr("Project speakers", "项目 speaker"), "speakers")]
        if not self.session.speakers:
            lines.append(tr("[yellow]No capture candidates.[/]", "[yellow]没有可采集候选样本。[/]"))
            return "\n".join(lines)
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            selected = sum(1 for clip in speaker.clips if clip.included)
            person = "" if speaker.person_public_id is None else f" {speaker.person_public_id}"
            label = tr(
                f"{marker} {speaker.name}{person}  selected {selected}/{len(speaker.clips)}",
                f"{marker} {speaker.name}{person}  已选 {selected}/{len(speaker.clips)}",
            )
            lines.append(escape(label))
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render samples for the selected speaker."""
        speaker = self._speaker()
        title = tr("Samples", "样本") if speaker is None else tr(f"{speaker.name} capture samples", f"{speaker.name} 采样样本")
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append(tr("[yellow]No speaker selected.[/]", "[yellow]未选择 speaker。[/]"))
            return "\n".join(lines)
        page_start, samples = self._visible_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_clip_index else " "
            checked = "x" if sample.included else " "
            line = f"{prefix} [{checked}] #{index + 1} {_sample_line(sample)}"
            lines.append(f"[reverse]{escape(line)}[/]" if prefix == ">" else escape(line))
        if not samples:
            lines.append(tr("[yellow]No samples for this speaker.[/]", "[yellow]当前 speaker 没有样本。[/]"))
        lines.append("")
        lines.append(self._sample_page_footer(speaker, page_start))
        return "\n".join(lines)

    def _play_sample(self, sample: VoiceprintCaptureClipEntry) -> None:
        """Start playback from source media for the selected planned clip."""
        self._stop_playback()
        command = build_audio_preview_command(
            media=self.session.source_path,
            start_seconds=sample.clip_begin_time_ms / 1000,
            duration_seconds=sample.duration_seconds,
        )
        self.playback_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_status(tr(f"Playing sample {_sample_time_range(sample)}.", f"正在播放样本 {_sample_time_range(sample)}。"))

    def _stop_playback(self) -> None:
        """Stop the current playback child process if it is still running."""
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

    def _speaker(self) -> VoiceprintCaptureSpeakerEntry | None:
        """Return the selected speaker, if any."""
        if not self.session.speakers:
            return None
        return self.session.speakers[self.selected_speaker_index]

    def _selected_sample(self) -> VoiceprintCaptureClipEntry | None:
        """Return the selected sample, if any."""
        speaker = self._speaker()
        if speaker is None or not speaker.clips:
            return None
        return speaker.clips[speaker.selected_clip_index]

    def _visible_samples(
        self,
        speaker: VoiceprintCaptureSpeakerEntry,
    ) -> tuple[int, list[VoiceprintCaptureClipEntry]]:
        """Return the current sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = _sample_page_start(speaker.selected_clip_index, page_size)
        return page_start, speaker.clips[page_start : page_start + page_size]

    def _sample_page_size(self) -> int:
        """Return the number of sample rows that fit in the pane."""
        if self.session.page_size is not None:
            return max(1, self.session.page_size)
        pane_height = self.query_one("#samples", Static).size.height
        if pane_height <= SAMPLE_PANE_RESERVED_ROWS:
            return DEFAULT_SAMPLE_PAGE_SIZE
        return max(1, pane_height - SAMPLE_PANE_RESERVED_ROWS)

    def _sample_page_footer(self, speaker: VoiceprintCaptureSpeakerEntry, page_start: int) -> str:
        """Render pagination status for the sample pane."""
        page_size = self._sample_page_size()
        page_count = _sample_page_count(len(speaker.clips), page_size)
        page_number = page_start // page_size + 1
        start = page_start + 1 if speaker.clips else 0
        end = min(page_start + page_size, len(speaker.clips))
        return tr(
            f"Page {page_number}/{page_count}  Samples {start}-{end}/{len(speaker.clips)}",
            f"第 {page_number}/{page_count} 页  样本 {start}-{end}/{len(speaker.clips)}",
        )

    def _pane_title(self, title: str, column: str) -> str:
        """Render a pane title with focused-column state."""
        escaped = escape(title)
        if self.focused_column == column:
            return f"[reverse][b] FOCUS [/b][/] [b]{escaped}[/b]"
        return f"[dim]{escaped}[/dim]"

    def _set_status(self, message: str) -> None:
        """Show a short status message."""
        self.query_one("#status", Static).update(escape(message))

    def _selected_clip_rel_paths(self) -> set[str]:
        """Return selected clip relative paths."""
        return {
            clip.rel_path
            for speaker in self.session.speakers
            for clip in speaker.clips
            if clip.included
        }


def load_voiceprint_capture_review_session(
    *,
    summary: VoiceprintCaptureSummary,
    source_path: Path,
    page_size: int | None = None,
    project_title: str | None = None,
    project_status: str | None = None,
    source_name: str | None = None,
    meeting_time: str | None = None,
) -> VoiceprintCaptureReviewSession:
    """
    Build a capture review session from a dry-run capture plan.

    Args:
        summary: Planned voiceprint capture summary.
        source_path: Source media path used for playback.
        page_size: Optional samples-per-page override.
        project_title: Optional human-readable project title.
        project_status: Optional project workflow status.
        source_name: Optional original source filename.
        meeting_time: Optional meeting time from project metadata.

    Returns:
        Voiceprint capture review session.
    """
    return VoiceprintCaptureReviewSession(
        project_id=_project_id_from_summary(summary),
        project_title=project_title,
        project_status=project_status,
        source_name=source_name,
        meeting_time=meeting_time,
        source_path=source_path,
        store_dir=summary.store_dir,
        db_path=summary.db_path,
        speakers=[_speaker_entry(speaker) for speaker in summary.speakers],
        page_size=page_size,
    )


def run_voiceprint_capture_review_tui(session: VoiceprintCaptureReviewSession) -> VoiceprintCaptureDecision:
    """
    Run the Textual voiceprint capture review app.

    Args:
        session: Voiceprint capture review inputs.

    Returns:
        User decision.
    """
    result = VoiceprintCaptureReviewApp(session).run()
    return result or VoiceprintCaptureDecision(False, frozenset())


def render_voiceprint_capture_review_summary(session: VoiceprintCaptureReviewSession) -> str:
    """
    Render a non-interactive capture review summary.

    Args:
        session: Voiceprint capture review inputs.

    Returns:
        Plain terminal text.
    """
    sample_total = sum(len(speaker.clips) for speaker in session.speakers)
    lines = [
        f"Voiceprint capture review: {session.project_title or session.project_id}",
        f"Project ID: {session.project_id}",
        f"Status: {session.project_status or '-'}",
        f"Source: {session.source_name or session.source_path.name}",
        f"Meeting time: {session.meeting_time or '-'}",
        f"Store: {session.store_dir}",
        f"Speakers: {len(session.speakers)} | Candidate samples: {sample_total}",
    ]
    for speaker in session.speakers:
        lines.append(f"{speaker.name} speaker={speaker.speaker_id} samples={len(speaker.clips)}")
    return "\n".join(lines)


def _speaker_entry(speaker) -> VoiceprintCaptureSpeakerEntry:
    """Build a mutable capture speaker entry."""
    return VoiceprintCaptureSpeakerEntry(
        speaker_id=speaker.speaker_id,
        name=speaker.name,
        person_public_id=speaker.person_public_id,
        clips=[
            VoiceprintCaptureClipEntry(
                rel_path=clip.rel_path,
                source_begin_time_ms=clip.source_begin_time_ms,
                source_end_time_ms=clip.source_end_time_ms,
                clip_begin_time_ms=clip.clip_begin_time_ms,
                clip_end_time_ms=clip.clip_end_time_ms,
                text=clip.text,
            )
            for clip in speaker.clips
        ],
    )


def _project_id_from_summary(summary: VoiceprintCaptureSummary) -> str:
    """Return the planned project id when available."""
    for speaker in summary.speakers:
        if speaker.clips:
            parts = Path(speaker.clips[0].rel_path).parts
            if len(parts) >= 2:
                return parts[1]
    return "-"


def _selected_speaker_summary(speaker: VoiceprintCaptureSpeakerEntry | None) -> str:
    """Render selected speaker summary."""
    if speaker is None:
        return "-"
    selected = sum(1 for clip in speaker.clips if clip.included)
    return f"{speaker.name} speaker {speaker.speaker_id} | selected {selected}/{len(speaker.clips)}"


def _selected_sample_summary(sample: VoiceprintCaptureClipEntry | None) -> str:
    """Render selected sample summary."""
    if sample is None:
        return "-"
    return f"{_sample_time_range(sample)} | {'included' if sample.included else 'excluded'}"


def _sample_line(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one capture sample row."""
    return f"{_sample_time_range(sample)} {_trim_text(sample.text)}"


def _sample_time_range(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one sample time range."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    return f"{start}-{end}"


def _sample_page_start(selected_index: int, page_size: int) -> int:
    """Return the first sample index for the selected sample's page."""
    return selected_index // page_size * page_size


def _last_sample_page_start(sample_count: int, page_size: int) -> int:
    """Return the first sample index of the last page."""
    return max(0, (sample_count - 1) // page_size * page_size)


def _sample_page_count(sample_count: int, page_size: int) -> int:
    """Return the number of sample pages."""
    return max(1, (sample_count + page_size - 1) // page_size)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into an inclusive range."""
    return min(max(value, minimum), maximum)


def _trim_text(text: str, *, limit: int = 90) -> str:
    """Trim transcript text for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
