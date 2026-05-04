"""Textual UI for browsing the global voiceprint library."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Header, Static

from app.presentation.tui.i18n import tr
from app.utils import format_ms_timestamp
from app.voiceprint_playback import build_voiceprint_play_command
from app.voiceprint_store import (
    VoiceprintSampleRow,
    VoiceprintSpeakerRow,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_voiceprint_speakers,
)

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"
def browse_status() -> str:
    """Return localized voiceprint-library status text."""
    return tr(
        (
            "Browse: h/l or left/right choose column | j/k or up/down move | "
            "PgUp/PgDn page samples | Space play/stop | ? help | q quit"
        ),
        "浏览：h/l 或 ←/→ 切列 | j/k 或 ↑/↓ 移动 | PgUp/PgDn 翻页 | Space 播放/停止 | ? 帮助 | q 退出",
    )


def shortcut_help() -> str:
    """Return localized voiceprint-library shortcut help."""
    return tr(
        """\
[b]Voiceprint Library Shortcuts[/b]

[b]What This Shows[/b]
Left pane            People stored in the global voiceprint library
Right pane           WAV samples for the selected person
Top status           Store path, counts, embedding coverage, selected sample

[b]Navigation[/b]
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Actions[/b]
space                Play or stop selected sample
?                    Show this help
q                    Quit

[dim]Delete from CLI: meeting-asr voiceprint delete-sample SPEAKER --sample N[/]
[dim]Delete person:  meeting-asr voiceprint delete-speaker SPEAKER[/]
""",
        """\
[b]声纹库快捷键[/b]

[b]这里展示什么[/b]
左侧                 全局声纹库里的人员
右侧                 当前人员的 WAV 样本
顶部状态             库路径、数量、embedding 覆盖率、当前样本

[b]导航[/b]
h/l 或 ←/→           切换当前列
j/k 或 ↑/↓           在当前列内移动
PageUp/PageDown      上一页/下一页 sample
[ / ]                上一页/下一页 sample

[b]操作[/b]
space                播放或停止当前 sample
?                    显示帮助
q                    退出

[dim]从 CLI 删除样本：meeting-asr voiceprint delete-sample SPEAKER --sample N[/]
[dim]删除人员：      meeting-asr voiceprint delete-speaker SPEAKER[/]
""",
    )


@dataclass(slots=True)
class VoiceprintSpeakerEntry:
    """Mutable TUI row for one stored voiceprint speaker."""

    speaker_id: int
    public_id: str
    name: str
    sample_count: int
    project_count: int
    embedded_sample_count: int
    embedding_model_count: int
    updated_at: str | None
    samples: list[VoiceprintSampleRow]
    selected_sample_index: int = 0


@dataclass(frozen=True, slots=True)
class VoiceprintLibrarySession:
    """Inputs needed by the voiceprint library TUI."""

    db_path: Path
    speakers: list[VoiceprintSpeakerEntry]
    page_size: int | None = None


class VoiceprintHelpScreen(ModalScreen[None]):
    """Modal shortcut help for the voiceprint library TUI."""

    CSS = """
    VoiceprintHelpScreen {
        align: center middle;
    }
    #voiceprint-help {
        width: 78;
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
        yield Static(shortcut_help(), id="voiceprint-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class VoiceprintLibraryApp(App[None]):
    """Keyboard-first TUI for browsing stored voiceprint samples."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #overview {
        border: round $accent;
        height: 6;
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
        Binding("?", "show_shortcuts", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session: VoiceprintLibrarySession) -> None:
        """
        Create the voiceprint library app.

        Args:
            session: Voiceprint library data.
        """
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
        yield Static(browse_status(), id="status")

    def on_mount(self) -> None:
        """Render the initial library state."""
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

    def action_play_sample(self) -> None:
        """Play or stop the selected sample."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止 sample 播放。"))
            return
        sample = self._selected_sample()
        if sample is None:
            self._set_status(tr("No sample for this speaker.", "当前 speaker 没有样本。"))
            return
        try:
            self._play_sample(sample)
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Playback failed: {exc}", f"播放失败：{exc}"))

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.push_screen(VoiceprintHelpScreen())

    def action_quit(self) -> None:
        """Exit the browser."""
        self.exit(None)

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
        if speaker is None or not speaker.samples:
            return
        target = speaker.selected_sample_index + delta
        speaker.selected_sample_index = _clamp(target, 0, len(speaker.samples) - 1)
        self._refresh()

    def _move_sample_page(self, delta: int) -> None:
        """Move the selected sample page."""
        speaker = self._speaker()
        if speaker is None or not speaker.samples:
            return
        page_size = self._sample_page_size()
        current_start = _sample_page_start(speaker.selected_sample_index, page_size)
        last_start = _last_sample_page_start(len(speaker.samples), page_size)
        speaker.selected_sample_index = _clamp(current_start + delta * page_size, 0, last_start)
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
        """Render library counts and selected sample state."""
        speaker = self._speaker()
        sample = self._selected_sample()
        sample_count = sum(item.sample_count for item in self.session.speakers)
        embedded = sum(item.embedded_sample_count for item in self.session.speakers)
        lines = [
            f"{tr('[b]Store[/b]', '[b]库路径[/b]')}    {escape(str(self.session.db_path))}",
            tr(
                f"[b]Library[/b]  speakers {len(self.session.speakers)} | samples {sample_count} | embedded {embedded}/{sample_count}",
                f"[b]声纹库[/b]   speaker {len(self.session.speakers)} | 样本 {sample_count} | embedding {embedded}/{sample_count}",
            ),
            tr(f"[b]Focus[/b]    {escape(_selected_speaker_summary(speaker))}", f"[b]当前[/b]     {escape(_selected_speaker_summary(speaker))}"),
            tr(f"[b]Sample[/b]   {escape(_selected_sample_summary(sample))}", f"[b]样本[/b]     {escape(_selected_sample_summary(sample))}"),
        ]
        return "\n".join(lines)

    def _speaker_pane(self) -> str:
        """Render the speaker list."""
        lines = [self._pane_title(tr("Voiceprint speakers", "声纹人员"), "speakers")]
        if not self.session.speakers:
            lines.append(tr("[yellow]No voiceprints recorded.[/]", "[yellow]尚未录入声纹。[/]"))
            return "\n".join(lines)
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            embedded = f"{speaker.embedded_sample_count}/{speaker.sample_count}"
            label = tr(
                f"{marker} {speaker.name}  samples {speaker.sample_count}  embedded {embedded}",
                f"{marker} {speaker.name}  样本 {speaker.sample_count}  embedding {embedded}",
            )
            lines.append(escape(label))
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render samples for the selected speaker."""
        speaker = self._speaker()
        title = tr("Samples", "样本") if speaker is None else tr(f"{speaker.name} samples", f"{speaker.name} 样本")
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append(tr("[yellow]No speaker selected.[/]", "[yellow]未选择 speaker。[/]"))
            return "\n".join(lines)
        page_start, samples = self._visible_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            line = f"{prefix} #{index + 1} {_sample_line(sample)}"
            lines.append(f"[reverse]{escape(line)}[/]" if prefix == ">" else escape(line))
        if not samples:
            lines.append(tr("[yellow]No samples for this speaker.[/]", "[yellow]当前 speaker 没有样本。[/]"))
        lines.append("")
        lines.append(self._sample_page_footer(speaker, page_start))
        return "\n".join(lines)

    def _play_sample(self, sample: VoiceprintSampleRow) -> None:
        """Start playback for the selected voiceprint clip."""
        self._stop_playback()
        command = build_voiceprint_play_command(sample.clip_path)
        self.playback_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_status(tr(f"Playing {sample.speaker_name} sample {sample.public_id}.", f"正在播放 {sample.speaker_name} 的样本 {sample.public_id}。"))

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

    def _speaker(self) -> VoiceprintSpeakerEntry | None:
        """Return the selected speaker, if any."""
        if not self.session.speakers:
            return None
        return self.session.speakers[self.selected_speaker_index]

    def _selected_sample(self) -> VoiceprintSampleRow | None:
        """Return the selected sample, if any."""
        speaker = self._speaker()
        if speaker is None or not speaker.samples:
            return None
        return speaker.samples[speaker.selected_sample_index]

    def _visible_samples(self, speaker: VoiceprintSpeakerEntry) -> tuple[int, list[VoiceprintSampleRow]]:
        """Return the current sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = _sample_page_start(speaker.selected_sample_index, page_size)
        return page_start, speaker.samples[page_start : page_start + page_size]

    def _sample_page_size(self) -> int:
        """Return the number of sample rows that fit in the pane."""
        if self.session.page_size is not None:
            return max(1, self.session.page_size)
        pane_height = self.query_one("#samples", Static).size.height
        if pane_height <= SAMPLE_PANE_RESERVED_ROWS:
            return DEFAULT_SAMPLE_PAGE_SIZE
        return max(1, pane_height - SAMPLE_PANE_RESERVED_ROWS)

    def _sample_page_footer(self, speaker: VoiceprintSpeakerEntry, page_start: int) -> str:
        """Render pagination status for the sample pane."""
        page_size = self._sample_page_size()
        page_count = _sample_page_count(len(speaker.samples), page_size)
        page_number = page_start // page_size + 1
        start = page_start + 1 if speaker.samples else 0
        end = min(page_start + page_size, len(speaker.samples))
        return tr(
            f"Page {page_number}/{page_count}  Samples {start}-{end}/{len(speaker.samples)}",
            f"第 {page_number}/{page_count} 页  样本 {start}-{end}/{len(speaker.samples)}",
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


def load_voiceprint_library_session(
    *,
    store_dir: Path | None = None,
    page_size: int | None = None,
) -> VoiceprintLibrarySession:
    """
    Load the global voiceprint library for browsing.

    Args:
        store_dir: Optional voiceprint store directory.
        page_size: Optional samples-per-page override.

    Returns:
        Voiceprint library session.
    """
    db_path = get_voiceprint_db_path(store_dir)
    speakers = list_voiceprint_speakers(db_path)
    samples = list_all_voiceprint_samples(db_path)
    samples_by_speaker: dict[int, list[VoiceprintSampleRow]] = defaultdict(list)
    for sample in samples:
        samples_by_speaker[sample.speaker_id].append(sample)
    return VoiceprintLibrarySession(
        db_path=db_path,
        speakers=[_speaker_entry(row, samples_by_speaker.get(row.speaker_id, [])) for row in speakers],
        page_size=page_size,
    )


def run_voiceprint_library_tui(session: VoiceprintLibrarySession) -> None:
    """
    Run the Textual voiceprint library browser.

    Args:
        session: Voiceprint library inputs.
    """
    VoiceprintLibraryApp(session).run()


def render_voiceprint_library_summary(session: VoiceprintLibrarySession) -> str:
    """
    Render a non-interactive voiceprint library summary.

    Args:
        session: Voiceprint library inputs.

    Returns:
        Plain terminal text.
    """
    sample_total = sum(speaker.sample_count for speaker in session.speakers)
    embedded_total = sum(speaker.embedded_sample_count for speaker in session.speakers)
    lines = [
        f"Voiceprint library: {session.db_path}",
        f"Speakers: {len(session.speakers)} | Samples: {sample_total} | Embedded: {embedded_total}/{sample_total}",
    ]
    if not session.speakers:
        lines.append("No voiceprints recorded.")
        return "\n".join(lines)
    for speaker in session.speakers:
        lines.append(_summary_line(speaker))
    return "\n".join(lines)


def _speaker_entry(row: VoiceprintSpeakerRow, samples: list[VoiceprintSampleRow]) -> VoiceprintSpeakerEntry:
    """Build a mutable speaker entry from store rows."""
    return VoiceprintSpeakerEntry(
        speaker_id=row.speaker_id,
        public_id=row.public_id,
        name=row.name,
        sample_count=row.sample_count,
        project_count=row.project_count,
        embedded_sample_count=row.embedded_sample_count,
        embedding_model_count=row.embedding_model_count,
        updated_at=row.updated_at,
        samples=samples,
    )


def _summary_line(speaker: VoiceprintSpeakerEntry) -> str:
    """Render one plain summary row."""
    return (
        f"{speaker.name} id={speaker.public_id} samples={speaker.sample_count} "
        f"projects={speaker.project_count} embedded={speaker.embedded_sample_count}/{speaker.sample_count}"
    )


def _selected_speaker_summary(speaker: VoiceprintSpeakerEntry | None) -> str:
    """Render selected speaker summary."""
    if speaker is None:
        return "-"
    return (
        tr(
            f"{speaker.name} id={speaker.public_id} | samples {speaker.sample_count} | "
            f"projects {speaker.project_count} | models {speaker.embedding_model_count}",
            f"{speaker.name} id={speaker.public_id} | 样本 {speaker.sample_count} | "
            f"项目 {speaker.project_count} | 模型 {speaker.embedding_model_count}",
        )
    )


def _selected_sample_summary(sample: VoiceprintSampleRow | None) -> str:
    """Render selected sample summary."""
    if sample is None:
        return "-"
    return tr(f"sample_id {sample.public_id} | clip {sample.clip_path}", f"样本ID {sample.public_id} | 文件 {sample.clip_path}")


def _sample_line(sample: VoiceprintSampleRow) -> str:
    """Render one sample row."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    text = _trim_text(sample.transcript_text)
    return f"{sample.project_id} speaker {sample.project_speaker_id} {start}-{end} {text}"


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
