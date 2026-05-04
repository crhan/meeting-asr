"""Unified Textual UI for project voiceprint capture and global library review."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static

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
from app.project_manager import load_manifest, project_paths, resolve_project_source_path
from app.speaker_review import build_audio_preview_command
from app.utils import format_ms_timestamp
from app.voiceprint_playback import build_voiceprint_play_command
from app.voiceprint_store import VoiceprintSampleRow
from app.voiceprints import VoiceprintCaptureSummary, plan_voiceprint_capture

Mode = Literal["project", "library"]
Column = Literal["speakers", "samples"]

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS: tuple[Column, Column] = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"
PROJECT_MODE = "project"
LIBRARY_MODE = "library"

STATUS_TEXT = (
    "Voiceprint: Tab switch Project/Global | p project | g global | h/l columns | "
    "j/k rows | Space play/stop | x include/exclude | s capture selected | ? help | q back/quit"
)
HELP_TEXT = """\
[b]Voiceprint Review Shortcuts[/b]

[b]Views[/b]
Project candidates   Clips planned from the current project before they enter the global library
Global library       Stored WAV samples grouped by stable person id

[b]Navigation[/b]
tab                  Switch Project candidates / Global library
p / g                Jump to Project / Global library
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Project Actions[/b]
space                Play or stop selected source-media sample
x                    Include/exclude selected planned sample
a                    Include/exclude all planned samples for the selected speaker
s                    Capture checked project samples into the global voiceprint store

[b]Library Actions[/b]
space                Play or stop selected stored WAV sample

[b]Exit[/b]
q                    Quit without writing new samples
?                    Show or close this help
"""


@dataclass(frozen=True, slots=True)
class VoiceprintReviewSession:
    """Inputs used by the unified voiceprint review TUI."""

    capture: VoiceprintCaptureReviewSession | None
    library: VoiceprintLibrarySession
    initial_mode: Mode = PROJECT_MODE
    return_hint: str = "quit"


@dataclass(frozen=True, slots=True)
class VoiceprintReviewDecision:
    """Result returned by the unified voiceprint review TUI."""

    saved: bool
    selected_clip_rel_paths: frozenset[str]


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
        yield Static(HELP_TEXT, id="voiceprint-review-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class _VoiceprintReviewBase:
    """Shared controller for the voiceprint review app and embeddable screen."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #overview {
        border: round $accent;
        height: 9;
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
        Binding("tab", "switch_mode", "Switch view", priority=True),
        Binding("p", "project_mode", "Project"),
        Binding("g", "library_mode", "Library"),
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
        self.playback_process: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(STATUS_TEXT, id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Render the initial view."""
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when the TUI closes."""
        self._stop_playback()

    def action_switch_mode(self) -> None:
        """Switch between project candidates and the global library."""
        if self.session.capture is None:
            self._set_status("Project candidates unavailable; showing the global library.")
            return
        self._stop_playback()
        self.mode = LIBRARY_MODE if self.mode == PROJECT_MODE else PROJECT_MODE
        self._set_status(f"Switched to {_mode_label(self.mode)}.")
        self._refresh()

    def action_project_mode(self) -> None:
        """Jump to the project capture view."""
        if self.session.capture is None:
            self._set_status("No project was provided, so project candidates are unavailable.")
            return
        self._switch_to(PROJECT_MODE)

    def action_library_mode(self) -> None:
        """Jump to the global library view."""
        self._switch_to(LIBRARY_MODE)

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
        if self.mode != PROJECT_MODE:
            self._set_status("Sample selection only applies to project candidates.")
            return
        sample = self._project_selected_sample()
        if sample is None:
            return
        sample.included = not sample.included
        self._set_status("Sample included." if sample.included else "Sample excluded.")
        self._refresh()

    def action_toggle_speaker(self) -> None:
        """Include or exclude all samples for the selected project speaker."""
        if self.mode != PROJECT_MODE:
            self._set_status("Speaker selection only applies to project candidates.")
            return
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        include = not all(clip.included for clip in speaker.clips)
        for clip in speaker.clips:
            clip.included = include
        self._set_status(f"{'Included' if include else 'Excluded'} all samples for {speaker.name}.")
        self._refresh()

    def action_play_sample(self) -> None:
        """Play or stop the selected sample for the active view."""
        if self._is_playing():
            self._stop_playback()
            self._set_status("Stopped sample playback.")
            return
        try:
            self._play_active_sample()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Playback failed: {exc}")

    def action_save(self) -> None:
        """Return selected project clips for persistence."""
        if self.mode != PROJECT_MODE:
            self._set_status("Switch to Project candidates before saving new samples.")
            return
        if self.session.capture is None:
            self._set_status("No project candidates to capture in this session.")
            return
        selected = self._selected_clip_rel_paths()
        if not selected:
            self._set_status("No samples selected. Toggle at least one sample before capture.")
            return
        self._finish(VoiceprintReviewDecision(True, frozenset(selected)))

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.app.push_screen(VoiceprintReviewHelpScreen())

    def action_quit(self) -> None:
        """Exit without writing project samples."""
        self._finish(VoiceprintReviewDecision(False, frozenset()))

    def _finish(self, decision: VoiceprintReviewDecision) -> None:
        """Return the review decision to the active Textual host."""
        raise NotImplementedError

    def _switch_to(self, mode: Mode) -> None:
        """Switch to a specific mode and refresh the screen."""
        if self.mode == mode:
            return
        self._stop_playback()
        self.mode = mode
        self._set_status(f"Switched to {_mode_label(mode)}.")
        self._refresh()

    def _move_focused_row(self, delta: int) -> None:
        """Move selection inside the focused column."""
        if self._focused_column() == "speakers":
            self._move_speaker(delta)
            return
        self._move_sample(delta)

    def _move_column(self, delta: int) -> None:
        """Move focus between speaker and sample panes."""
        index = COLUMNS.index(self._focused_column())
        self._set_focused_column(COLUMNS[_clamp(index + delta, 0, len(COLUMNS) - 1)])
        self._refresh()

    def _move_speaker(self, delta: int) -> None:
        """Move the selected speaker in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_speaker(delta)
            return
        self._move_library_speaker(delta)

    def _move_sample(self, delta: int) -> None:
        """Move the selected sample in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_sample(delta)
            return
        self._move_library_sample(delta)

    def _move_sample_page(self, delta: int) -> None:
        """Move the selected sample page in the active view."""
        if self.mode == PROJECT_MODE:
            self._move_project_sample_page(delta)
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
        return self._library_overview_pane()

    def _speaker_pane(self) -> str:
        """Render speakers for the active view."""
        if self.mode == PROJECT_MODE:
            return self._project_speaker_pane()
        return self._library_speaker_pane()

    def _sample_pane(self) -> str:
        """Render samples for the active view."""
        if self.mode == PROJECT_MODE:
            return self._project_sample_pane()
        return self._library_sample_pane()

    def _project_overview_pane(self) -> str:
        """Render project capture counts and selection state."""
        capture = self.session.capture
        if capture is None:
            return "[b]Mode[/b]     Global library only\n[yellow]No project candidates loaded.[/]"
        selected = len(self._selected_clip_rel_paths())
        total = sum(len(speaker.clips) for speaker in capture.speakers)
        speaker = self._project_speaker()
        sample = self._project_selected_sample()
        lines = [
            self._mode_line(),
            f"[b]Project[/b]  {escape(capture.project_id)}",
            f"[b]Source[/b]   {escape(str(capture.source_path))}",
            f"[b]Store[/b]    {escape(str(capture.db_path))}",
            f"[b]Selected[/b] {selected}/{total} candidate sample(s)",
            f"[b]Focus[/b]    {escape(_project_speaker_summary(speaker))}",
            f"[b]Sample[/b]   {escape(_project_sample_summary(sample))}",
        ]
        return "\n".join(lines)

    def _library_overview_pane(self) -> str:
        """Render global library counts and selection state."""
        speaker = self._library_speaker()
        sample = self._library_selected_sample()
        sample_count = sum(item.sample_count for item in self.session.library.speakers)
        embedded = sum(item.embedded_sample_count for item in self.session.library.speakers)
        lines = [
            self._mode_line(),
            f"[b]Store[/b]    {escape(str(self.session.library.db_path))}",
            f"[b]Library[/b]  speakers {len(self.session.library.speakers)} | samples {sample_count} | embedded {embedded}/{sample_count}",
            f"[b]Focus[/b]    {escape(_library_speaker_summary(speaker))}",
            f"[b]Sample[/b]   {escape(_library_sample_summary(sample))}",
        ]
        return "\n".join(lines)

    def _project_speaker_pane(self) -> str:
        """Render project speakers and selected sample counts."""
        lines = [self._pane_title("Project candidates", "speakers")]
        capture = self.session.capture
        if capture is None or not capture.speakers:
            lines.append("[yellow]No capture candidates.[/]")
            return "\n".join(lines)
        for index, speaker in enumerate(capture.speakers):
            marker = ">" if index == self.project_selected_speaker_index else " "
            selected = sum(1 for clip in speaker.clips if clip.included)
            person = "" if speaker.person_public_id is None else f" {speaker.person_public_id}"
            label = f"{marker} {speaker.name}{person}  selected {selected}/{len(speaker.clips)}"
            lines.append(escape(label))
        return "\n".join(lines)

    def _library_speaker_pane(self) -> str:
        """Render global voiceprint people."""
        lines = [self._pane_title("Global voiceprint people", "speakers")]
        if not self.session.library.speakers:
            lines.append("[yellow]No voiceprints recorded.[/]")
            return "\n".join(lines)
        for index, speaker in enumerate(self.session.library.speakers):
            marker = ">" if index == self.library_selected_speaker_index else " "
            embedded = f"{speaker.embedded_sample_count}/{speaker.sample_count}"
            label = f"{marker} {speaker.name} {speaker.public_id}  samples {speaker.sample_count}  embedded {embedded}"
            lines.append(escape(label))
        return "\n".join(lines)

    def _project_sample_pane(self) -> str:
        """Render project capture samples for the selected speaker."""
        speaker = self._project_speaker()
        title = "Project samples" if speaker is None else f"{speaker.name} project samples"
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append("[yellow]No speaker selected.[/]")
            return "\n".join(lines)
        page_start, samples = self._visible_project_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_clip_index else " "
            checked = "x" if sample.included else " "
            line = f"{prefix} [{checked}] #{index + 1} {_project_sample_line(sample)}"
            lines.append(f"[reverse]{escape(line)}[/]" if prefix == ">" else escape(line))
        if not samples:
            lines.append("[yellow]No samples for this speaker.[/]")
        lines.extend(["", self._project_sample_page_footer(speaker, page_start)])
        return "\n".join(lines)

    def _library_sample_pane(self) -> str:
        """Render stored WAV samples for the selected person."""
        speaker = self._library_speaker()
        title = "Library samples" if speaker is None else f"{speaker.name} stored samples"
        lines = [self._pane_title(title, "samples")]
        if speaker is None:
            lines.append("[yellow]No speaker selected.[/]")
            return "\n".join(lines)
        page_start, samples = self._visible_library_samples(speaker)
        for offset, sample in enumerate(samples):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            line = f"{prefix} #{index + 1} {_library_sample_line(sample)}"
            lines.append(f"[reverse]{escape(line)}[/]" if prefix == ">" else escape(line))
        if not samples:
            lines.append("[yellow]No samples for this person.[/]")
        lines.extend(["", self._library_sample_page_footer(speaker, page_start)])
        return "\n".join(lines)

    def _play_active_sample(self) -> None:
        """Start playback for the selected sample in the active view."""
        if self.mode == PROJECT_MODE:
            sample = self._project_selected_sample()
            if sample is None:
                self._set_status("No project sample selected.")
                return
            self._play_project_sample(sample)
            return
        sample = self._library_selected_sample()
        if sample is None:
            self._set_status("No library sample selected.")
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
        self.playback_process = _start_player(command)
        self._set_status(f"Playing project sample {_project_sample_time_range(sample)}.")

    def _play_library_sample(self, sample: VoiceprintSampleRow) -> None:
        """Start WAV playback for one stored library sample."""
        self._stop_playback()
        self.playback_process = _start_player(build_voiceprint_play_command(sample.clip_path))
        self._set_status(f"Playing {sample.speaker_name} sample {sample.public_id}.")

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

    def _move_project_sample(self, delta: int) -> None:
        """Move the selected project sample index."""
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        target = speaker.selected_clip_index + delta
        speaker.selected_clip_index = _clamp(target, 0, len(speaker.clips) - 1)
        self._refresh()

    def _move_library_sample(self, delta: int) -> None:
        """Move the selected library sample index."""
        speaker = self._library_speaker()
        if speaker is None or not speaker.samples:
            return
        target = speaker.selected_sample_index + delta
        speaker.selected_sample_index = _clamp(target, 0, len(speaker.samples) - 1)
        self._refresh()

    def _move_project_sample_page(self, delta: int) -> None:
        """Move the selected project sample page."""
        speaker = self._project_speaker()
        if speaker is None or not speaker.clips:
            return
        page_size = self._sample_page_size()
        current_start = _sample_page_start(speaker.selected_clip_index, page_size)
        last_start = _last_sample_page_start(len(speaker.clips), page_size)
        speaker.selected_clip_index = _clamp(current_start + delta * page_size, 0, last_start)
        self._refresh()

    def _move_library_sample_page(self, delta: int) -> None:
        """Move the selected library sample page."""
        speaker = self._library_speaker()
        if speaker is None or not speaker.samples:
            return
        page_size = self._sample_page_size()
        current_start = _sample_page_start(speaker.selected_sample_index, page_size)
        last_start = _last_sample_page_start(len(speaker.samples), page_size)
        speaker.selected_sample_index = _clamp(current_start + delta * page_size, 0, last_start)
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

    def _visible_project_samples(
        self,
        speaker: VoiceprintCaptureSpeakerEntry,
    ) -> tuple[int, list[VoiceprintCaptureClipEntry]]:
        """Return the current project sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = _sample_page_start(speaker.selected_clip_index, page_size)
        return page_start, speaker.clips[page_start : page_start + page_size]

    def _visible_library_samples(self, speaker: VoiceprintSpeakerEntry) -> tuple[int, list[VoiceprintSampleRow]]:
        """Return the current library sample page start and rows."""
        page_size = self._sample_page_size()
        page_start = _sample_page_start(speaker.selected_sample_index, page_size)
        return page_start, speaker.samples[page_start : page_start + page_size]

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
        return self.session.library.page_size

    def _project_sample_page_footer(self, speaker: VoiceprintCaptureSpeakerEntry, page_start: int) -> str:
        """Render project sample pagination status."""
        return _page_footer("Samples", len(speaker.clips), page_start, self._sample_page_size())

    def _library_sample_page_footer(self, speaker: VoiceprintSpeakerEntry, page_start: int) -> str:
        """Render library sample pagination status."""
        return _page_footer("Samples", len(speaker.samples), page_start, self._sample_page_size())

    def _pane_title(self, title: str, column: Column) -> str:
        """Render a pane title with focused-column state."""
        escaped = escape(title)
        if self._focused_column() == column:
            return f"[reverse][b] FOCUS [/b][/] [b]{escaped}[/b]"
        return f"[dim]{escaped}[/dim]"

    def _mode_line(self) -> str:
        """Render active mode and switch affordance."""
        next_view = _next_view_label(self.mode, self.session.capture is not None)
        next_text = "no alternate project view" if next_view is None else f"Tab -> {next_view}"
        return (
            "[reverse][b] VOICEPRINT REVIEW [/b][/]  "
            f"view=[bold cyan]{_mode_label(self.mode)}[/] | {next_text} | q: {escape(self.session.return_hint)}"
        )

    def _focused_column(self) -> Column:
        """Return the focused column for the active view."""
        if self.mode == PROJECT_MODE:
            return self.project_focused_column
        return self.library_focused_column

    def _set_focused_column(self, column: Column) -> None:
        """Set the focused column for the active view."""
        if self.mode == PROJECT_MODE:
            self.project_focused_column = column
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

    def _finish(self, decision: VoiceprintReviewDecision) -> None:
        """Dismiss the screen with a review decision."""
        self.dismiss(decision)


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
        planned = plan_voiceprint_capture(
            project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
        )
        capture_session = load_voiceprint_capture_review_session(
            summary=planned,
            source_path=_voiceprint_capture_source_path(project_dir),
            page_size=page_size,
        )
    library_session = load_voiceprint_library_session(store_dir=store_dir, page_size=page_size)
    return VoiceprintReviewSession(capture=capture_session, library=library_session, return_hint=return_hint), planned


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


def _voiceprint_capture_source_path(project_dir: Path) -> Path:
    """
    Resolve the project source media path used for sample playback.

    Args:
        project_dir: Project root.

    Returns:
        Resolved source media path.
    """
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    return resolve_project_source_path(paths.root, manifest)


def render_voiceprint_review_summary(session: VoiceprintReviewSession) -> str:
    """
    Render a non-interactive summary for the unified voiceprint review session.

    Args:
        session: Unified voiceprint review inputs.

    Returns:
        Plain terminal text.
    """
    lines = ["Voiceprint review"]
    if session.capture is None:
        lines.append("Project candidates: unavailable")
    else:
        sample_total = sum(len(speaker.clips) for speaker in session.capture.speakers)
        lines.extend(
            [
                f"Project candidates: {session.capture.project_id}",
                f"Candidate speakers: {len(session.capture.speakers)} | samples: {sample_total}",
            ]
        )
        for speaker in session.capture.speakers:
            lines.append(f"  {speaker.name} speaker={speaker.speaker_id} samples={len(speaker.clips)}")
    sample_total = sum(speaker.sample_count for speaker in session.library.speakers)
    embedded_total = sum(speaker.embedded_sample_count for speaker in session.library.speakers)
    lines.extend(
        [
            f"Global library: {session.library.db_path}",
            f"Library people: {len(session.library.speakers)} | samples: {sample_total} | embedded: {embedded_total}/{sample_total}",
        ]
    )
    for speaker in session.library.speakers:
        lines.append(f"  {speaker.name} id={speaker.public_id} samples={speaker.sample_count}")
    return "\n".join(lines)


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


def _mode_label(mode: Mode) -> str:
    """Return the human-readable mode label."""
    return "Project candidates" if mode == PROJECT_MODE else "Global library"


def _next_view_label(mode: Mode, project_available: bool) -> str | None:
    """Return the view reached by pressing Tab, if there is one."""
    if mode == LIBRARY_MODE and not project_available:
        return None
    return "Global library" if mode == PROJECT_MODE else "Project candidates"


def _project_speaker_summary(speaker: VoiceprintCaptureSpeakerEntry | None) -> str:
    """Render selected project speaker summary."""
    if speaker is None:
        return "-"
    selected = sum(1 for clip in speaker.clips if clip.included)
    person = "" if speaker.person_public_id is None else f" | person {speaker.person_public_id}"
    return f"{speaker.name} speaker {speaker.speaker_id}{person} | selected {selected}/{len(speaker.clips)}"


def _library_speaker_summary(speaker: VoiceprintSpeakerEntry | None) -> str:
    """Render selected library speaker summary."""
    if speaker is None:
        return "-"
    return (
        f"{speaker.name} id={speaker.public_id} | samples {speaker.sample_count} | "
        f"projects {speaker.project_count} | models {speaker.embedding_model_count}"
    )


def _project_sample_summary(sample: VoiceprintCaptureClipEntry | None) -> str:
    """Render selected project sample summary."""
    if sample is None:
        return "-"
    return f"{_project_sample_time_range(sample)} | {'included' if sample.included else 'excluded'}"


def _library_sample_summary(sample: VoiceprintSampleRow | None) -> str:
    """Render selected library sample summary."""
    if sample is None:
        return "-"
    return f"sample_id {sample.public_id} | clip {sample.clip_path}"


def _project_sample_line(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one project capture sample row."""
    return f"{_project_sample_time_range(sample)} {_trim_text(sample.text)}"


def _library_sample_line(sample: VoiceprintSampleRow) -> str:
    """Render one stored library sample row."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    return f"{sample.project_id} speaker {sample.project_speaker_id} {start}-{end} {_trim_text(sample.transcript_text)}"


def _project_sample_time_range(sample: VoiceprintCaptureClipEntry) -> str:
    """Render one project sample time range."""
    start = format_ms_timestamp(sample.source_begin_time_ms)
    end = format_ms_timestamp(sample.source_end_time_ms)
    return f"{start}-{end}"


def _page_footer(label: str, item_count: int, page_start: int, page_size: int) -> str:
    """Render pagination status for the active sample pane."""
    page_count = _sample_page_count(item_count, page_size)
    page_number = page_start // page_size + 1
    start = page_start + 1 if item_count else 0
    end = min(page_start + page_size, item_count)
    return f"Page {page_number}/{page_count}  {label} {start}-{end}/{item_count}"


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
