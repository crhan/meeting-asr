"""Inline transcript correction popup for project speaker review."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

from app.models import SentenceSegment
from app.presentation.tui.diff_render import styled_before_after


@dataclass(frozen=True, slots=True)
class SentenceCorrectionEdit:
    """One sentence text edit made inside the speaker review TUI."""

    sentence_id: int | None
    speaker_id: int | None
    begin_time_ms: int
    end_time_ms: int
    original_text: str
    corrected_text: str


class CorrectionTextArea(TextArea):
    """Large transcript editor with non-destructive cursor shortcuts."""

    BINDINGS = [
        Binding("enter", "submit_correction", "Apply", show=False, priority=True),
        Binding("escape", "cancel_correction", "Cancel", show=False, priority=True),
        Binding("ctrl+f", "cursor_right", "Forward", show=False, priority=True),
        Binding("ctrl+b", "cursor_left", "Backward", show=False, priority=True),
    ]

    def action_submit_correction(self) -> None:
        """Submit the current editor text to the parent modal."""
        screen = self.app.screen
        if hasattr(screen, "submit_correction"):
            screen.submit_correction(self.text)

    def action_cancel_correction(self) -> None:
        """Cancel the correction popup."""
        self.app.screen.dismiss(None)


class SentenceCorrectionScreen(ModalScreen[SentenceCorrectionEdit | None]):
    """Modal popup for editing the selected transcript sentence."""

    CSS = """
    SentenceCorrectionScreen {
        align: center middle;
    }
    #correction-box {
        width: 92%;
        height: 72%;
        max-height: 86%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #correction-title {
        text-style: bold;
    }
    #correction-original {
        color: $text-muted;
        height: 7;
        margin: 1 0;
    }
    #correction-input {
        height: 1fr;
        border: tall $primary;
    }
    #correction-status {
        height: 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_correction", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        speaker_label: str,
        speaker_name: str,
        segment: SentenceSegment,
    ) -> None:
        """
        Create a sentence correction modal.

        Args:
            speaker_label: Anonymous speaker label.
            speaker_name: Current speaker display name.
            segment: Selected transcript sentence.
        """
        super().__init__()
        self.speaker_label = speaker_label
        self.speaker_name = speaker_name
        self.segment = segment

    def compose(self) -> ComposeResult:
        """Build the correction popup."""
        original = self.segment.text.strip()
        title = f"Edit Transcript Text | {self.speaker_label} / {self.speaker_name}"
        with Vertical(id="correction-box"):
            yield Static(title, id="correction-title")
            with ScrollableContainer(id="correction-original"):
                yield Static(f"[b]Original:[/]\n{original}")
            yield CorrectionTextArea(original, id="correction-input", soft_wrap=True, show_line_numbers=False)
            yield Static("Enter applies this edit. Esc cancels. Ctrl-F/Ctrl-B move cursor.", id="correction-status")

    def on_mount(self) -> None:
        """Focus the correction editor."""
        field = self.query_one("#correction-input", TextArea)
        field.focus()
        field.cursor_location = (0, len(self.segment.text.strip()))

    def submit_correction(self, value: str) -> None:
        """Return the edited sentence text to the parent TUI."""
        corrected = value.strip()
        original = self.segment.text.strip()
        if not corrected:
            self.query_one("#correction-status", Static).update("Corrected text cannot be empty.")
            return
        if corrected == original:
            self.dismiss(None)
            return
        self.dismiss(
            SentenceCorrectionEdit(
                sentence_id=self.segment.sentence_id,
                speaker_id=self.segment.speaker_id,
                begin_time_ms=self.segment.begin_time_ms,
                end_time_ms=self.segment.end_time_ms,
                original_text=original,
                corrected_text=corrected,
            )
        )

    def action_cancel_correction(self) -> None:
        """Cancel the correction popup."""
        self.dismiss(None)


class CorrectionQueuedScreen(ModalScreen[None]):
    """Modal feedback after a transcript correction is staged."""

    CSS = """
    CorrectionQueuedScreen {
        align: center middle;
    }
    #queued-box {
        width: 92%;
        height: auto;
        max-height: 82%;
        border: thick $success;
        padding: 1 2;
        background: $surface;
    }
    #queued-title {
        text-style: bold;
        color: $success;
    }
    #queued-body {
        height: auto;
        max-height: 1fr;
        margin: 1 0;
    }
    #queued-actions {
        height: 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("enter", "close_feedback", "Continue"),
        Binding("escape", "close_feedback", "Continue", show=False),
        Binding("q", "close_feedback", "Continue"),
        Binding("s", "save_and_run", "Save and run"),
    ]

    def __init__(self, edit: SentenceCorrectionEdit, *, count: int = 1) -> None:
        """
        Create staged-correction feedback.

        Args:
            edit: Staged sentence correction.
            count: Total staged correction count.
        """
        super().__init__()
        self.edit = edit
        self.count = count

    def compose(self) -> ComposeResult:
        """Build staged correction feedback."""
        with Vertical(id="queued-box"):
            yield Static("Transcript correction staged", id="queued-title")
            with ScrollableContainer(id="queued-body"):
                yield Static(self._body(), id="queued-diff")
            yield Static(self._actions(), id="queued-actions")

    def _body(self) -> Text:
        """Build the token-level staged correction preview."""
        body = Text(no_wrap=False)
        body.append("This edit is staged in the TUI.\n", style="bold")
        body.append(f"Total staged edits: {self.count}\n\n")
        body.append_text(styled_before_after(self.edit.original_text, self.edit.corrected_text))
        return body

    def _actions(self) -> str:
        """Return available actions for the staged correction modal."""
        return (
            "Press [b]s[/b] to save and run correction in the TUI.\n"
            "Press [b]Enter[/b] to keep reviewing; the sample stays marked as edited."
        )

    def action_close_feedback(self) -> None:
        """Close feedback and continue reviewing."""
        self.dismiss(None)

    def action_save_and_run(self) -> None:
        """Save review state and run correction processing."""
        self.dismiss(None)
        self.app.action_save()
