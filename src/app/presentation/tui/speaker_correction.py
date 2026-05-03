"""Inline transcript correction popup for project speaker review."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from app.models import SentenceSegment


@dataclass(frozen=True, slots=True)
class SentenceCorrectionEdit:
    """One sentence text edit made inside the speaker review TUI."""

    sentence_id: int | None
    speaker_id: int | None
    begin_time_ms: int
    end_time_ms: int
    original_text: str
    corrected_text: str


class CorrectionInput(Input):
    """Correction input with local cancel handling."""

    BINDINGS = [Binding("escape", "cancel_correction", "Cancel", show=False)]

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
        width: 88;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #correction-title {
        text-style: bold;
    }
    #correction-original {
        color: $text-muted;
        margin: 1 0;
    }
    #correction-input {
        height: 3;
    }
    #correction-status {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_correction", "Cancel", show=False),
        Binding("q", "cancel_correction", "Cancel"),
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
            yield Static(f"Original: {original}", id="correction-original")
            yield CorrectionInput(value=original, id="correction-input")
            yield Static("Enter applies this edit. Esc cancels.", id="correction-status")

    def on_mount(self) -> None:
        """Focus and select the correction input."""
        field = self.query_one("#correction-input", Input)
        field.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Return the edited sentence text to the parent TUI."""
        event.stop()
        corrected = event.value.strip()
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
        width: 82;
        height: auto;
        border: thick $success;
        padding: 1 2;
        background: $surface;
    }
    #queued-title {
        text-style: bold;
        color: $success;
    }
    #queued-body {
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("enter", "close_feedback", "Continue"),
        Binding("escape", "close_feedback", "Continue", show=False),
        Binding("q", "close_feedback", "Continue"),
        Binding("s", "save_and_run", "Save and run"),
    ]

    def __init__(self, edit: SentenceCorrectionEdit) -> None:
        """
        Create staged-correction feedback.

        Args:
            edit: Staged sentence correction.
        """
        super().__init__()
        self.edit = edit

    def compose(self) -> ComposeResult:
        """Build staged correction feedback."""
        body = "\n".join(
            [
                "[b]This edit is staged in the TUI.[/b]",
                "",
                f"Before: {self.edit.original_text}",
                f"After:  {self.edit.corrected_text}",
                "",
                "Press [b]s[/b] to save speaker names and run full-document correction.",
                "Press [b]Enter[/b] to keep reviewing; the sample stays marked as edited.",
            ]
        )
        with Vertical(id="queued-box"):
            yield Static("Transcript correction staged", id="queued-title")
            yield Static(body, id="queued-body")

    def action_close_feedback(self) -> None:
        """Close feedback and continue reviewing."""
        self.dismiss(None)

    def action_save_and_run(self) -> None:
        """Save review state so the CLI can run correction processing."""
        self.app.action_save()
