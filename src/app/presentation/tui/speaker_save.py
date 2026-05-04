"""Save workflow modal for project speaker review."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import Static

from app.correction_types import CorrectionEditSummary


@dataclass(frozen=True, slots=True)
class SpeakerReviewSaveOutcome:
    """Result shown after saving project review state."""

    mapping_path: Path | None
    transcript_path: Path | None
    srt_path: Path | None
    correction_summary: CorrectionEditSummary | None = None


class SpeakerReviewSaveScreen(ModalScreen[None]):
    """Modal progress and confirmation screen for project review save."""

    CSS = """
    SpeakerReviewSaveScreen {
        align: center middle;
    }
    #save-box {
        width: 92;
        height: auto;
        max-height: 86%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #save-title {
        text-style: bold;
    }
    #save-body {
        margin: 1 0;
    }
    #save-actions {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("d", "view_diff", "View diff"),
        Binding("a", "accept_proposal", "Accept proposal"),
        Binding("enter", "close_feedback", "Continue"),
        Binding("escape", "close_feedback", "Continue", show=False),
        Binding("q", "close_feedback", "Continue"),
    ]

    def __init__(
        self,
        *,
        decision: Any,
        save_handler: Callable[[Any], SpeakerReviewSaveOutcome],
        accept_handler: Callable[[Path | None], SpeakerReviewSaveOutcome] | None,
        on_result: Callable[[SpeakerReviewSaveOutcome], None],
    ) -> None:
        """
        Create save workflow screen.

        Args:
            decision: Speaker review decision to persist.
            save_handler: Function that writes mapping and prepares corrections.
            accept_handler: Function that accepts a pending correction proposal.
            on_result: Callback used to update the parent TUI after success.
        """
        super().__init__()
        self.decision = decision
        self.save_handler = save_handler
        self.accept_handler = accept_handler
        self.on_result = on_result
        self.outcome: SpeakerReviewSaveOutcome | None = None
        self.running = False
        self.error: str | None = None

    def compose(self) -> ComposeResult:
        """Build modal layout."""
        with Vertical(id="save-box"):
            yield Static("Saving project review", id="save-title")
            yield Static("Starting save workflow...", id="save-body")
            yield Static("Working...", id="save-actions")

    def on_mount(self) -> None:
        """Start saving as soon as the modal is visible."""
        self._start_save()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Update the modal when a save or accept worker finishes."""
        if event.worker.group != "speaker-review-save":
            return
        if event.state == WorkerState.SUCCESS:
            self._complete(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self._fail(str(event.worker.error))

    def action_accept_proposal(self) -> None:
        """Accept the pending full-document correction proposal."""
        if self.running or self.accept_handler is None:
            return
        proposal_path = self._pending_proposal_path()
        if proposal_path is None:
            return
        self._set_running("Accepting correction proposal...")
        self.run_worker(
            lambda: self.accept_handler(proposal_path),
            group="speaker-review-save",
            name="accept",
            thread=True,
        )

    def action_view_diff(self) -> None:
        """Open the proposal diff inside the TUI."""
        if self.running:
            return
        diff_path = self._pending_diff_path()
        if diff_path is None:
            return
        self.app.push_screen(CorrectionProposalDiffScreen(diff_path))

    def action_close_feedback(self) -> None:
        """Close the save feedback modal when no worker is running."""
        if not self.running:
            self.dismiss(None)

    def _start_save(self) -> None:
        """Run the initial save workflow."""
        self._set_running("Saving speaker names and preparing corrections...")
        self.run_worker(self._run_save, group="speaker-review-save", name="save", thread=True)

    def _run_save(self) -> SpeakerReviewSaveOutcome:
        """Call the injected save handler."""
        return self.save_handler(self.decision)

    def _complete(self, outcome: SpeakerReviewSaveOutcome) -> None:
        """Render successful worker outcome."""
        self.running = False
        self.error = None
        merged = self._merged_outcome(outcome)
        self.outcome = merged
        self.on_result(merged)
        self.query_one("#save-title", Static).update(self._title())
        self.query_one("#save-body", Static).update(self._body())
        self.query_one("#save-actions", Static).update(self._actions())

    def _fail(self, error: str) -> None:
        """Render a failed worker outcome."""
        self.running = False
        self.error = error
        self.query_one("#save-title", Static).update("[red]Project review save failed[/]")
        self.query_one("#save-body", Static).update(escape(error))
        self.query_one("#save-actions", Static).update("Press Enter to return to review.")

    def _set_running(self, message: str) -> None:
        """Render a running state."""
        self.running = True
        self.error = None
        self.query_one("#save-title", Static).update("Saving project review")
        self.query_one("#save-body", Static).update(escape(message))
        self.query_one("#save-actions", Static).update("Working...")

    def _title(self) -> str:
        """Return the current modal title."""
        summary = None if self.outcome is None else self.outcome.correction_summary
        if summary is not None and summary.accepted:
            return "[green]Project review saved and correction accepted[/]"
        if self._pending_proposal_path() is not None:
            return "[yellow]Project review saved; correction proposal needs review[/]"
        return "[green]Project review saved[/]"

    def _body(self) -> str:
        """Render save result details."""
        if self.outcome is None:
            return ""
        lines = ["[b]Speaker outputs[/b]"]
        lines.extend(_path_lines(self.outcome))
        if self.outcome.correction_summary is not None:
            lines.extend(["", "[b]Transcript correction[/b]"])
            lines.extend(_summary_lines(self.outcome.correction_summary))
        return "\n".join(lines)

    def _actions(self) -> str:
        """Render available next actions."""
        if self._pending_proposal_path() is not None:
            return "Press d to review diff | a to accept proposal | Enter to continue reviewing"
        return "Press Enter to continue reviewing | q quits from the main screen"

    def _pending_proposal_path(self) -> Path | None:
        """Return the pending proposal JSON path if one needs confirmation."""
        summary = None if self.outcome is None else self.outcome.correction_summary
        if summary is None or summary.accepted:
            return None
        if summary.proposal_json_path is None or summary.proposed_change_count == 0:
            return None
        return summary.proposal_json_path

    def _pending_diff_path(self) -> Path | None:
        """Return the pending diff path if it can be inspected."""
        summary = None if self.outcome is None else self.outcome.correction_summary
        if summary is None or summary.accepted:
            return None
        return summary.proposal_diff_path

    def _merged_outcome(self, outcome: SpeakerReviewSaveOutcome) -> SpeakerReviewSaveOutcome:
        """Preserve speaker output paths when accepting a proposal."""
        if self.outcome is None or outcome.mapping_path is not None:
            return outcome
        return SpeakerReviewSaveOutcome(
            self.outcome.mapping_path,
            self.outcome.transcript_path,
            self.outcome.srt_path,
            outcome.correction_summary,
        )


class CorrectionProposalDiffScreen(ModalScreen[None]):
    """Scrollable modal for inspecting a pending correction diff."""

    CSS = """
    CorrectionProposalDiffScreen {
        align: center middle;
    }
    #diff-box {
        width: 96%;
        height: 90%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #diff-title {
        height: 1;
        text-style: bold;
    }
    #diff-scroll {
        height: 1fr;
        margin: 1 0;
    }
    #diff-actions {
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "scroll_down", "Down"),
        Binding("k", "scroll_up", "Up"),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("pagedown", "page_down", "Page down"),
        Binding("pageup", "page_up", "Page up"),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
        Binding("enter", "close_diff", "Back"),
        Binding("escape", "close_diff", "Back", show=False),
        Binding("q", "close_diff", "Back"),
    ]

    def __init__(self, diff_path: Path) -> None:
        """
        Create a diff inspection modal.

        Args:
            diff_path: Proposal diff path.
        """
        super().__init__()
        self.diff_path = diff_path

    def compose(self) -> ComposeResult:
        """Build diff inspection layout."""
        with Vertical(id="diff-box"):
            yield Static(f"Correction proposal diff: {escape(str(self.diff_path))}", id="diff-title")
            with VerticalScroll(id="diff-scroll"):
                yield Static(self._diff_text(), id="diff-content")
            yield Static("j/k or arrows scroll | PageUp/PageDown | Enter returns", id="diff-actions")

    def action_scroll_down(self) -> None:
        """Scroll the diff down."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll the diff up."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_up()

    def action_page_down(self) -> None:
        """Scroll the diff one page down."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_page_down()

    def action_page_up(self) -> None:
        """Scroll the diff one page up."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_page_up()

    def action_scroll_home(self) -> None:
        """Scroll the diff to the top."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_home()

    def action_scroll_end(self) -> None:
        """Scroll the diff to the bottom."""
        self.query_one("#diff-scroll", VerticalScroll).scroll_end()

    def action_close_diff(self) -> None:
        """Close the diff modal."""
        self.dismiss(None)

    def _diff_text(self) -> str:
        """Return escaped diff text or a readable error."""
        try:
            text = self.diff_path.read_text(encoding="utf-8")
        except OSError as exc:
            return escape(f"Unable to read diff: {exc}")
        return escape(text or "(empty diff)")


def _path_lines(outcome: SpeakerReviewSaveOutcome) -> list[str]:
    """Render saved speaker artifact paths."""
    return [
        f"- Mapping: {escape(str(outcome.mapping_path))}",
        f"- Transcript: {escape(str(outcome.transcript_path))}",
        f"- Subtitle: {escape(str(outcome.srt_path))}",
    ]


def _summary_lines(summary: CorrectionEditSummary) -> list[str]:
    """Render correction summary fields."""
    state = "accepted" if summary.accepted else "proposal ready"
    lines = [
        f"- State: {state}",
        f"- Sample changes: {summary.sample_change_count}",
        f"- Proposed changes: {summary.proposed_change_count}",
        f"- Changed sentences: {summary.change_count}",
    ]
    if summary.proposal_diff_path is not None:
        lines.append(f"- Diff: {escape(str(summary.proposal_diff_path))}")
    if summary.corrected_named_transcript_path is not None:
        lines.append(f"- Corrected transcript: {escape(str(summary.corrected_named_transcript_path))}")
    lines.extend(_understanding_lines(summary))
    return lines


def _understanding_lines(summary: CorrectionEditSummary) -> list[str]:
    """Render inferred correction rules."""
    if not summary.understanding:
        return []
    lines = ["", "[b]Understanding[/b]"]
    for item in summary.understanding:
        lines.append(
            f"- {escape(item.wrong_text)} -> {escape(item.corrected_text)} "
            f"({item.proposed_count} proposed)"
        )
    return lines
