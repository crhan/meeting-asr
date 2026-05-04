"""Save workflow modal for project speaker review."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import Static

from app.correction_types import CorrectionEditSummary
from app.presentation.tui.diff_render import (
    append_segmented_line,
    styled_unified_diff,
    word_diff_segments,
)


@dataclass(frozen=True, slots=True)
class SpeakerReviewSaveOutcome:
    """Result shown after saving project review state."""

    mapping_path: Path | None
    transcript_path: Path | None
    srt_path: Path | None
    correction_summary: CorrectionEditSummary | None = None


@dataclass(frozen=True, slots=True)
class CorrectionProposalSelection:
    """User selection returned by the proposal review modal."""

    proposal_path: Path
    selected_indices: tuple[int, ...]
    accept_now: bool = False


@dataclass(frozen=True, slots=True)
class ProposalChangeView:
    """One proposal change shown in the TUI."""

    index: int
    sentence_id: int | None
    speaker_name: str
    original_text: str
    corrected_text: str


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
        accept_handler: Callable[[Path | None, tuple[int, ...] | None], SpeakerReviewSaveOutcome] | None,
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
        self.selected_change_indices: tuple[int, ...] | None = None
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
        if self.selected_change_indices is not None and not self.selected_change_indices:
            self.query_one("#save-actions", Static).update("No changes selected. Press d to select changes.")
            return
        self._set_running("Accepting correction proposal...")
        self.run_worker(
            lambda: self.accept_handler(proposal_path, self.selected_change_indices),
            group="speaker-review-save",
            name="accept",
            thread=True,
        )

    def action_view_diff(self) -> None:
        """Open the proposal diff inside the TUI."""
        if self.running:
            return
        diff_path = self._pending_diff_path()
        proposal_path = self._pending_proposal_path()
        if diff_path is None or proposal_path is None:
            return
        self.app.push_screen(
            CorrectionProposalDiffScreen(
                diff_path=diff_path,
                proposal_path=proposal_path,
                selected_indices=self.selected_change_indices,
            ),
            self._handle_proposal_selection,
        )

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
            count = self._selected_count_label()
            return f"Press d to review/select changes | a to accept {count} | Enter to continue reviewing"
        return "Press Enter to continue reviewing | q quits from the main screen"

    def _handle_proposal_selection(self, selection: CorrectionProposalSelection | None) -> None:
        """Store selected proposal changes or accept them immediately."""
        if selection is None:
            return
        self.selected_change_indices = selection.selected_indices
        self.query_one("#save-actions", Static).update(self._actions())
        if selection.accept_now:
            self.action_accept_proposal()

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

    def _selected_count_label(self) -> str:
        """Return selected change count text for actions."""
        summary = None if self.outcome is None else self.outcome.correction_summary
        total = 0 if summary is None else summary.proposed_change_count
        if self.selected_change_indices is None:
            return f"all {total} change(s)"
        return f"{len(self.selected_change_indices)}/{total} change(s)"

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


class CorrectionProposalDiffScreen(ModalScreen[CorrectionProposalSelection | None]):
    """Scrollable modal for inspecting and selecting pending correction changes."""

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
    #diff-path {
        height: 1;
        color: $text-muted;
    }
    #diff-legend {
        height: 1;
        color: $text-muted;
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
        Binding("j", "next_change", "Next change"),
        Binding("k", "previous_change", "Previous change"),
        Binding("down", "next_change", "Next change", show=False),
        Binding("up", "previous_change", "Previous change", show=False),
        Binding("pagedown", "page_down", "Page down"),
        Binding("pageup", "page_up", "Page up"),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
        Binding("x", "toggle_change", "Toggle"),
        Binding("a", "accept_selected", "Apply selected"),
        Binding("enter", "close_diff", "Back"),
        Binding("escape", "close_diff", "Back", show=False),
        Binding("q", "close_diff", "Back"),
    ]

    def __init__(
        self,
        *,
        diff_path: Path,
        proposal_path: Path,
        selected_indices: tuple[int, ...] | None,
    ) -> None:
        """
        Create a diff inspection modal.

        Args:
            diff_path: Proposal diff path.
            proposal_path: Proposal JSON path.
            selected_indices: Existing selected proposed change indices.
        """
        super().__init__()
        self.diff_path = diff_path
        self.proposal_path = proposal_path
        self.changes = _load_proposal_changes(proposal_path)
        self.current_change_index = 0
        if selected_indices is None:
            self.selected_indices = {change.index for change in self.changes}
        else:
            self.selected_indices = set(selected_indices)

    def compose(self) -> ComposeResult:
        """Build diff inspection layout."""
        with Vertical(id="diff-box"):
            yield Static("Correction proposal diff", id="diff-title")
            yield Static(escape(str(self.diff_path)), id="diff-path")
            yield Static(self._legend(), id="diff-legend")
            with ScrollableContainer(id="diff-scroll"):
                yield Static(self._diff_renderable(), id="diff-content")
            yield Static("up/down or j/k choose change | x include/exclude | a apply selected | Enter returns", id="diff-actions")

    def action_page_down(self) -> None:
        """Scroll the diff one page down."""
        self.query_one("#diff-scroll", ScrollableContainer).scroll_page_down()

    def action_page_up(self) -> None:
        """Scroll the diff one page up."""
        self.query_one("#diff-scroll", ScrollableContainer).scroll_page_up()

    def action_scroll_home(self) -> None:
        """Scroll the diff to the top."""
        self.query_one("#diff-scroll", ScrollableContainer).scroll_home()

    def action_scroll_end(self) -> None:
        """Scroll the diff to the bottom."""
        self.query_one("#diff-scroll", ScrollableContainer).scroll_end()

    def action_close_diff(self) -> None:
        """Close the diff modal."""
        self.dismiss(self._selection(accept_now=False))

    def action_next_change(self) -> None:
        """Select the next proposed change."""
        self._move_change(1)

    def action_previous_change(self) -> None:
        """Select the previous proposed change."""
        self._move_change(-1)

    def action_toggle_change(self) -> None:
        """Toggle whether the current proposed change will be accepted."""
        if not self.changes:
            return
        index = self.changes[self.current_change_index].index
        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        self._refresh_diff()

    def action_accept_selected(self) -> None:
        """Return selected changes and request immediate acceptance."""
        self.dismiss(self._selection(accept_now=True))

    def _diff_renderable(self) -> Text:
        """Return styled diff text or a readable error."""
        if self.changes:
            return _styled_proposal_changes(
                self.changes,
                selected_indices=self.selected_indices,
                current_index=self.current_change_index,
            )
        try:
            text = self.diff_path.read_text(encoding="utf-8")
        except OSError as exc:
            return Text(f"Unable to read diff: {exc}", style="red")
        return styled_unified_diff(text)

    def _legend(self) -> str:
        """Return current selection legend."""
        return (
            f"[green]selected {len(self.selected_indices)}/{len(self.changes)}[/]  "
            "[bold]up/down j/k[/] change  [bold]x[/] include/exclude"
        )

    def _move_change(self, delta: int) -> None:
        """Move current proposed change selection."""
        if not self.changes:
            return
        self.current_change_index = max(0, min(len(self.changes) - 1, self.current_change_index + delta))
        self._refresh_diff()

    def _refresh_diff(self) -> None:
        """Refresh proposal review content after selection changes."""
        self.query_one("#diff-legend", Static).update(self._legend())
        self.query_one("#diff-content", Static).update(self._diff_renderable())

    def _selection(self, *, accept_now: bool) -> CorrectionProposalSelection:
        """Return current selection state."""
        return CorrectionProposalSelection(
            proposal_path=self.proposal_path,
            selected_indices=tuple(sorted(self.selected_indices)),
            accept_now=accept_now,
        )


def _path_lines(outcome: SpeakerReviewSaveOutcome) -> list[str]:
    """Render saved speaker artifact paths."""
    return [
        f"- Mapping: {escape(str(outcome.mapping_path))}",
        f"- Transcript: {escape(str(outcome.transcript_path))}",
        f"- Subtitle: {escape(str(outcome.srt_path))}",
    ]


def _summary_lines(summary: CorrectionEditSummary) -> list[str]:
    """Render correction summary fields."""
    state = _correction_summary_state(summary)
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


def _correction_summary_state(summary: CorrectionEditSummary) -> str:
    """Return a human-readable correction workflow state."""
    if summary.accepted:
        return "accepted"
    if summary.proposal_json_path is not None:
        return "proposal ready"
    if summary.sample_change_count == 0 and summary.proposed_change_count == 0:
        return "no transcript changes"
    return "no proposal"


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


def _load_proposal_changes(proposal_path: Path) -> list[ProposalChangeView]:
    """Load proposed changes for selective TUI review."""
    try:
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    changes = payload.get("proposed_changes") if isinstance(payload, dict) else None
    if not isinstance(changes, list):
        return []
    return [
        _proposal_change_view(index, item)
        for index, item in enumerate(changes)
        if isinstance(item, dict)
    ]


def _proposal_change_view(index: int, payload: dict) -> ProposalChangeView:
    """Parse one proposal change for TUI review."""
    sentence_id = payload.get("sentence_id")
    return ProposalChangeView(
        index=index,
        sentence_id=int(sentence_id) if sentence_id not in (None, "") else None,
        speaker_name=str(payload.get("speaker_name") or ""),
        original_text=str(payload.get("original_text") or ""),
        corrected_text=str(payload.get("corrected_text") or ""),
    )


def _styled_proposal_changes(
    changes: list[ProposalChangeView],
    *,
    selected_indices: set[int],
    current_index: int,
) -> Text:
    """Return selectable, token-styled proposal changes."""
    rendered = Text(no_wrap=False)
    for position, change in enumerate(changes):
        current = position == current_index
        selected = change.index in selected_indices
        _append_change_header(rendered, change, position, len(changes), selected, current)
        old_segments, new_segments = word_diff_segments(change.original_text, change.corrected_text)
        append_segmented_line(rendered, "- ", old_segments, removed=True)
        append_segmented_line(rendered, "+ ", new_segments, removed=False)
        rendered.append("\n")
    return rendered


def _append_change_header(
    rendered: Text,
    change: ProposalChangeView,
    position: int,
    total: int,
    selected: bool,
    current: bool,
) -> None:
    """Append one selectable proposal change header."""
    marker = ">" if current else " "
    checkbox = "[x]" if selected else "[ ]"
    label = f"{marker} {checkbox} Change {position + 1}/{total}"
    details = f" sentence_id={change.sentence_id} speaker={change.speaker_name}"
    style = "bold white on dark_blue" if current else "bold"
    rendered.append(label + details + "\n", style=style)
