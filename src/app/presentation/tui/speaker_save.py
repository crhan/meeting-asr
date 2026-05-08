"""Save workflow modal for project speaker review."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
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
from app.presentation.tui.i18n import tr


@dataclass(frozen=True, slots=True)
class SpeakerReviewSaveOutcome:
    """Result shown after saving project review state."""

    mapping_path: Path | None
    transcript_path: Path | None
    srt_path: Path | None
    correction_summary: CorrectionEditSummary | None = None


@dataclass(frozen=True, slots=True)
class SpeakerReviewNameChange:
    """One speaker name change written by project review save."""

    label: str
    before: str | None
    after: str


@dataclass(frozen=True, slots=True)
class SpeakerReviewIgnoreChange:
    """One speaker ignore-state change written by project review save."""

    label: str
    before: bool
    after: bool


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
        Binding("v", "followup", "Follow-up"),
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
        followup_handler: Callable[[], None] | None = None,
        followup_label: str = "continue",
        speaker_changes: Sequence[SpeakerReviewNameChange] = (),
        ignore_changes: Sequence[SpeakerReviewIgnoreChange] = (),
    ) -> None:
        """
        Create save workflow screen.

        Args:
            decision: Speaker review decision to persist.
            save_handler: Function that writes mapping and prepares corrections.
            accept_handler: Function that accepts a pending correction proposal.
            on_result: Callback used to update the parent TUI after success.
            followup_handler: Optional action to run after closing the save modal.
            followup_label: Human-readable follow-up action shown in the modal.
            speaker_changes: Human-readable speaker name diff for this save.
            ignore_changes: Human-readable speaker ignore diff for this save.
        """
        super().__init__()
        self.decision = decision
        self.save_handler = save_handler
        self.accept_handler = accept_handler
        self.on_result = on_result
        self.followup_handler = followup_handler
        self.followup_label = followup_label
        self.speaker_changes = tuple(speaker_changes)
        self.ignore_changes = tuple(ignore_changes)
        self.outcome: SpeakerReviewSaveOutcome | None = None
        self.selected_change_indices: tuple[int, ...] | None = None
        self.running = False
        self.error: str | None = None

    def compose(self) -> ComposeResult:
        """Build modal layout."""
        with Vertical(id="save-box"):
            yield Static(tr("Saving project review", "正在保存 Project Review"), id="save-title")
            yield Static(tr("Starting save workflow...", "正在启动保存流程..."), id="save-body")
            yield Static(tr("Working...", "处理中..."), id="save-actions")

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
            self.query_one("#save-actions", Static).update(tr("No changes selected. Press d to select changes.", "没有选中修改。按 d 选择修改。"))
            return
        self._set_running(tr("Accepting correction proposal...", "正在接受修正建议..."))
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

    def action_followup(self) -> None:
        """Close the modal and run the configured follow-up workflow."""
        if self.running or self.followup_handler is None or self.outcome is None:
            return
        self.dismiss(None)
        self.app.call_after_refresh(self.followup_handler)

    def _start_save(self) -> None:
        """Run the initial save workflow."""
        self._set_running(tr("Saving speaker names and preparing corrections...", "正在保存 speaker 姓名并准备文字修正..."))
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
        self.query_one("#save-title", Static).update(tr("[red]Project review save failed[/]", "[red]Project Review 保存失败[/]"))
        self.query_one("#save-body", Static).update(escape(error))
        self.query_one("#save-actions", Static).update(tr("Press Enter to return to review.", "按 Enter 返回 review。"))

    def _set_running(self, message: str) -> None:
        """Render a running state."""
        self.running = True
        self.error = None
        self.query_one("#save-title", Static).update(tr("Saving project review", "正在保存 Project Review"))
        self.query_one("#save-body", Static).update(escape(message))
        self.query_one("#save-actions", Static).update(tr("Working...", "处理中..."))

    def _title(self) -> str:
        """Return the current modal title."""
        summary = None if self.outcome is None else self.outcome.correction_summary
        if summary is not None and summary.accepted:
            return tr("[green]Project review saved and correction accepted[/]", "[green]Project Review 已保存，文字修正已接受[/]")
        if self._pending_proposal_path() is not None:
            return tr("[yellow]Project review saved; correction proposal needs review[/]", "[yellow]Project Review 已保存；文字修正建议需要确认[/]")
        return tr("[green]Project review saved[/]", "[green]Project Review 已保存[/]")

    def _body(self) -> str:
        """Render save result details."""
        if self.outcome is None:
            return ""
        lines = [tr("[b]Speaker name changes[/b]", "[b]Speaker 姓名变更[/b]")]
        lines.extend(_speaker_change_lines(self.speaker_changes))
        lines.extend(["", tr("[b]Speaker ignore changes[/b]", "[b]Speaker 忽略变更[/b]")])
        lines.extend(_speaker_ignore_change_lines(self.ignore_changes))
        if self.outcome.correction_summary is not None:
            lines.extend(["", tr("[b]Transcript correction[/b]", "[b]文字修正[/b]")])
            lines.extend(_summary_lines(self.outcome.correction_summary))
        return "\n".join(lines)

    def _actions(self) -> str:
        """Render available next actions."""
        followup = f" | v {self.followup_label}" if self.followup_handler is not None else ""
        if self._pending_proposal_path() is not None:
            count = self._selected_count_label()
            return tr(
                f"Press d to review/select changes | a to accept {count}{followup} | Enter to continue reviewing",
                f"按 d 查看/选择修改 | 按 a 接受 {count}{followup} | Enter 继续 review",
            )
        return tr(
            f"Press Enter to continue reviewing{followup} | q quits from the main screen",
            f"按 Enter 继续 review{followup} | q 从主界面退出",
        )

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
            yield Static(tr("Correction proposal diff", "文字修正建议 diff"), id="diff-title")
            yield Static(escape(str(self.diff_path)), id="diff-path")
            yield Static(self._legend(), id="diff-legend")
            with ScrollableContainer(id="diff-scroll"):
                yield Static(self._diff_renderable(), id="diff-content")
            yield Static(
                tr(
                    "up/down or j/k choose change | x include/exclude | a apply selected | Esc returns",
                    "↑/↓ 或 j/k 选择修改 | x 选中/排除 | a 应用已选 | Esc 返回",
                ),
                id="diff-actions",
            )

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
            return Text(tr(f"Unable to read diff: {exc}", f"无法读取 diff：{exc}"), style="red")
        return styled_unified_diff(text)

    def _legend(self) -> str:
        """Return current selection legend."""
        return (
            tr(
                f"[green]selected {len(self.selected_indices)}/{len(self.changes)}[/]  "
                "[bold]up/down j/k[/] change  [bold]x[/] include/exclude",
                f"[green]已选 {len(self.selected_indices)}/{len(self.changes)}[/]  "
                "[bold]↑/↓ j/k[/] 切换修改  [bold]x[/] 选中/排除",
            )
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


def speaker_name_changes(
    speakers: Sequence[object],
    saved_names_by_speaker: dict[int, str],
) -> tuple[SpeakerReviewNameChange, ...]:
    """
    Build speaker name changes against the saved speaker map.

    Args:
        speakers: Current TUI speaker rows.
        saved_names_by_speaker: Speaker names already persisted in speaker_map.

    Returns:
        Speaker name changes that will be written by save.
    """
    changes = []
    for speaker in speakers:
        if bool(getattr(speaker, "ignored", False)):
            continue
        speaker_id = int(getattr(speaker, "speaker_id"))
        before = saved_names_by_speaker.get(speaker_id)
        after = str(getattr(speaker, "current_name")).strip() or str(getattr(speaker, "label"))
        if before != after:
            changes.append(SpeakerReviewNameChange(str(getattr(speaker, "label")), before, after))
    return tuple(changes)


def speaker_ignore_changes(
    speakers: Sequence[object],
    saved_ignored_speaker_ids: frozenset[int],
) -> tuple[SpeakerReviewIgnoreChange, ...]:
    """
    Build speaker ignore changes against the saved ignore metadata.

    Args:
        speakers: Current TUI speaker rows.
        saved_ignored_speaker_ids: Speaker ids already persisted as ignored.

    Returns:
        Speaker ignore-state changes that will be written by save.
    """
    changes = []
    for speaker in speakers:
        speaker_id = int(getattr(speaker, "speaker_id"))
        before = speaker_id in saved_ignored_speaker_ids
        after = (
            bool(getattr(speaker, "ignored", False))
            and str(getattr(speaker, "current_name")) == str(getattr(speaker, "label"))
        )
        if before != after:
            changes.append(SpeakerReviewIgnoreChange(str(getattr(speaker, "label")), before, after))
    return tuple(changes)


def _speaker_change_lines(changes: Sequence[SpeakerReviewNameChange]) -> list[str]:
    """Render speaker name changes."""
    if not changes:
        return [tr("- No speaker name changes; outputs were regenerated.", "- 无 speaker 姓名变更；仅重新生成产物。")]
    return [
        tr(
            f"- {escape(item.label)}: {escape(item.before or '<not saved>')} -> {escape(item.after)}",
            f"- {escape(item.label)}：{escape(item.before or '<未保存>')} -> {escape(item.after)}",
        )
        for item in changes
    ]


def _speaker_ignore_change_lines(changes: Sequence[SpeakerReviewIgnoreChange]) -> list[str]:
    """Render speaker ignore-state changes."""
    if not changes:
        return [tr("- No speaker ignore changes.", "- 无 speaker 忽略变更。")]
    return [
        tr(
            f"- {escape(item.label)}: {_ignore_state_label(item.before)} -> {_ignore_state_label(item.after)}",
            f"- {escape(item.label)}：{_ignore_state_label(item.before)} -> {_ignore_state_label(item.after)}",
        )
        for item in changes
    ]


def _ignore_state_label(value: bool) -> str:
    """Return a compact ignore-state label."""
    return tr("ignored", "已忽略") if value else tr("active", "未忽略")


def _summary_lines(summary: CorrectionEditSummary) -> list[str]:
    """Render correction summary fields."""
    state = _correction_summary_state(summary)
    lines = [
        tr(f"- State: {state}", f"- 状态：{state}"),
        tr(f"- Sample changes: {summary.sample_change_count}", f"- 样例修改：{summary.sample_change_count}"),
        tr(f"- Proposed changes: {summary.proposed_change_count}", f"- 建议修改：{summary.proposed_change_count}"),
        tr(f"- Changed sentences: {summary.change_count}", f"- 已修改句子：{summary.change_count}"),
    ]
    if summary.proposal_diff_path is not None:
        lines.append(tr(f"- Diff: {escape(str(summary.proposal_diff_path))}", f"- Diff：{escape(str(summary.proposal_diff_path))}"))
    if summary.corrected_named_transcript_path is not None:
        lines.append(
            tr(
                f"- Corrected transcript: {escape(str(summary.corrected_named_transcript_path))}",
                f"- 修正后转写：{escape(str(summary.corrected_named_transcript_path))}",
            )
        )
    lines.extend(_understanding_lines(summary))
    return lines


def _correction_summary_state(summary: CorrectionEditSummary) -> str:
    """Return a human-readable correction workflow state."""
    if summary.accepted:
        return tr("accepted", "已接受")
    if summary.proposal_json_path is not None:
        return tr("proposal ready", "建议已生成")
    if summary.sample_change_count == 0 and summary.proposed_change_count == 0:
        return tr("no transcript changes", "无文字修改")
    return tr("no proposal", "无建议")


def _understanding_lines(summary: CorrectionEditSummary) -> list[str]:
    """Render inferred correction rules."""
    if not summary.understanding:
        return []
    lines = ["", tr("[b]Understanding[/b]", "[b]理解[/b]")]
    for item in summary.understanding:
        count_text = tr(f"({item.proposed_count} proposed)", f"（建议 {item.proposed_count} 处）")
        lines.append(
            f"- {escape(item.wrong_text)} -> {escape(item.corrected_text)} {count_text}"
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
    label = tr(f"{marker} {checkbox} Change {position + 1}/{total}", f"{marker} {checkbox} 修改 {position + 1}/{total}")
    details = tr(
        f" sentence_id={change.sentence_id} speaker={change.speaker_name}",
        f" 句子ID={change.sentence_id} speaker={change.speaker_name}",
    )
    style = "bold white on dark_blue" if current else "bold"
    rendered.append(label + details + "\n", style=style)
