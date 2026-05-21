"""Inline transcript correction popup for project speaker review."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

from app.models import SentenceSegment
from app.presentation.tui.diff_render import styled_before_after
from app.presentation.tui.i18n import tr


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
        Binding("ctrl+o", "open_external_editor", "External editor", show=False, priority=True),
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

    def action_open_external_editor(self) -> None:
        """Open the parent modal's external editor flow."""
        screen = self.app.screen
        if hasattr(screen, "open_external_editor"):
            screen.open_external_editor()


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
        original_text: str | None = None,
        initial_text: str | None = None,
    ) -> None:
        """
        Create a sentence correction modal.

        Args:
            speaker_label: Anonymous speaker label.
            speaker_name: Current speaker display name.
            segment: Selected transcript sentence.
            original_text: Baseline text used for diffing this edit.
            initial_text: Text shown in the editor when the modal opens.
        """
        super().__init__()
        self.speaker_label = speaker_label
        self.speaker_name = speaker_name
        self.segment = segment
        self.original_text = (original_text if original_text is not None else segment.text).strip()
        self.initial_text = (initial_text if initial_text is not None else segment.text).strip()

    def compose(self) -> ComposeResult:
        """Build the correction popup."""
        title = tr(
            f"Edit Transcript Text | {self.speaker_label} / {self.speaker_name}",
            f"编辑转写文本 | {self.speaker_label} / {self.speaker_name}",
        )
        with Vertical(id="correction-box"):
            yield Static(title, id="correction-title")
            with ScrollableContainer(id="correction-original"):
                yield Static(tr(f"[b]Original:[/]\n{self.original_text}", f"[b]原文：[/]\n{self.original_text}"))
            yield CorrectionTextArea(self.initial_text, id="correction-input", soft_wrap=True, show_line_numbers=False)
            yield Static(
                tr(
                    "Enter applies this edit. Esc cancels. Ctrl-O opens $EDITOR. Ctrl-F/Ctrl-B move cursor.",
                    "Enter 应用修改。Esc 取消。Ctrl-O 打开 $EDITOR。Ctrl-F/Ctrl-B 移动光标。",
                ),
                id="correction-status",
            )

    def on_mount(self) -> None:
        """Focus the correction editor."""
        field = self.query_one("#correction-input", TextArea)
        field.focus()
        field.cursor_location = _text_end_location(self.initial_text)

    def submit_correction(self, value: str) -> None:
        """Return the edited sentence text to the parent TUI."""
        corrected = value.strip()
        original = self.original_text
        if not corrected:
            self.query_one("#correction-status", Static).update(tr("Corrected text cannot be empty.", "修正后的文本不能为空。"))
            return
        if corrected == original and self.initial_text == original:
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

    def open_external_editor(self) -> None:
        """Edit the current text in the user's terminal editor."""
        field = self.query_one("#correction-input", TextArea)
        try:
            edited = self._run_external_editor(field.text)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#correction-status", Static).update(
                tr(f"External editor failed: {exc}", f"外部编辑器失败：{exc}")
            )
            return
        updated = edited.strip()
        if not updated:
            self.query_one("#correction-status", Static).update(
                tr("External editor returned empty text; keeping current text.", "外部编辑器返回空文本，已保留当前文本。")
            )
            return
        field.text = updated
        field.cursor_location = _text_end_location(updated)
        field.focus()

    def _run_external_editor(self, text: str) -> str:
        """Run ``$VISUAL`` or ``$EDITOR`` on a temporary UTF-8 text file."""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.write("\n")
        try:
            with self.app.suspend():
                completed = subprocess.run([*shlex.split(editor), str(temp_path)], check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{editor} exited with {completed.returncode}")
            return temp_path.read_text(encoding="utf-8")
        finally:
            temp_path.unlink(missing_ok=True)

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
            yield Static(tr("Transcript correction staged", "文字修正已暂存"), id="queued-title")
            with ScrollableContainer(id="queued-body"):
                yield Static(self._body(), id="queued-diff")
            yield Static(self._actions(), id="queued-actions")

    def _body(self) -> Text:
        """Build the token-level staged correction preview."""
        body = Text(no_wrap=False)
        body.append(tr("This edit is staged in the TUI.\n", "这条修改已暂存在 TUI 中。\n"), style="bold")
        body.append(tr(f"Total staged edits: {self.count}\n\n", f"当前暂存修改数：{self.count}\n\n"))
        body.append_text(styled_before_after(self.edit.original_text, self.edit.corrected_text))
        return body

    def _actions(self) -> str:
        """Return available actions for the staged correction modal."""
        return (
            tr(
                "Press [b]s[/b] to save and run correction in the TUI.\n"
                "Press [b]Enter[/b] to keep reviewing; the sample stays marked as edited.",
                "按 [b]s[/b] 在 TUI 内保存并运行修正流程。\n按 [b]Enter[/b] 继续 review；该 sample 会保持 edited 标记。",
            )
        )

    def action_close_feedback(self) -> None:
        """Close feedback and continue reviewing."""
        self.dismiss(None)

    def action_save_and_run(self) -> None:
        """Save review state and run correction processing."""
        self.dismiss(None)
        self.app.action_save()


def _text_end_location(text: str) -> tuple[int, int]:
    """Return the TextArea cursor location at the end of ``text``."""
    lines = text.splitlines() or [""]
    return len(lines) - 1, len(lines[-1])
