"""Textual UI for reviewing project speaker names."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.worker import Worker, WorkerState
from textual.widgets import Header, Static

from app.models import SentenceSegment
from app.speaker_match_status import best_candidate_name
from app.speaker_review import build_audio_preview_command
from app.voiceprint_embedding import VoiceprintEmbedSummary, embed_voiceprint_samples
from app.presentation.tui.project import ProjectPickerScreen, load_project_picker_session
from app.presentation.tui.speaker_correction import (
    CorrectionQueuedScreen,
    SentenceCorrectionEdit,
    SentenceCorrectionScreen,
)
from app.presentation.tui.i18n import tr
from app.presentation.tui.speaker_help import ShortcutHelpScreen, browse_status, edit_status
from app.presentation.tui.speaker_identity import IdentityEditScreen, IdentitySelection
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_models import ReviewSpeaker, SpeakerReviewDecision, SpeakerReviewSession
from app.presentation.tui.speaker_people import (
    KnownPerson,
    find_person_by_name,
)
from app.presentation.tui.speaker_rematch import (
    SpeakerRematchProcessingScreen,
    SpeakerRematchResult,
    compact_rematch_line,
    run_speaker_rematch,
)
from app.presentation.tui.speaker_save import SpeakerReviewSaveOutcome, SpeakerReviewSaveScreen, speaker_name_changes
from app.presentation.tui.speaker_session import load_speaker_review_session, load_voiceprint_review_progress
from app.presentation.tui.speaker_status import (
    SpeakerReviewOverview,
    VoiceprintReviewProgress,
    match_badge,
    render_overview_pane,
    render_selected_speaker_line,
    speaker_status,
    status_icon,
    status_style,
)
from app.presentation.tui.voiceprint_review import (
    VoiceprintReviewScreen,
    VoiceprintReviewWorkflowSummary,
    load_voiceprint_review_session,
)
from app.utils import format_ms_timestamp

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"

__all__ = [
    "FOCUSED_PANE_CLASS",
    "IdentityEditScreen",
    "KnownPerson",
    "ReviewSpeaker",
    "SentenceCorrectionScreen",
    "ShortcutHelpScreen",
    "SpeakerMatchCandidate",
    "SpeakerRematchProcessingScreen",
    "SpeakerRematchResult",
    "SpeakerReviewApp",
    "SpeakerReviewDecision",
    "SpeakerReviewOverview",
    "SpeakerReviewSession",
    "UNFOCUSED_PANE_CLASS",
    "VoiceprintReviewProgress",
    "load_speaker_review_session",
    "run_speaker_review_tui",
]


class SpeakerReviewApp(App[SpeakerReviewDecision]):
    """Keyboard-first TUI for reviewing project speaker identities."""

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
        width: 30%;
    }
    #samples {
        width: 70%;
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
        Binding("/", "edit_name", "Edit name"),
        Binding("a", "accept_match", "Accept match"),
        Binding("i", "ignore_speaker", "Ignore"),
        Binding("e", "edit_sample_text", "Edit text"),
        Binding("c", "edit_sample_text", "Correct text", show=False),
        Binding("p", "switch_project", "Switch project"),
        Binding("v", "voiceprint_review", "Voiceprint"),
        Binding("m", "rematch_speakers", "Rematch"),
        Binding("b", "embed_voiceprints", "Embed voiceprints"),
        Binding("?", "show_shortcuts", "Help"),
        Binding("s", "save", "Save"),
        Binding("q", "quit_review", "Quit"),
    ]

    def __init__(
        self,
        session: SpeakerReviewSession,
        *,
        save_handler: Callable[[SpeakerReviewDecision], SpeakerReviewSaveOutcome] | None = None,
        accept_handler: Callable[[Path | None, tuple[int, ...] | None], SpeakerReviewSaveOutcome] | None = None,
        project_save_handler: Callable[[Path, SpeakerReviewDecision], SpeakerReviewSaveOutcome] | None = None,
        project_accept_handler: Callable[
            [Path, Path | None, tuple[int, ...] | None],
            SpeakerReviewSaveOutcome,
        ]
        | None = None,
    ) -> None:
        """
        Create the TUI app.

        Args:
            session: Speaker review inputs.
            save_handler: Optional in-TUI save workflow callback.
            accept_handler: Optional in-TUI correction proposal accept callback.
            project_save_handler: Project-aware save callback used after switching projects.
            project_accept_handler: Project-aware correction accept callback used after switching projects.
        """
        super().__init__()
        self.session = session
        self.save_handler = save_handler
        self.accept_handler = accept_handler
        self.project_save_handler = project_save_handler
        self.project_accept_handler = project_accept_handler
        self.selected_speaker_index = 0
        self.focused_column = "speakers"
        self.playback_process: subprocess.Popen | None = None
        self.known_people = list(session.people)
        self.correction_edits: list[SentenceCorrectionEdit] = []
        self.identity_baseline = _identity_snapshot(session.speakers)

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(browse_status(), id="status")

    def on_mount(self) -> None:
        """Render the initial review state."""
        self._enter_browse_mode(browse_status())
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when the TUI closes."""
        self._stop_playback()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Update project review after an embedded voiceprint capture finishes."""
        if event.worker.group == "speaker-review-voiceprint-embed":
            self._handle_voiceprint_embed_state(event)
            return
        if event.worker.group == "speaker-review-rematch":
            self._handle_speaker_rematch_state(event)
            return

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
        """Play or stop the selected sample as audio."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止 sample 播放。"))
            return
        try:
            self._play_sample(self._selected_sample())
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Preview failed: {exc}", f"预览失败：{exc}"))

    def action_edit_name(self) -> None:
        """Open the explicit known-person selection modal."""
        speaker = self._speaker()
        self._set_status(edit_status())
        self.push_screen(
            IdentityEditScreen(
                speaker_label=speaker.label,
                current_name=speaker.current_name,
                current_person_id=speaker.person_id,
                match=speaker.match,
                people=self.known_people,
                store_dir=self.session.store_dir,
            ),
            self._handle_identity_selection,
        )

    def action_accept_match(self) -> None:
        """Accept the current voiceprint match candidate."""
        speaker = self._speaker()
        candidate = None if speaker.match is None else best_candidate_name(speaker.match)
        if candidate is None:
            self._set_status(tr("No usable match for this speaker.", "当前 speaker 没有可用匹配。"))
            return
        person_id = None if speaker.match is None else speaker.match.best_person_id
        if person_id is None:
            person = find_person_by_name(candidate, self.known_people)
            person_id = None if person is None else person.person_id
        person_public_id = _known_person_public_id(self.known_people, person_id)
        person_public_id = person_public_id or (None if speaker.match is None else speaker.match.best_person_public_id)
        self._set_speaker_identity(speaker, candidate, person_id, person_public_id)
        self._set_status(tr(f"Accepted match for {speaker.label}: {candidate}.", f"已接受 {speaker.label} 的匹配：{candidate}。"))
        self._refresh()

    def action_ignore_speaker(self) -> None:
        """Keep the anonymous speaker label so voiceprint capture skips it."""
        speaker = self._speaker()
        speaker.current_name = speaker.label
        speaker.ignored = True
        speaker.person_id = None
        speaker.person_public_id = None
        self._set_status(
            tr(
                f"Ignored {speaker.label}; it will stay anonymous and be skipped by capture.",
                f"已忽略 {speaker.label}；它会保持匿名，并在声纹采样中跳过。",
            )
        )
        self._refresh()

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.push_screen(ShortcutHelpScreen())

    def action_switch_project(self) -> None:
        """Open the embedded project picker and switch review context."""
        if self._has_unsaved_review_changes():
            self._set_status(tr("Save current project changes with s before switching projects.", "切换项目之前请先按 s 保存当前项目修改。"))
            return
        try:
            picker_session = load_project_picker_session(self.session.projects_dir)
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Project switch unavailable: {exc}", f"项目切换不可用：{exc}"))
            return
        if not picker_session.projects:
            self._set_status(tr("No projects found to switch to.", "没有可切换的项目。"))
            return
        self.push_screen(ProjectPickerScreen(picker_session), self._handle_project_switch)

    def action_voiceprint_review(self) -> None:
        """Open the embedded voiceprint review screen from project review."""
        if self._has_unsaved_speaker_names():
            self._set_status(tr("Save speaker names with s before opening voiceprint review.", "打开声纹 Review 前请先按 s 保存 speaker 姓名。"))
            return
        try:
            session, planned = load_voiceprint_review_session(
                project_dir=self.session.project_dir,
                store_dir=self.session.store_dir,
                page_size=self.session.page_size,
                return_hint=tr("return to Project Review", "返回 Project Review"),
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Voiceprint review unavailable: {exc}", f"声纹 Review 不可用：{exc}"))
            return
        self.push_screen(
            VoiceprintReviewScreen(
                session,
                project_dir=self.session.project_dir,
                planned=planned,
                store_dir=self.session.store_dir,
                on_complete=self._handle_voiceprint_review_workflow,
            )
        )

    def action_rematch_speakers(self) -> None:
        """Rerun speaker matching against the current global voiceprint library."""
        if self._has_unsaved_review_changes():
            self._set_status(tr("Save current review changes with s before rematching speakers.", "重新匹配前请先按 s 保存当前 review 修改。"))
            return
        self._stop_playback()
        self.push_screen(SpeakerRematchProcessingScreen())
        self.run_worker(
            lambda: run_speaker_rematch(
                self.session.project_dir,
                store_dir=self.session.store_dir,
                page_size=self.session.page_size,
                allow_correction=self.session.allow_correction,
            ),
            group="speaker-review-rematch",
            name="rematch",
            thread=True,
        )

    def action_embed_voiceprints(self) -> None:
        """Generate embeddings for captured voiceprint samples from Project Review."""
        if not self.session.overview.voiceprint.captured_sample_ids:
            self._set_status(
                tr(
                    "No captured voiceprint samples yet. Press v to capture voiceprints first.",
                    "还没有已采集的声纹样本。请先按 v 进行声纹采样。",
                )
            )
            return
        self._set_status(tr("Embedding voiceprint samples...", "正在生成声纹 embedding..."))
        self.run_worker(
            lambda: embed_voiceprint_samples(
                store_dir=self.session.store_dir,
                provider=None,
                endpoint=None,
                model=None,
                rebuild=False,
            ),
            group="speaker-review-voiceprint-embed",
            name="embed",
            thread=True,
        )

    def action_save(self) -> None:
        """Save review state or return the reviewed mapping to the CLI command."""
        decision = self._decision()
        if self.save_handler is None and self.project_save_handler is None:
            self.exit(decision)
            return
        self.push_screen(
            SpeakerReviewSaveScreen(
                decision=decision,
                save_handler=self._save_handler_for_current_project(),
                accept_handler=self._accept_handler_for_current_project(),
                on_result=self._handle_save_outcome,
                followup_handler=self.action_voiceprint_review,
                followup_label=tr("capture voiceprints", "声纹采样"),
                speaker_changes=speaker_name_changes(
                    self.session.speakers,
                    self.session.overview.saved_names_by_speaker,
                ),
            )
        )

    def action_edit_sample_text(self) -> None:
        """Edit the selected transcript sentence inside the TUI."""
        if not self.session.allow_correction:
            self._set_status(
                tr(
                    "Transcript correction is available from project review, not speaker-only review.",
                    "文字修正只能在 Project Review 中使用，speaker-only review 不支持。",
                )
            )
            return
        speaker = self._speaker()
        self.push_screen(
            SentenceCorrectionScreen(
                speaker_label=speaker.label,
                speaker_name=speaker.current_name,
                segment=self._selected_sample(),
            ),
            self._handle_sentence_correction,
        )

    def action_quit_review(self) -> None:
        """Exit without saving."""
        self.exit(SpeakerReviewDecision(saved=False, mapping={}, action="quit"))

    def _handle_sentence_correction(self, edit: SentenceCorrectionEdit | None) -> None:
        """Record an inline sentence correction returned by the modal."""
        if edit is None:
            self._set_status(tr("Transcript correction canceled.", "已取消文字修正。"))
            return
        self._upsert_correction_edit(edit)
        self._replace_segment_text(edit)
        count = len(self.correction_edits)
        self._set_status(
            tr(
                f"{count} text correction(s) staged. Press s to save and run correction.",
                f"已暂存 {count} 条文字修正。按 s 保存并运行修正流程。",
            )
        )
        self._refresh()
        self.push_screen(CorrectionQueuedScreen(edit, count=count))

    def _handle_identity_selection(self, selection: IdentitySelection | None) -> None:
        """Apply one identity selected in the modal."""
        if selection is None:
            self._set_status(tr("Identity edit canceled.", "已取消身份编辑。"))
            return
        speaker = self._speaker()
        self._remember_known_person(selection)
        self._set_speaker_identity(speaker, selection.name, selection.person_id, selection.public_id)
        action = tr("Created", "已创建") if selection.created else tr("Set", "已设置")
        self._set_status(f"{action} {speaker.label} -> {selection.name} {selection.public_id}.")
        self._refresh()

    def _handle_project_switch(self, project_dir: Path | None) -> None:
        """Replace the current review session with another project."""
        if project_dir is None:
            self._set_status(tr("Project switch canceled.", "已取消项目切换。"))
            return
        if project_dir.resolve() == self.session.project_dir.resolve():
            self._set_status(tr("Already reviewing the selected project.", "已经在 review 当前选中的项目。"))
            return
        try:
            session = load_speaker_review_session(
                project_dir,
                page_size=self.session.page_size,
                store_dir=self.session.store_dir,
                allow_correction=self.session.allow_correction,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(tr(f"Project switch failed: {exc}", f"项目切换失败：{exc}"))
            return
        self._stop_playback()
        self.session = session
        self.selected_speaker_index = 0
        self.focused_column = "speakers"
        self.known_people = list(session.people)
        self.correction_edits.clear()
        self.identity_baseline = _identity_snapshot(session.speakers)
        self._enter_browse_mode(tr(f"Switched to project {session.overview.project_id}.", f"已切换到项目 {session.overview.project_id}。"))
        self._refresh()

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
        self._enter_browse_mode(browse_status())
        self._refresh()

    def _move_speaker(self, delta: int) -> None:
        """Move the selected speaker index."""
        total = len(self.session.speakers)
        self.selected_speaker_index = (self.selected_speaker_index + delta) % total
        self._refresh()

    def _move_sample(self, delta: int) -> None:
        """Move the selected sample index."""
        speaker = self._speaker()
        target = speaker.selected_sample_index + delta
        speaker.selected_sample_index = _clamp(target, 0, speaker.segment_count - 1)
        self._refresh()

    def _move_sample_page(self, delta: int) -> None:
        """Move the selected sample page."""
        speaker = self._speaker()
        page_size = self._sample_page_size()
        current_start = _sample_page_start(speaker.selected_sample_index, page_size)
        last_start = _last_sample_page_start(speaker.segment_count, page_size)
        target_start = _clamp(current_start + delta * page_size, 0, last_start)
        speaker.selected_sample_index = target_start
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
        """Render stable project and workflow state."""
        return render_overview_pane(self.session.speakers, self.session.overview, self._speaker())

    def _speaker_pane(self) -> str:
        """Render the left speaker list."""
        lines = [self._pane_title(tr("Speakers", "Speakers"), "speakers")]
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            style = status_style(speaker_status(speaker))
            label = f"{marker} {status_icon(speaker)} {speaker.label}  {speaker.current_name}  {match_badge(speaker)}"
            lines.append(f"[{style}]{escape(label)}[/]")
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render the selected speaker samples."""
        speaker = self._speaker()
        lines = [self._pane_title(tr(f"{speaker.label} samples", f"{speaker.label} samples"), "samples")]
        lines.append(render_selected_speaker_line(speaker))
        page_start, segments = self._visible_segments(speaker)
        for offset, segment in enumerate(segments):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            time_range = _segment_time_range(segment)
            text = _trim_sample_text(segment.text)
            sample_line = f"{prefix} [cyan]{time_range}[/] {escape(text)}"
            if self._has_correction_edit(segment):
                sample_line += " [yellow]edited[/]"
            if index == speaker.selected_sample_index:
                sample_line = f"[reverse]{sample_line}[/]"
            lines.append(sample_line)
        lines.append("")
        lines.append(self._sample_page_footer(speaker, page_start))
        return "\n".join(lines)

    def _play_sample(self, segment: SentenceSegment) -> None:
        """Start playback for the selected segment."""
        self._stop_playback()
        start_seconds = max(0.0, segment.begin_time_ms / 1000.0 - 0.5)
        duration_seconds = max(2.0, (segment.end_time_ms - segment.begin_time_ms) / 1000.0 + 1.0)
        command = build_audio_preview_command(
            media=self.session.source_media,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        self.playback_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_status(tr(f"Playing selected sample: {_segment_time_range(segment)}.", f"正在播放当前 sample：{_segment_time_range(segment)}。"))

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

    def _mapping(self) -> dict[int, str]:
        """Return the current speaker mapping."""
        return {
            speaker.speaker_id: speaker.current_name.strip() or speaker.label
            for speaker in self.session.speakers
        }

    def _person_mapping(self) -> dict[int, int]:
        """Return the current speaker to voiceprint person id mapping."""
        return {
            speaker.speaker_id: speaker.person_id
            for speaker in self.session.speakers
            if speaker.person_id is not None and not speaker.ignored
        }

    def _person_public_mapping(self) -> dict[int, str]:
        """Return the current speaker to voiceprint person public id mapping."""
        return {
            speaker.speaker_id: speaker.person_public_id
            for speaker in self.session.speakers
            if speaker.person_public_id is not None and not speaker.ignored
        }

    def _decision(self) -> SpeakerReviewDecision:
        """Build the current save decision."""
        correction_edits = tuple(self.correction_edits)
        latest_edit = correction_edits[-1] if correction_edits else None
        action = "correct-inline" if correction_edits else "save"
        return SpeakerReviewDecision(
            saved=True,
            mapping=self._mapping(),
            action=action,
            person_mapping=self._person_mapping(),
            person_public_mapping=self._person_public_mapping(),
            correction_edit=latest_edit,
            correction_edits=correction_edits,
            project_dir=self.session.project_dir,
        )

    def _upsert_correction_edit(self, edit: SentenceCorrectionEdit) -> None:
        """Insert or replace one staged edit by sentence identity."""
        for index, staged in enumerate(self.correction_edits):
            if _same_edit(staged, edit):
                self.correction_edits[index] = edit
                return
        self.correction_edits.append(edit)

    def _has_correction_edit(self, segment: SentenceSegment) -> bool:
        """Return whether a visible sample has a staged correction."""
        return any(_same_sentence(segment, edit) for edit in self.correction_edits)

    def _handle_save_outcome(self, outcome: SpeakerReviewSaveOutcome) -> None:
        """Update review state after the in-TUI save workflow completes."""
        self._mark_speaker_names_saved()
        summary = outcome.correction_summary
        if summary is not None and summary.accepted:
            self.correction_edits.clear()
            self._set_status(
                tr(
                    "Saved names and accepted transcript correction. Press v to capture voiceprints.",
                    "已保存姓名并接受文字修正。按 v 继续声纹采样。",
                )
            )
            self._refresh()
            return
        self._set_status(
            tr(
                "Saved project review. Press v to capture voiceprints, or continue reviewing.",
                "已保存 Project Review。按 v 继续声纹采样，或继续 review。",
            )
        )

    def _mark_speaker_names_saved(self) -> None:
        """Keep in-memory workflow state aligned with the just-written speaker map."""
        overview = replace(self.session.overview, saved_names_by_speaker=self._mapping())
        self.session = replace(self.session, overview=overview)
        self.identity_baseline = _identity_snapshot(self.session.speakers)

    def _save_handler_for_current_project(self) -> Callable[[SpeakerReviewDecision], SpeakerReviewSaveOutcome]:
        """Return a save handler bound to the currently visible project."""
        if self.project_save_handler is not None:
            return lambda decision: self.project_save_handler(self.session.project_dir, decision)
        if self.save_handler is not None:
            return self.save_handler
        raise RuntimeError("Save handler is not configured.")

    def _accept_handler_for_current_project(
        self,
    ) -> Callable[[Path | None, tuple[int, ...] | None], SpeakerReviewSaveOutcome] | None:
        """Return a correction accept handler bound to the current project."""
        if self.project_accept_handler is not None:
            return lambda proposal_path, selected_indices: self.project_accept_handler(
                self.session.project_dir,
                proposal_path,
                selected_indices,
            )
        return self.accept_handler

    def _handle_voiceprint_review_workflow(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """Refresh Project Review after embedded Voiceprint Review completes."""
        overview = replace(
            self.session.overview,
            voiceprint=load_voiceprint_review_progress(self.session.overview.project_id, self.session.store_dir),
            match_file_exists=True,
        )
        self.session = replace(self.session, overview=overview)
        self._set_status(
            tr(
                f"Voiceprint ready: captured {summary.capture.sample_count}, embedded {summary.embedding.embedded_count}; "
                f"historical risks {summary.evaluation.historical_risk_count}.",
                f"声纹已就绪：采集 {summary.capture.sample_count}，embedding 新增 {summary.embedding.embedded_count}；"
                f"历史风险 {summary.evaluation.historical_risk_count}。",
            )
        )
        self._refresh()

    def _handle_voiceprint_embed_state(self, event: Worker.StateChanged) -> None:
        """Update the TUI after a voiceprint embedding worker state change."""
        if event.state == WorkerState.SUCCESS:
            self._handle_voiceprint_embed_success(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self._set_status(
                tr(
                    f"Voiceprint embedding failed: {event.worker.error}. "
                    "Run meeting-asr doctor --require-voiceprint-embedding.",
                    f"声纹 embedding 失败：{event.worker.error}。请运行 meeting-asr doctor --require-voiceprint-embedding。",
                )
            )

    def _handle_speaker_rematch_state(self, event: Worker.StateChanged) -> None:
        """Update Project Review after voiceprint rematch finishes."""
        if event.state == WorkerState.SUCCESS:
            self._handle_speaker_rematch_success(event.worker.result)
            return
        if event.state == WorkerState.ERROR:
            self._dismiss_rematch_processing()
            self._set_status(
                tr(
                    f"Speaker rematch failed: {event.worker.error}. Run meeting-asr doctor --require-voiceprint-embedding.",
                    f"Speaker 重新匹配失败：{event.worker.error}。请运行 meeting-asr doctor --require-voiceprint-embedding。",
                )
            )

    def _handle_speaker_rematch_success(self, result: SpeakerRematchResult) -> None:
        """Replace current review data with the newly matched project state."""
        selected_speaker_id = self._speaker().speaker_id
        self._dismiss_rematch_processing()
        self.session = result.session
        self.known_people = list(result.session.people)
        self.selected_speaker_index = _speaker_index_by_id(result.session.speakers, selected_speaker_id)
        self.focused_column = "speakers"
        self.identity_baseline = _identity_snapshot(result.session.speakers)
        self._enter_browse_mode(compact_rematch_line(result))
        self._refresh()

    def _dismiss_rematch_processing(self) -> None:
        """Close the rematch processing modal if it is currently visible."""
        if isinstance(self.screen, SpeakerRematchProcessingScreen):
            self.screen.dismiss(None)

    def _handle_voiceprint_embed_success(self, summary: VoiceprintEmbedSummary) -> None:
        """Refresh project voiceprint progress after embedding finishes."""
        overview = replace(
            self.session.overview,
            voiceprint=load_voiceprint_review_progress(self.session.overview.project_id, self.session.store_dir),
        )
        self.session = replace(self.session, overview=overview)
        self._set_status(
            tr(
                f"Voiceprint embedding ready: embedded {summary.embedded_count}, skipped {summary.skipped_count}.",
                f"声纹 embedding 已完成：生成 {summary.embedded_count} 个，跳过 {summary.skipped_count} 个。",
            )
        )
        self._refresh()

    def _has_unsaved_speaker_names(self) -> bool:
        """Return whether speaker names differ from the saved project map."""
        saved = self.session.overview.saved_names_by_speaker
        return any((speaker.current_name.strip() or speaker.label) != saved.get(speaker.speaker_id) for speaker in self.session.speakers)

    def _has_unsaved_review_changes(self) -> bool:
        """Return whether switching projects would discard visible TUI edits."""
        return _identity_snapshot(self.session.speakers) != self.identity_baseline or bool(self.correction_edits)

    def _set_speaker_identity(
        self,
        speaker: ReviewSpeaker,
        name: str,
        person_id: int | None,
        person_public_id: str | None,
    ) -> None:
        """Set a speaker display name and optional voiceprint person id."""
        speaker.current_name = name.strip() or speaker.label
        speaker.ignored = speaker.current_name == speaker.label
        speaker.person_id = None if speaker.ignored else person_id
        speaker.person_public_id = None if speaker.ignored else person_public_id

    def _remember_known_person(self, selection: IdentitySelection) -> None:
        """Keep newly-created or match-only people available in later modal opens."""
        if find_person_by_name(selection.name, self.known_people) is None:
            self.known_people.append(KnownPerson(selection.person_id, selection.name, selection.public_id))

    def _speaker(self) -> ReviewSpeaker:
        """Return the selected speaker."""
        return self.session.speakers[self.selected_speaker_index]

    def _selected_sample(self) -> SentenceSegment:
        """Return the selected sample segment."""
        speaker = self._speaker()
        return speaker.segments[speaker.selected_sample_index]

    def _replace_segment_text(self, edit: SentenceCorrectionEdit) -> None:
        """Update the in-memory sample text after a TUI correction."""
        for speaker in self.session.speakers:
            for segment in speaker.segments:
                if _same_sentence(segment, edit):
                    segment.text = edit.corrected_text
                    return

    def _visible_segments(self, speaker: ReviewSpeaker) -> tuple[int, list[SentenceSegment]]:
        """Return the current sample page start and segments."""
        page_size = self._sample_page_size()
        page_start = _sample_page_start(speaker.selected_sample_index, page_size)
        return page_start, speaker.segments[page_start : page_start + page_size]

    def _sample_page_size(self) -> int:
        """Return the number of sample rows that fit in the pane."""
        if self.session.page_size is not None:
            return max(1, self.session.page_size)
        pane_height = self.query_one("#samples", Static).size.height
        if pane_height <= SAMPLE_PANE_RESERVED_ROWS:
            return DEFAULT_SAMPLE_PAGE_SIZE
        return max(1, pane_height - SAMPLE_PANE_RESERVED_ROWS)

    def _sample_page_footer(self, speaker: ReviewSpeaker, page_start: int) -> str:
        """Render pagination status for the sample pane."""
        page_size = self._sample_page_size()
        page_count = _sample_page_count(speaker.segment_count, page_size)
        page_number = page_start // page_size + 1
        start = page_start + 1
        end = min(page_start + page_size, speaker.segment_count)
        return (
            f"Page {page_number}/{page_count}  Samples {start}-{end}/"
            f"{speaker.segment_count}  PageUp/PageDown or bracket keys"
        )

    def _pane_title(self, title: str, column: str) -> str:
        """Render a pane title with focused-column state."""
        escaped = escape(title)
        if self.focused_column == column:
            return f"[reverse][b] FOCUS [/b][/] [b]{escaped}[/b]"
        return f"[dim]{escaped}[/dim]"

    def _enter_browse_mode(self, status: str) -> None:
        """Return keyboard handling to browse mode."""
        self.set_focus(None)
        self._set_status(status)

    def _set_status(self, message: str) -> None:
        """Show a short status message."""
        self.query_one("#status", Static).update(escape(message))


def run_speaker_review_tui(
    session: SpeakerReviewSession,
    *,
    save_handler: Callable[[SpeakerReviewDecision], SpeakerReviewSaveOutcome] | None = None,
    accept_handler: Callable[[Path | None, tuple[int, ...] | None], SpeakerReviewSaveOutcome] | None = None,
    project_save_handler: Callable[[Path, SpeakerReviewDecision], SpeakerReviewSaveOutcome] | None = None,
    project_accept_handler: Callable[
        [Path, Path | None, tuple[int, ...] | None],
        SpeakerReviewSaveOutcome,
    ]
    | None = None,
) -> SpeakerReviewDecision:
    """
    Run the Textual speaker review app.

    Args:
        session: Speaker review inputs.
        save_handler: Optional in-TUI save callback.
        accept_handler: Optional in-TUI correction accept callback.
        project_save_handler: Optional project-aware save callback.
        project_accept_handler: Optional project-aware correction accept callback.

    Returns:
        Save decision and mapping.
    """
    decision = SpeakerReviewApp(
        session,
        save_handler=save_handler,
        accept_handler=accept_handler,
        project_save_handler=project_save_handler,
        project_accept_handler=project_accept_handler,
    ).run()
    return decision or SpeakerReviewDecision(saved=False, mapping={}, action="quit")


def _sample_page_start(selected_index: int, page_size: int) -> int:
    """Return the first sample index for the selected sample's page."""
    return selected_index // page_size * page_size


def _last_sample_page_start(segment_count: int, page_size: int) -> int:
    """Return the first sample index of the last page."""
    return max(0, (segment_count - 1) // page_size * page_size)


def _sample_page_count(segment_count: int, page_size: int) -> int:
    """Return the number of sample pages."""
    return max(1, (segment_count + page_size - 1) // page_size)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer into an inclusive range."""
    return min(max(value, minimum), maximum)


def _segment_time_range(segment: SentenceSegment) -> str:
    """Format a transcript segment time range."""
    start = format_ms_timestamp(segment.begin_time_ms)
    end = format_ms_timestamp(segment.end_time_ms)
    return f"{start}-{end}"


def _same_sentence(segment: SentenceSegment, edit: SentenceCorrectionEdit) -> bool:
    """Return whether a segment matches an inline correction edit."""
    return (
        segment.sentence_id == edit.sentence_id
        and segment.speaker_id == edit.speaker_id
        and segment.begin_time_ms == edit.begin_time_ms
        and segment.end_time_ms == edit.end_time_ms
    )


def _same_edit(left: SentenceCorrectionEdit, right: SentenceCorrectionEdit) -> bool:
    """Return whether two inline edits refer to the same transcript sentence."""
    return (
        left.sentence_id == right.sentence_id
        and left.speaker_id == right.speaker_id
        and left.begin_time_ms == right.begin_time_ms
        and left.end_time_ms == right.end_time_ms
    )


def _known_person_public_id(people: list[KnownPerson], person_id: int | None) -> str | None:
    """Return a known public id for an internal person id."""
    if person_id is None:
        return None
    for person in people:
        if person.person_id == person_id:
            return person.public_id
    return None


def _speaker_index_by_id(speakers: list[ReviewSpeaker], speaker_id: int) -> int:
    """Return the index for a speaker id, falling back to the first row."""
    for index, speaker in enumerate(speakers):
        if speaker.speaker_id == speaker_id:
            return index
    return 0


def _identity_snapshot(speakers: list[ReviewSpeaker]) -> dict[int, tuple[str, bool, int | None, str | None]]:
    """Return the current in-TUI speaker identity state."""
    return {
        speaker.speaker_id: (
            speaker.current_name.strip() or speaker.label,
            speaker.ignored,
            speaker.person_id,
            speaker.person_public_id,
        )
        for speaker in speakers
    }


def _trim_sample_text(text: str, *, limit: int = 90) -> str:
    """Trim a transcript sample for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
