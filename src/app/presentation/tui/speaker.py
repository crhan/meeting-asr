"""Textual UI for reviewing project speaker names."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.worker import Worker, WorkerState
from textual.widgets import Header, Static

from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
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
from app.presentation.tui.speaker_help import (
    ShortcutHelpScreen,
    browse_status,
    edit_status,
    timeline_status,
)
from app.presentation.tui.speaker_identity import IdentityEditScreen, IdentitySelection
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_models import (
    ReviewSpeaker,
    SentenceReassignment,
    SpeakerClusterDiagnostic,
    SpeakerClusterSampleScore,
    SpeakerReviewDecision,
    SpeakerReviewSession,
)
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
from app.presentation.tui.speaker_save import (
    SpeakerReviewSaveOutcome,
    SpeakerReviewSaveScreen,
    sentence_reassignment_changes,
    speaker_ignore_changes,
    speaker_name_changes,
)
from app.presentation.tui.speaker_session import load_speaker_review_session, load_voiceprint_review_progress
from app.presentation.tui.speaker_timeline import (
    SpeakerPickOption,
    SpeakerPickScreen,
    TimelineRow,
    build_timeline_rows,
    capture_speaker_baseline,
    move_segment_between_speakers,
    render_timeline_pane,
    segment_key,
    speaker_by_id,
    speaker_pick_options,
)
from app.presentation.tui.speaker_status import (
    SpeakerReviewOverview,
    VoiceprintReviewProgress,
    is_ignored,
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
TIMELINE_PANE_RESERVED_ROWS = 4
DEFAULT_TIMELINE_PAGE_SIZE = 16
COLUMNS = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"

VIEW_MODE_SPEAKERS = "speakers"
VIEW_MODE_TIMELINE = "timeline"


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Current sample playback state for visible Project Review progress."""

    key: tuple[int | None, int, int]
    label: str
    text: str
    started_at: float
    duration_seconds: float | None


__all__ = [
    "FOCUSED_PANE_CLASS",
    "IdentityEditScreen",
    "KnownPerson",
    "ReviewSpeaker",
    "SentenceCorrectionScreen",
    "SentenceReassignment",
    "ShortcutHelpScreen",
    "SpeakerMatchCandidate",
    "SpeakerPickScreen",
    "SpeakerRematchProcessingScreen",
    "SpeakerRematchResult",
    "SpeakerReviewApp",
    "SpeakerReviewDecision",
    "SpeakerReviewOverview",
    "SpeakerReviewSession",
    "TimelineRow",
    "UNFOCUSED_PANE_CLASS",
    "VIEW_MODE_SPEAKERS",
    "VIEW_MODE_TIMELINE",
    "VoiceprintReviewProgress",
    "build_timeline_rows",
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
    #timeline-main {
        height: 1fr;
        display: none;
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
    #timeline {
        width: 100%;
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
        Binding("t", "toggle_view", "Toggle view"),
        Binding("r", "reassign_speaker", "Reassign sentence"),
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
        self.playback_state: PlaybackState | None = None
        self.playback_timer: Any | None = None
        self.known_people = list(session.people)
        self.correction_edits: list[SentenceCorrectionEdit] = []
        self.identity_baseline = _identity_snapshot(session.speakers)
        self.view_mode = VIEW_MODE_SPEAKERS
        self.timeline_selected_index = 0
        self.original_speaker_by_segment: dict[tuple[int | None, int, int], int] = (
            capture_speaker_baseline(session.speakers)
        )

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        with Horizontal(id="timeline-main"):
            yield Static(id="timeline", classes="pane")
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
        """Move down in the focused column or timeline."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            self._move_timeline(1)
            return
        self._move_focused_row(1)

    def action_up(self) -> None:
        """Move up in the focused column or timeline."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            self._move_timeline(-1)
            return
        self._move_focused_row(-1)

    def action_left(self) -> None:
        """Focus the previous column (no-op in timeline view)."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            return
        self._move_column(-1)

    def action_right(self) -> None:
        """Focus the next column (no-op in timeline view)."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            return
        self._move_column(1)

    def action_next_sample_page(self) -> None:
        """Select the first row on the next page."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            self._move_timeline_page(1)
            return
        self._move_sample_page(1)

    def action_previous_sample_page(self) -> None:
        """Select the first row on the previous page."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            self._move_timeline_page(-1)
            return
        self._move_sample_page(-1)

    def action_play_sample(self) -> None:
        """Play or stop the selected sample as audio."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止 sample 播放。"))
            self._refresh()
            return
        try:
            self._play_sample(self._active_segment())
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
                ignore_changes=speaker_ignore_changes(
                    self.session.speakers,
                    self.session.overview.saved_ignored_speaker_ids,
                ),
                reassignment_changes=sentence_reassignment_changes(
                    self.session.speakers,
                    decision.sentence_reassignments,
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
        segment = self._active_segment()
        speaker = self._speaker_for_segment(segment)
        self.push_screen(
            SentenceCorrectionScreen(
                speaker_label=speaker.label,
                speaker_name=speaker.current_name,
                segment=segment,
            ),
            self._handle_sentence_correction,
        )

    def action_toggle_view(self) -> None:
        """Switch between speaker-grouped and chronological views."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            self.view_mode = VIEW_MODE_SPEAKERS
            self._set_status(browse_status())
        else:
            self.view_mode = VIEW_MODE_TIMELINE
            self._sync_timeline_to_selection()
            self._set_status(timeline_status())
        self._apply_view_mode()
        self._refresh()

    def action_reassign_speaker(self) -> None:
        """Reassign the selected sentence to a different or new speaker."""
        segment = self._reassignment_segment()
        if segment is None:
            self._set_status(tr("No sentences to reassign.", "没有可改的句子。"))
            return
        source = self._speaker_for_segment(segment)
        options = _speaker_pick_options_with_new(self.session.speakers)
        self.push_screen(
            SpeakerPickScreen(
                sentence_text=segment.text,
                current_speaker_id=source.speaker_id,
                options=options,
            ),
            lambda choice: self._handle_speaker_reassignment(segment, source.speaker_id, choice),
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
        self.original_speaker_by_segment = capture_speaker_baseline(session.speakers)
        self.view_mode = VIEW_MODE_SPEAKERS
        self.timeline_selected_index = 0
        self._apply_view_mode()
        self._enter_browse_mode(tr(f"Switched to project {session.overview.project_id}.", f"已切换到项目 {session.overview.project_id}。"))
        self._refresh()

    def _handle_speaker_reassignment(
        self,
        segment: SentenceSegment,
        source_speaker_id: int,
        new_speaker_id: int | None,
    ) -> None:
        """Move a sentence to another speaker after the picker resolves."""
        if new_speaker_id is None:
            self._set_status(tr("Speaker reassignment canceled.", "已取消 speaker 重新指派。"))
            return
        if new_speaker_id == source_speaker_id:
            self._set_status(tr("Same speaker chosen; no change.", "选择的还是当前 speaker，不做改动。"))
            return
        source = speaker_by_id(self.session.speakers, source_speaker_id)
        target = speaker_by_id(self.session.speakers, new_speaker_id)
        if source is None:
            self._set_status(tr("Reassignment target unavailable.", "无法找到重新指派的目标 speaker。"))
            return
        if target is None:
            target = _new_review_speaker(new_speaker_id)
            self.session.speakers.append(target)
        if not move_segment_between_speakers(source, target, segment):
            self._set_status(tr("Sentence not found on its current speaker.", "未在当前 speaker 中找到该句子。"))
            return
        target.ignored = False
        if not target.current_name.strip():
            target.current_name = target.label
        self._retarget_correction_edits(segment, source_speaker_id, target.speaker_id)
        self._sync_timeline_to_segment(segment)
        self.selected_speaker_index = self.session.speakers.index(target)
        self._set_status(
            tr(
                f"Moved sentence {_segment_time_range(segment)} to {target.label} {target.current_name}.",
                f"已把 {_segment_time_range(segment)} 的句子改到 {target.label} {target.current_name}。",
            )
        )
        self._refresh()

    def _retarget_correction_edits(
        self,
        segment: SentenceSegment,
        previous_speaker_id: int,
        new_speaker_id: int,
    ) -> None:
        """Realign staged transcript edits to the segment's new speaker_id."""
        if previous_speaker_id == new_speaker_id:
            return
        for index, edit in enumerate(self.correction_edits):
            if (
                edit.sentence_id == segment.sentence_id
                and edit.begin_time_ms == segment.begin_time_ms
                and edit.end_time_ms == segment.end_time_ms
                and edit.speaker_id == previous_speaker_id
            ):
                self.correction_edits[index] = replace(edit, speaker_id=new_speaker_id)

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
        if self.view_mode == VIEW_MODE_TIMELINE:
            self.query_one("#timeline", Static).update(self._timeline_pane())
            return
        self.query_one("#speakers", Static).update(self._speaker_pane())
        self.query_one("#samples", Static).update(self._sample_pane())

    def _refresh_focus_styles(self) -> None:
        """Make the focused pane visually obvious."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            timeline_pane = self.query_one("#timeline", Static)
            timeline_pane.set_class(True, FOCUSED_PANE_CLASS)
            timeline_pane.set_class(False, UNFOCUSED_PANE_CLASS)
            for column in COLUMNS:
                pane = self.query_one(f"#{column}", Static)
                pane.set_class(False, FOCUSED_PANE_CLASS)
                pane.set_class(True, UNFOCUSED_PANE_CLASS)
            return
        timeline_pane = self.query_one("#timeline", Static)
        timeline_pane.set_class(False, FOCUSED_PANE_CLASS)
        timeline_pane.set_class(True, UNFOCUSED_PANE_CLASS)
        for column in COLUMNS:
            pane = self.query_one(f"#{column}", Static)
            focused = column == self.focused_column
            pane.set_class(focused, FOCUSED_PANE_CLASS)
            pane.set_class(not focused, UNFOCUSED_PANE_CLASS)

    def _apply_view_mode(self) -> None:
        """Show or hide top-level panes based on the current view mode."""
        try:
            main = self.query_one("#main")
            timeline_main = self.query_one("#timeline-main")
        except Exception:  # noqa: BLE001
            return
        timeline_active = self.view_mode == VIEW_MODE_TIMELINE
        main.display = not timeline_active
        timeline_main.display = timeline_active

    def _timeline_pane(self) -> str:
        """Render the chronological transcript view."""
        rows = self._timeline_rows()
        edited_keys = {
            (edit.sentence_id, edit.begin_time_ms, edit.end_time_ms)
            for edit in self.correction_edits
        }
        reassigned_keys = self._reassigned_segment_keys()
        page_size = self._timeline_page_size()
        page_start = _sample_page_start(self.timeline_selected_index, page_size)
        return render_timeline_pane(
            rows,
            selected_index=self.timeline_selected_index,
            page_start=page_start,
            page_size=page_size,
            speaker_count=len(self.session.speakers),
            edited_keys=edited_keys,
            reassigned_keys=reassigned_keys,
        )

    def _timeline_rows(self) -> list[TimelineRow]:
        """Return the current chronological row list (recomputed on demand)."""
        return build_timeline_rows(self.session.speakers)

    def _move_timeline(self, delta: int) -> None:
        """Move the timeline cursor by ``delta`` rows."""
        rows = self._timeline_rows()
        if not rows:
            return
        target = self.timeline_selected_index + delta
        self.timeline_selected_index = _clamp(target, 0, len(rows) - 1)
        self._refresh()

    def _move_timeline_page(self, delta: int) -> None:
        """Move the timeline cursor by one full page."""
        rows = self._timeline_rows()
        if not rows:
            return
        page_size = self._timeline_page_size()
        current_start = _sample_page_start(self.timeline_selected_index, page_size)
        last_start = _last_sample_page_start(len(rows), page_size)
        target_start = _clamp(current_start + delta * page_size, 0, last_start)
        self.timeline_selected_index = target_start
        self._refresh()

    def _timeline_page_size(self) -> int:
        """Return the number of timeline rows that fit in the pane."""
        if self.session.page_size is not None:
            return max(1, self.session.page_size)
        try:
            pane_height = self.query_one("#timeline", Static).size.height
        except Exception:  # noqa: BLE001
            pane_height = 0
        if pane_height <= TIMELINE_PANE_RESERVED_ROWS:
            return DEFAULT_TIMELINE_PAGE_SIZE
        return max(1, pane_height - TIMELINE_PANE_RESERVED_ROWS)

    def _sync_timeline_to_selection(self) -> None:
        """Move the timeline cursor to the currently selected speaker sample."""
        rows = self._timeline_rows()
        if not rows:
            self.timeline_selected_index = 0
            return
        speaker = self._speaker()
        if speaker.segments:
            sample_index = _clamp(speaker.selected_sample_index, 0, speaker.segment_count - 1)
            target = speaker.segments[sample_index]
            self._sync_timeline_to_segment(target, rows=rows)
            return
        self.timeline_selected_index = 0

    def _sync_timeline_to_segment(
        self,
        segment: SentenceSegment,
        *,
        rows: list[TimelineRow] | None = None,
    ) -> None:
        """Move the timeline cursor to a specific segment."""
        rows = rows if rows is not None else self._timeline_rows()
        target_key = segment_key(segment)
        for index, row in enumerate(rows):
            if segment_key(row.segment) == target_key:
                self.timeline_selected_index = index
                return
        self.timeline_selected_index = 0

    def _active_segment(self) -> SentenceSegment:
        """Return the currently selected sentence for either view mode."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            rows = self._timeline_rows()
            if not rows:
                raise RuntimeError("Timeline has no sentences to operate on.")
            self.timeline_selected_index = max(0, min(self.timeline_selected_index, len(rows) - 1))
            return rows[self.timeline_selected_index].segment
        return self._selected_sample()

    def _reassignment_segment(self) -> SentenceSegment | None:
        """Return the sentence currently targeted for speaker reassignment."""
        if self.view_mode == VIEW_MODE_TIMELINE:
            rows = self._timeline_rows()
            if not rows:
                return None
            self.timeline_selected_index = max(0, min(self.timeline_selected_index, len(rows) - 1))
            return rows[self.timeline_selected_index].segment
        speaker = self._speaker()
        if not speaker.segments:
            return None
        return self._selected_sample()

    def _speaker_for_segment(self, segment: SentenceSegment) -> ReviewSpeaker:
        """Return the review speaker that currently owns a segment."""
        for speaker in self.session.speakers:
            for owned in speaker.segments:
                if owned is segment or segment_key(owned) == segment_key(segment):
                    return speaker
        return self._speaker()

    def _reassigned_segment_keys(self) -> set[tuple[int | None, int, int]]:
        """Return segment keys whose speaker_id changed since load."""
        keys: set[tuple[int | None, int, int]] = set()
        for speaker in self.session.speakers:
            for seg in speaker.segments:
                key = segment_key(seg)
                original = self.original_speaker_by_segment.get(key)
                current = seg.speaker_id if seg.speaker_id is not None else speaker.speaker_id
                if original is not None and original != current:
                    keys.add(key)
        return keys

    def _overview_pane(self) -> str:
        """Render stable project and workflow state."""
        return render_overview_pane(self.session.speakers, self.session.overview, self._speaker())

    def _speaker_pane(self) -> str:
        """Render the left speaker list."""
        lines = [self._pane_title(tr("Speakers", "Speakers"), "speakers")]
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            style = status_style(speaker_status(speaker))
            identity = f"{marker} {status_icon(speaker)} {speaker.label}  {speaker.current_name}"
            line = (
                f"[{style}]{escape(identity)}[/]  "
                f"{escape(match_badge(speaker))}  "
                f"{_cluster_badge(self._cluster_diagnostic(speaker))}"
            )
            lines.append(line)
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render the selected speaker samples."""
        speaker = self._speaker()
        lines = [self._pane_title(tr(f"{speaker.label} samples", f"{speaker.label} samples"), "samples")]
        lines.append(
            render_selected_speaker_line(speaker)
            + " | "
            + _cluster_detail(self._cluster_diagnostic(speaker))
        )
        page_start, segments = self._visible_segments(speaker)
        for offset, segment in enumerate(segments):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            time_range = _segment_time_range(segment)
            text = _trim_sample_text(segment.text)
            playing = "[bold magenta]PLAY[/]" if self._playing_segment_key() == segment_key(segment) else "[dim]    [/]"
            sample_line = (
                f"{prefix} {playing} [cyan]{time_range}[/] "
                f"{_sample_cluster_badge(self._cluster_sample_score(speaker, segment))} "
                f"{escape(text)}"
            )
            if self._has_correction_edit(segment):
                sample_line += " [yellow]edited[/]"
            if segment_key(segment) in self._reassigned_segment_keys():
                sample_line += " [yellow]reassigned[/]"
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
        self.playback_state = PlaybackState(
            segment_key(segment),
            _segment_time_range(segment),
            _trim_sample_text(segment.text, limit=80),
            time.monotonic(),
            duration_seconds,
        )
        self._restart_playback_timer()
        self._refresh_playback_status()
        self._refresh()

    def _restart_playback_timer(self) -> None:
        """Restart the UI timer used to update playback progress."""
        self._stop_playback_timer()
        self.playback_timer = self.set_interval(0.5, self._refresh_playback_status)

    def _stop_playback_timer(self) -> None:
        """Stop the playback progress timer if it exists."""
        timer = self.playback_timer
        self.playback_timer = None
        if timer is not None:
            timer.stop()

    def _refresh_playback_status(self) -> None:
        """Refresh playback progress in the status bar and sample pane."""
        state = self.playback_state
        if state is None:
            return
        if not self._is_playing():
            label = state.label
            self.playback_state = None
            self._stop_playback_timer()
            self._set_status(tr(f"Finished playing sample {label}.", f"已播放完成：{label}。"))
            self._refresh()
            return
        self._set_status(_playback_status_text(state))

    def _stop_playback(self) -> None:
        """Stop the current playback child process if it is still running."""
        self._stop_playback_timer()
        self.playback_state = None
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

    def _playing_segment_key(self) -> tuple[int | None, int, int] | None:
        """Return the currently playing segment key, if playback is active."""
        state = self.playback_state
        if state is None or not self._is_playing():
            return None
        return state.key

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

    def _ignored_speaker_ids(self) -> tuple[int, ...]:
        """Return project speaker ids deliberately kept anonymous."""
        return tuple(sorted(speaker.speaker_id for speaker in self.session.speakers if is_ignored(speaker)))

    def _decision(self) -> SpeakerReviewDecision:
        """Build the current save decision."""
        correction_edits = tuple(self.correction_edits)
        latest_edit = correction_edits[-1] if correction_edits else None
        reassignments = self._sentence_reassignments()
        action = "correct-inline" if correction_edits else "save"
        return SpeakerReviewDecision(
            saved=True,
            mapping=self._mapping(),
            action=action,
            person_mapping=self._person_mapping(),
            person_public_mapping=self._person_public_mapping(),
            ignored_speaker_ids=self._ignored_speaker_ids(),
            correction_edit=latest_edit,
            correction_edits=correction_edits,
            sentence_reassignments=reassignments,
            project_dir=self.session.project_dir,
        )

    def _sentence_reassignments(self) -> tuple[SentenceReassignment, ...]:
        """Build sentence speaker reassignments since the last load."""
        seen: set[tuple[int | None, int, int]] = set()
        out: list[SentenceReassignment] = []
        for speaker in self.session.speakers:
            for seg in speaker.segments:
                key = segment_key(seg)
                if key in seen:
                    continue
                original = self.original_speaker_by_segment.get(key)
                current = seg.speaker_id if seg.speaker_id is not None else speaker.speaker_id
                if original is None or original == current:
                    continue
                out.append(
                    SentenceReassignment(
                        sentence_id=seg.sentence_id,
                        begin_time_ms=seg.begin_time_ms,
                        end_time_ms=seg.end_time_ms,
                        original_speaker_id=original,
                        new_speaker_id=current,
                    )
                )
                seen.add(key)
        return tuple(out)

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
        reassignment_note = _summarize_reassignment_outcome(outcome.reassignment_result)
        summary = outcome.correction_summary
        if summary is not None and summary.accepted:
            self.correction_edits.clear()
            base = tr(
                "Saved names and accepted transcript correction.",
                "已保存姓名并接受文字修正。",
            )
            tail = tr(" Press v to capture voiceprints.", " 按 v 继续声纹采样。")
            self._set_status(base + reassignment_note + tail)
            self._refresh()
            return
        base = tr(
            "Saved project review.",
            "已保存 Project Review。",
        )
        tail = tr(
            " Press v to capture voiceprints, or continue reviewing.",
            " 按 v 继续声纹采样，或继续 review。",
        )
        self._set_status(base + reassignment_note + tail)

    def _mark_speaker_names_saved(self) -> None:
        """Keep in-memory workflow state aligned with the just-written speaker map."""
        overview = replace(
            self.session.overview,
            saved_names_by_speaker=self._mapping(),
            saved_ignored_speaker_ids=frozenset(self._ignored_speaker_ids()),
        )
        self.session = replace(self.session, overview=overview)
        self.identity_baseline = _identity_snapshot(self.session.speakers)
        self.original_speaker_by_segment = capture_speaker_baseline(self.session.speakers)

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
        self.original_speaker_by_segment = capture_speaker_baseline(result.session.speakers)
        self.view_mode = VIEW_MODE_SPEAKERS
        self.timeline_selected_index = 0
        self._apply_view_mode()
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
        """Return whether speaker names or ignore state differ from saved project state."""
        saved = self.session.overview.saved_names_by_speaker
        ignored = self.session.overview.saved_ignored_speaker_ids
        for speaker in self.session.speakers:
            current_ignored = is_ignored(speaker)
            if current_ignored != (speaker.speaker_id in ignored):
                return True
            if current_ignored:
                continue
            if (speaker.current_name.strip() or speaker.label) != saved.get(speaker.speaker_id):
                return True
        return False

    def _has_unsaved_review_changes(self) -> bool:
        """Return whether switching projects would discard visible TUI edits."""
        if _identity_snapshot(self.session.speakers) != self.identity_baseline:
            return True
        if self.correction_edits:
            return True
        return bool(self._reassigned_segment_keys())

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

    def _cluster_diagnostic(self, speaker: ReviewSpeaker) -> SpeakerClusterDiagnostic | None:
        """Return optional cluster diagnostics for one speaker."""
        return self.session.cluster_diagnostics.get(speaker.speaker_id)

    def _cluster_sample_score(
        self,
        speaker: ReviewSpeaker,
        segment: SentenceSegment,
    ) -> SpeakerClusterSampleScore | None:
        """Return optional sample-to-cluster score for one visible row."""
        diagnostic = self._cluster_diagnostic(speaker)
        if diagnostic is None:
            return None
        return diagnostic.samples.get(segment_key(segment))

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


def _cluster_badge(diagnostic: SpeakerClusterDiagnostic | None) -> str:
    """Render one compact speaker cluster badge."""
    if diagnostic is None:
        return "[dim]cluster=-[/]"
    score = _format_optional_score(diagnostic.centroid_mean)
    style = _cluster_status_style(diagnostic.status)
    return f"[{style}]cluster={score} {escape(diagnostic.status)}[/]"


def _cluster_detail(diagnostic: SpeakerClusterDiagnostic | None) -> str:
    """Render selected speaker cluster detail."""
    if diagnostic is None:
        return "[dim]cluster report=-[/]"
    mean = _format_optional_score(diagnostic.centroid_mean)
    minimum = _format_optional_score(diagnostic.centroid_min)
    style = _cluster_status_style(diagnostic.status)
    counts = f"{diagnostic.clip_count}/{diagnostic.segment_count}"
    components = ",".join(str(value) for value in diagnostic.component_sizes) or "-"
    return (
        f"[{style}]cluster {escape(diagnostic.status)}[/] "
        f"mean={mean} min={minimum} clips={counts} "
        f"components={diagnostic.component_count} sizes={components}"
    )


def _sample_cluster_badge(score: SpeakerClusterSampleScore | None) -> str:
    """Render one sample-to-centroid score badge."""
    if score is None:
        return "[dim]cluster=-[/]"
    value = _format_optional_score(score.score)
    style = _sample_score_style(score.status)
    return f"[{style}]cluster={value} {escape(score.status)}[/]"


def _cluster_status_style(status: str) -> str:
    """Return Rich style for a speaker cluster status."""
    return {
        "ok": "green",
        "insufficient": "yellow",
        "warning": "yellow",
        "mixed": "bold red",
    }.get(status, "dim")


def _sample_score_style(status: str) -> str:
    """Return Rich style for one sample cluster status."""
    return {"ok": "green", "warning": "yellow", "critical": "bold red", "low-info": "dim"}.get(status, "dim")


def _format_optional_score(value: float | None) -> str:
    """Format an optional diagnostic score."""
    return "-" if value is None else f"{value:.3f}"


def _playback_status_text(state: PlaybackState) -> str:
    """Render a human-readable playback progress line."""
    elapsed = max(0.0, time.monotonic() - state.started_at)
    duration = state.duration_seconds if state.duration_seconds and state.duration_seconds > 0 else None
    text = _trim_sample_text(state.text, limit=56)
    if duration is None:
        return tr(
            f"Playing {state.label} | elapsed {_format_duration(elapsed)} | {text} | Space stops playback.",
            f"正在播放 {state.label} | 已播放 {_format_duration(elapsed)} | {text} | Space 停止播放。",
        )
    progress = min(1.0, elapsed / duration)
    return tr(
        f"Playing {state.label} | {_format_duration(elapsed)}/{_format_duration(duration)} | {progress:.0%} | {text} | Space stops playback.",
        f"正在播放 {state.label} | {_format_duration(elapsed)}/{_format_duration(duration)} | {progress:.0%} | {text} | Space 停止播放。",
    )


def _format_duration(seconds: float) -> str:
    """Format seconds as compact mm:ss progress."""
    total = max(0, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    return f"{minutes:02d}:{remainder:02d}"


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


def _speaker_pick_options_with_new(speakers: list[ReviewSpeaker]) -> list[SpeakerPickOption]:
    """Return existing speaker picker options plus one anonymous new speaker."""
    options = speaker_pick_options(speakers)
    new_speaker_id = _next_review_speaker_id(speakers)
    label = speaker_id_to_label(new_speaker_id)
    options.append(
        SpeakerPickOption(
            speaker_id=new_speaker_id,
            label=label,
            name=tr("new speaker", "新 speaker"),
            ignored=False,
        )
    )
    return options


def _next_review_speaker_id(speakers: list[ReviewSpeaker]) -> int:
    """Return the next unused project speaker id."""
    if not speakers:
        return 0
    return max(speaker.speaker_id for speaker in speakers) + 1


def _new_review_speaker(speaker_id: int) -> ReviewSpeaker:
    """Create an empty anonymous Project Review speaker."""
    label = speaker_id_to_label(speaker_id)
    return ReviewSpeaker(
        speaker_id=speaker_id,
        label=label,
        segments=[],
        current_name=label,
        match=None,
    )


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


def _summarize_reassignment_outcome(result: object | None) -> str:
    """Return a short status fragment describing the reassignment pipeline.

    Args:
        result: ``SentenceReassignmentApplyResult`` or ``None``.

    Returns:
        Localized status fragment to append to the save status line. Empty
        when no reassignments ran.
    """
    if result is None:
        return ""
    deleted = len(getattr(result, "deleted_samples", ()) or ())
    rematched = getattr(result, "match_summary", None) is not None
    rematch_reason = getattr(result, "rematch_skipped_reason", None)
    pieces: list[str] = []
    pieces.append(tr(" Reassignments applied", " 已应用归属变更"))
    if deleted:
        pieces.append(
            tr(f", {deleted} voiceprint sample(s) invalidated", f"，失效 {deleted} 条声纹样本")
        )
    if rematched:
        pieces.append(tr(", voiceprint matches refreshed", "，声纹匹配已刷新"))
    elif rematch_reason:
        pieces.append(tr(f", rematch skipped: {rematch_reason}", f"，重新匹配被跳过：{rematch_reason}"))
    pieces.append(".")
    return "".join(pieces)


def _trim_sample_text(text: str, *, limit: int = 90) -> str:
    """Trim a transcript sample for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
