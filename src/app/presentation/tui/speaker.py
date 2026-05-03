"""Textual UI for reviewing project speaker names."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static

from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import ProjectManifest, load_manifest, project_paths, resolve_project_source_path
from app.speaker_labeling import load_transcript_result
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    best_candidate_name,
    voiceprint_match_status,
)
from app.speaker_review import build_audio_preview_command
from app.presentation.tui.speaker_correction import (
    CorrectionQueuedScreen,
    SentenceCorrectionEdit,
    SentenceCorrectionScreen,
)
from app.presentation.tui.speaker_help import BROWSE_STATUS, EDIT_STATUS, ShortcutHelpScreen
from app.presentation.tui.speaker_identity import IdentityEditScreen, IdentitySelection
from app.presentation.tui.speaker_matches import (
    SpeakerMatchCandidate,
    accepted_review_name,
    accepted_review_person_id,
    load_match_candidates,
)
from app.presentation.tui.speaker_people import (
    KnownPerson,
    find_person_by_name,
    load_existing_person_mapping,
    load_people,
)
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
from app.utils import format_ms_timestamp
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_embedded_sample_ids,
    list_voiceprint_samples_for_project,
)

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 5
COLUMNS = ("speakers", "samples")
FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"

@dataclass(slots=True)
class ReviewSpeaker:
    """Mutable review state for one project speaker."""

    speaker_id: int
    label: str
    segments: list[SentenceSegment]
    current_name: str
    match: SpeakerMatchCandidate | None
    selected_sample_index: int = 0
    ignored: bool = False
    person_id: int | None = None

    @property
    def segment_count(self) -> int:
        """Return the total number of transcript segments for this speaker."""
        return len(self.segments)


@dataclass(frozen=True, slots=True)
class SpeakerReviewSession:
    """Inputs needed by the speaker review TUI."""

    project_dir: Path
    source_media: Path
    overview: SpeakerReviewOverview
    speakers: list[ReviewSpeaker]
    people_names: list[str]
    page_size: int | None = None
    allow_correction: bool = False
    people: tuple[KnownPerson, ...] = ()
    store_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class SpeakerReviewDecision:
    """Result returned by the TUI when it exits."""

    saved: bool
    mapping: dict[int, str]
    action: str = "save"
    person_mapping: dict[int, int] = field(default_factory=dict)
    correction_edit: SentenceCorrectionEdit | None = None


class SpeakerReviewApp(App[SpeakerReviewDecision]):
    """Keyboard-first TUI for reviewing project speaker identities."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #overview {
        border: round $accent;
        height: 8;
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
        Binding("?", "show_shortcuts", "Help"),
        Binding("s", "save", "Save"),
        Binding("q", "quit_review", "Quit"),
    ]

    def __init__(self, session: SpeakerReviewSession) -> None:
        """
        Create the TUI app.

        Args:
            session: Speaker review inputs.
        """
        super().__init__()
        self.session = session
        self.selected_speaker_index = 0
        self.focused_column = "speakers"
        self.playback_process: subprocess.Popen | None = None
        self.known_people = list(session.people)
        self.correction_edit: SentenceCorrectionEdit | None = None

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(BROWSE_STATUS, id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Render the initial review state."""
        self._enter_browse_mode(BROWSE_STATUS)
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
        """Play or stop the selected sample as audio."""
        if self._is_playing():
            self._stop_playback()
            self._set_status("Stopped sample playback.")
            return
        try:
            self._play_sample(self._selected_sample())
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Preview failed: {exc}")

    def action_edit_name(self) -> None:
        """Open the explicit known-person selection modal."""
        speaker = self._speaker()
        self._set_status(EDIT_STATUS)
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
            self._set_status("No usable match for this speaker.")
            return
        person_id = None if speaker.match is None else speaker.match.best_person_id
        if person_id is None:
            person = find_person_by_name(candidate, self.known_people)
            person_id = None if person is None else person.person_id
        self._set_speaker_identity(speaker, candidate, person_id)
        self._set_status(f"Accepted match for {speaker.label}: {candidate}.")
        self._refresh()

    def action_ignore_speaker(self) -> None:
        """Keep the anonymous speaker label so voiceprint capture skips it."""
        speaker = self._speaker()
        speaker.current_name = speaker.label
        speaker.ignored = True
        speaker.person_id = None
        self._set_status(f"Ignored {speaker.label}; it will stay anonymous and be skipped by capture.")
        self._refresh()

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.push_screen(ShortcutHelpScreen())

    def action_save(self) -> None:
        """Return the reviewed mapping to the CLI command."""
        action = "correct-inline" if self.correction_edit is not None else "save"
        self.exit(
            SpeakerReviewDecision(
                saved=True,
                mapping=self._mapping(),
                action=action,
                person_mapping=self._person_mapping(),
                correction_edit=self.correction_edit,
            )
        )

    def action_edit_sample_text(self) -> None:
        """Edit the selected transcript sentence inside the TUI."""
        if not self.session.allow_correction:
            self._set_status("Transcript correction is available from project review, not speaker-only review.")
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
            self._set_status("Transcript correction canceled.")
            return
        self.correction_edit = edit
        self._replace_segment_text(edit)
        self._set_status("Text correction staged. Press s to save names and run full-document correction.")
        self._refresh()
        self.push_screen(CorrectionQueuedScreen(edit))

    def _handle_identity_selection(self, selection: IdentitySelection | None) -> None:
        """Apply one identity selected in the modal."""
        if selection is None:
            self._set_status("Identity edit canceled.")
            return
        speaker = self._speaker()
        self._remember_known_person(selection)
        self._set_speaker_identity(speaker, selection.name, selection.person_id)
        action = "Created" if selection.created else "Set"
        self._set_status(f"{action} {speaker.label} to {selection.name} #{selection.person_id}.")
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
        self._enter_browse_mode(BROWSE_STATUS)
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
        lines = [self._pane_title("Speakers", "speakers")]
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            style = status_style(speaker_status(speaker))
            label = f"{marker} {status_icon(speaker)} {speaker.label}  {speaker.current_name}  {match_badge(speaker)}"
            lines.append(f"[{style}]{escape(label)}[/]")
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render the selected speaker samples."""
        speaker = self._speaker()
        lines = [self._pane_title(f"{speaker.label} samples", "samples")]
        lines.append(render_selected_speaker_line(speaker))
        page_start, segments = self._visible_segments(speaker)
        for offset, segment in enumerate(segments):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            time_range = _segment_time_range(segment)
            text = _trim_sample_text(segment.text)
            sample_line = f"{prefix} [cyan]{time_range}[/] {escape(text)}"
            if self.correction_edit is not None and _same_sentence(segment, self.correction_edit):
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
        self._set_status(f"Playing selected sample: {_segment_time_range(segment)}.")

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

    def _set_speaker_identity(
        self,
        speaker: ReviewSpeaker,
        name: str,
        person_id: int | None,
    ) -> None:
        """Set a speaker display name and optional voiceprint person id."""
        speaker.current_name = name.strip() or speaker.label
        speaker.ignored = speaker.current_name == speaker.label
        speaker.person_id = None if speaker.ignored else person_id

    def _remember_known_person(self, selection: IdentitySelection) -> None:
        """Keep newly-created or match-only people available in later modal opens."""
        if find_person_by_name(selection.name, self.known_people) is None:
            self.known_people.append(KnownPerson(selection.person_id, selection.name))

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


def load_speaker_review_session(
    project_dir: Path,
    *,
    page_size: int | None = None,
    store_dir: Path | None = None,
    allow_correction: bool = False,
) -> SpeakerReviewSession:
    """
    Load all data needed by the speaker review TUI.

    Args:
        project_dir: Project root.
        page_size: Optional samples-per-page override.
        store_dir: Optional voiceprint store directory.
        allow_correction: Whether the TUI may launch transcript correction.

    Returns:
        Speaker review session.
    """
    paths = project_paths(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    segments_by_speaker = _segments_by_speaker(result.sentences)
    if not segments_by_speaker:
        raise RuntimeError("No detected speakers found in the transcript.")
    manifest = load_manifest(paths.root)
    source_media = resolve_project_source_path(paths.root, manifest)
    mapping_path = paths.speakers_dir / "speaker_map.json"
    match_path = paths.speakers_dir / "speaker_matches.json"
    mapping = _load_existing_mapping(mapping_path)
    matches = load_match_candidates(match_path)
    people = load_people(store_dir)
    person_mapping = load_existing_person_mapping(paths.speakers_dir / "speaker_person_map.json")
    speakers = _build_review_speakers(segments_by_speaker, mapping, person_mapping, matches)
    return SpeakerReviewSession(
        project_dir=paths.root,
        source_media=source_media,
        overview=_build_review_overview(
            manifest=manifest,
            source_media=source_media,
            sentences=result.sentences,
            match_file_exists=match_path.exists(),
            saved_names_by_speaker=mapping,
            store_dir=store_dir,
        ),
        speakers=speakers,
        people_names=[person.name for person in people],
        page_size=page_size,
        allow_correction=allow_correction,
        people=tuple(people),
        store_dir=store_dir,
    )


def run_speaker_review_tui(session: SpeakerReviewSession) -> SpeakerReviewDecision:
    """
    Run the Textual speaker review app.

    Args:
        session: Speaker review inputs.

    Returns:
        Save decision and mapping.
    """
    decision = SpeakerReviewApp(session).run()
    return decision or SpeakerReviewDecision(saved=False, mapping={}, action="quit")


def _build_review_overview(
    *,
    manifest: ProjectManifest,
    source_media: Path,
    sentences: list[SentenceSegment],
    match_file_exists: bool,
    saved_names_by_speaker: dict[int, str],
    store_dir: Path | None,
) -> SpeakerReviewOverview:
    """
    Build the immutable project state shown by the TUI.

    Args:
        manifest: Project manifest.
        source_media: Resolved source media path.
        sentences: Transcript sentences.
        match_file_exists: Whether a match result file exists.
        saved_names_by_speaker: Existing speaker map loaded from disk.
        store_dir: Optional voiceprint store directory.

    Returns:
        Project overview for display.
    """
    return SpeakerReviewOverview(
        project_id=manifest.project_id,
        title=manifest.title,
        project_status=manifest.status,
        source_name=manifest.source.filename or source_media.name,
        duration_ms=_project_duration_ms(sentences),
        match_file_exists=match_file_exists,
        saved_names_by_speaker=dict(saved_names_by_speaker),
        voiceprint=_load_voiceprint_review_progress(manifest.project_id, store_dir),
    )


def _load_voiceprint_review_progress(project_id: str, store_dir: Path | None) -> VoiceprintReviewProgress:
    """
    Load project-scoped voiceprint capture and embedding state.

    Args:
        project_id: Project id from the manifest.
        store_dir: Optional voiceprint store directory.

    Returns:
        Voiceprint progress for the current project.
    """
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_voiceprint_samples_for_project(project_id, db_path)
    names_by_speaker: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        names_by_speaker[sample.project_speaker_id].add(sample.speaker_name)
    model, embedded_ids, embed_error = _load_embedding_state(db_path)
    return VoiceprintReviewProgress(
        captured_names_by_speaker={
            speaker_id: frozenset(names)
            for speaker_id, names in names_by_speaker.items()
        },
        captured_sample_ids=frozenset(sample.sample_id for sample in samples),
        embed_model=model,
        embedded_sample_ids=embedded_ids,
        embed_error=embed_error,
    )


def _load_embedding_state(db_path: Path) -> tuple[str | None, frozenset[int] | None, str | None]:
    """
    Load embedded sample ids for the configured voiceprint model.

    Args:
        db_path: Voiceprint SQLite path.

    Returns:
        Model, embedded sample ids, and optional error text.
    """
    try:
        _, model = resolve_voiceprint_embedding_options(provider=None, model=None)
        return model, frozenset(list_embedded_sample_ids(model, db_path)), None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _project_duration_ms(sentences: list[SentenceSegment]) -> int:
    """
    Return the transcript duration from the latest sentence end.

    Args:
        sentences: Transcript sentences.

    Returns:
        Duration in milliseconds.
    """
    return max((sentence.end_time_ms for sentence in sentences), default=0)


def _segments_by_speaker(sentences: list[SentenceSegment]) -> dict[int, list[SentenceSegment]]:
    """Group non-empty transcript segments by speaker id."""
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for sentence in sentences:
        if sentence.speaker_id is not None and sentence.text.strip():
            grouped[sentence.speaker_id].append(sentence)
    return dict(grouped)


def _load_existing_mapping(path: Path) -> dict[int, str]:
    """Load the current project speaker map if it exists."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


def _build_review_speakers(
    segments_by_speaker: dict[int, list[SentenceSegment]],
    mapping: dict[int, str],
    person_mapping: dict[int, int],
    matches: dict[int, SpeakerMatchCandidate],
) -> list[ReviewSpeaker]:
    """Build mutable speaker review rows."""
    return [
        _review_speaker(speaker_id, segments, mapping, person_mapping, matches)
        for speaker_id, segments in sorted(segments_by_speaker.items())
    ]


def _review_speaker(
    speaker_id: int,
    segments: list[SentenceSegment],
    mapping: dict[int, str],
    person_mapping: dict[int, int],
    matches: dict[int, SpeakerMatchCandidate],
) -> ReviewSpeaker:
    """Build one speaker review row."""
    label = speaker_id_to_label(speaker_id)
    match = matches.get(speaker_id)
    current_name = mapping.get(speaker_id) or accepted_review_name(match) or label
    person_id = person_mapping.get(speaker_id) or accepted_review_person_id(match)
    ignored = (
        speaker_id in mapping
        and current_name == label
        and (match is None or voiceprint_match_status(match) != MATCH_STATUS_BELOW_THRESHOLD)
    )
    return ReviewSpeaker(speaker_id, label, segments, current_name, match, ignored=ignored, person_id=person_id)


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


def _trim_sample_text(text: str, *, limit: int = 90) -> str:
    """Trim a transcript sample for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
