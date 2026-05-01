"""Textual UI for reviewing project speaker names."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Static

from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import load_manifest, project_paths, resolve_project_source_path
from app.speaker_labeling import load_transcript_result
from app.speaker_review import build_audio_preview_command
from app.utils import format_ms_timestamp
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_speakers

DEFAULT_SAMPLE_PAGE_SIZE = 6
SAMPLE_PANE_RESERVED_ROWS = 4
BROWSE_STATUS = (
    "Browse mode: move speakers/samples first, flip pages when needed, "
    "then press / only when you want to name this speaker."
)
EDIT_STATUS = (
    "Edit mode: type a name or search people. Tab chooses first suggestion, "
    "Enter applies, Esc cancels."
)


@dataclass(frozen=True, slots=True)
class SpeakerMatchCandidate:
    """One voiceprint match candidate for a project speaker."""

    name: str
    score: float | None
    accepted: bool


@dataclass(slots=True)
class ReviewSpeaker:
    """Mutable review state for one project speaker."""

    speaker_id: int
    label: str
    segments: list[SentenceSegment]
    current_name: str
    match: SpeakerMatchCandidate | None
    selected_sample_index: int = 0

    @property
    def segment_count(self) -> int:
        """Return the total number of transcript segments for this speaker."""
        return len(self.segments)


@dataclass(frozen=True, slots=True)
class SpeakerReviewSession:
    """Inputs needed by the speaker review TUI."""

    project_dir: Path
    source_media: Path
    speakers: list[ReviewSpeaker]
    people_names: list[str]
    page_size: int | None = None


@dataclass(frozen=True, slots=True)
class SpeakerReviewDecision:
    """Result returned by the TUI when it exits."""

    saved: bool
    mapping: dict[int, str]


class SpeakerReviewApp(App[SpeakerReviewDecision]):
    """Keyboard-first TUI for reviewing project speaker identities."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    .pane {
        border: round $accent;
        height: 100%;
        padding: 0 1;
    }
    #speakers {
        width: 32%;
    }
    #samples {
        width: 40%;
    }
    #identity {
        width: 28%;
    }
    #name-input {
        dock: bottom;
    }
    #status {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "next_speaker", "Next speaker"),
        Binding("k", "previous_speaker", "Previous speaker"),
        Binding("down", "next_sample", "Next sample", show=False),
        Binding("up", "previous_sample", "Previous sample", show=False),
        Binding("pagedown", "next_sample_page", "Next page"),
        Binding("pageup", "previous_sample_page", "Previous page"),
        Binding("]", "next_sample_page", "Next page", show=False),
        Binding("[", "previous_sample_page", "Previous page", show=False),
        Binding("space", "play_sample", "Play sample"),
        Binding("/", "edit_name", "Edit name"),
        Binding("tab", "accept_suggestion", "Accept suggestion", show=False),
        Binding("escape", "cancel_edit", "Cancel edit", show=False),
        Binding("a", "accept_match", "Accept match"),
        Binding("i", "ignore_speaker", "Ignore"),
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
        self.playback_process: subprocess.Popen | None = None
        self.search_query = ""

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        with Horizontal(id="main"):
            yield Static(id="speakers", classes="pane")
            yield Static(id="samples", classes="pane")
            yield Static(id="identity", classes="pane")
        yield Input(
            placeholder="Type a name or search known people",
            id="name-input",
            disabled=True,
        )
        yield Static(BROWSE_STATUS, id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Render the initial review state."""
        self._enter_browse_mode(BROWSE_STATUS)
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when the TUI closes."""
        self._stop_playback()

    def action_next_speaker(self) -> None:
        """Select the next speaker."""
        self._move_speaker(1)

    def action_previous_speaker(self) -> None:
        """Select the previous speaker."""
        self._move_speaker(-1)

    def action_next_sample(self) -> None:
        """Select the next visible sample."""
        self._move_sample(1)

    def action_previous_sample(self) -> None:
        """Select the previous visible sample."""
        self._move_sample(-1)

    def action_next_sample_page(self) -> None:
        """Select the first sample on the next page."""
        self._move_sample_page(1)

    def action_previous_sample_page(self) -> None:
        """Select the first sample on the previous page."""
        self._move_sample_page(-1)

    def action_play_sample(self) -> None:
        """Play the selected sample as audio."""
        try:
            self._play_sample(self._selected_sample())
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Preview failed: {exc}")

    def action_edit_name(self) -> None:
        """Focus the name input for manual entry."""
        field = self.query_one("#name-input", Input)
        field.display = True
        field.disabled = False
        field.value = self._speaker().current_name
        field.focus()
        self._set_status(EDIT_STATUS)

    def action_cancel_edit(self) -> None:
        """Cancel name editing and return to browse mode."""
        self._enter_browse_mode(BROWSE_STATUS)
        self._refresh()

    def action_accept_match(self) -> None:
        """Accept the current voiceprint match candidate."""
        speaker = self._speaker()
        if speaker.match is None or speaker.match.name == "unknown":
            self._set_status("No usable match for this speaker.")
            return
        speaker.current_name = speaker.match.name
        self._set_status(f"Accepted match for {speaker.label}: {speaker.match.name}.")
        self._refresh()

    def action_accept_suggestion(self) -> None:
        """Accept the first visible identity suggestion."""
        suggestions = self._suggestions(self._speaker())
        if not suggestions:
            self._set_status("No matching person suggestion.")
            return
        self._speaker().current_name = suggestions[0]
        self._enter_browse_mode(f"Set {self._speaker().label} to {suggestions[0]}.")
        self._refresh()

    def action_ignore_speaker(self) -> None:
        """Keep the anonymous speaker label so voiceprint capture skips it."""
        self._speaker().current_name = self._speaker().label
        self._set_status(f"Kept {self._speaker().label} anonymous.")
        self._refresh()

    def action_save(self) -> None:
        """Return the reviewed mapping to the CLI command."""
        self.exit(SpeakerReviewDecision(saved=True, mapping=self._mapping()))

    def action_quit_review(self) -> None:
        """Exit without saving."""
        self.exit(SpeakerReviewDecision(saved=False, mapping={}))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Refresh suggestions while the user types a name."""
        self.search_query = event.value
        self._refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply the typed name to the selected speaker."""
        name = event.value.strip()
        if name:
            self._speaker().current_name = name
            status = f"Set {self._speaker().label} to {name}."
        else:
            status = BROWSE_STATUS
        self._enter_browse_mode(status)
        self._refresh()

    def _move_speaker(self, delta: int) -> None:
        """Move the selected speaker index."""
        total = len(self.session.speakers)
        self.selected_speaker_index = (self.selected_speaker_index + delta) % total
        self._enter_browse_mode(BROWSE_STATUS)
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
        self.query_one("#speakers", Static).update(self._speaker_pane())
        self.query_one("#samples", Static).update(self._sample_pane())
        self.query_one("#identity", Static).update(self._identity_pane())

    def _speaker_pane(self) -> str:
        """Render the left speaker list."""
        lines = ["[b]Speakers[/b]"]
        for index, speaker in enumerate(self.session.speakers):
            marker = ">" if index == self.selected_speaker_index else " "
            style = _status_style(_speaker_status(speaker))
            label = f"{marker} {_status_icon(speaker)} {speaker.label}  {speaker.current_name}"
            lines.append(f"[{style}]{escape(label)}[/]")
        return "\n".join(lines)

    def _sample_pane(self) -> str:
        """Render the selected speaker samples."""
        speaker = self._speaker()
        lines = [f"[b]{escape(speaker.label)} samples[/b]"]
        page_start, segments = self._visible_segments(speaker)
        for offset, segment in enumerate(segments):
            index = page_start + offset
            prefix = ">" if index == speaker.selected_sample_index else " "
            time_range = _segment_time_range(segment)
            text = _trim_sample_text(segment.text)
            sample_line = f"{prefix} [cyan]{time_range}[/] {escape(text)}"
            if index == speaker.selected_sample_index:
                sample_line = f"[reverse]{sample_line}[/]"
            lines.append(sample_line)
        lines.append("")
        lines.append(self._sample_page_footer(speaker, page_start))
        return "\n".join(lines)

    def _identity_pane(self) -> str:
        """Render current identity state and suggestions."""
        speaker = self._speaker()
        lines = ["[b]Identity[/b]", f"Current: [green]{escape(speaker.current_name)}[/]"]
        lines.extend(_match_lines(speaker.match))
        lines.append("")
        lines.append("[b]People[/b]")
        suggestions = self._suggestions(speaker)
        lines.extend(f"- {escape(name)}" for name in suggestions[:8])
        if not suggestions:
            lines.append("- Type a new name with /")
        lines.extend(_help_lines())
        return "\n".join(lines)

    def _suggestions(self, speaker: ReviewSpeaker) -> list[str]:
        """Return names relevant to the selected speaker and search query."""
        names = _identity_candidates(speaker, self.session.people_names)
        query = self.search_query.strip().casefold()
        if not query:
            return names
        return [name for name in names if query in name.casefold()]

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

    def _mapping(self) -> dict[int, str]:
        """Return the current speaker mapping."""
        return {
            speaker.speaker_id: speaker.current_name.strip() or speaker.label
            for speaker in self.session.speakers
        }

    def _speaker(self) -> ReviewSpeaker:
        """Return the selected speaker."""
        return self.session.speakers[self.selected_speaker_index]

    def _selected_sample(self) -> SentenceSegment:
        """Return the selected sample segment."""
        speaker = self._speaker()
        return speaker.segments[speaker.selected_sample_index]

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

    def _enter_browse_mode(self, status: str) -> None:
        """Disable name input and return keyboard handling to browse mode."""
        field = self.query_one("#name-input", Input)
        field.value = ""
        field.display = False
        field.disabled = True
        field.blur()
        self.search_query = ""
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
) -> SpeakerReviewSession:
    """
    Load all data needed by the speaker review TUI.

    Args:
        project_dir: Project root.
        page_size: Optional samples-per-page override.
        store_dir: Optional voiceprint store directory.

    Returns:
        Speaker review session.
    """
    paths = project_paths(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    segments_by_speaker = _segments_by_speaker(result.sentences)
    if not segments_by_speaker:
        raise RuntimeError("No detected speakers found in the transcript.")
    manifest = load_manifest(paths.root)
    mapping = _load_existing_mapping(paths.speakers_dir / "speaker_map.json")
    matches = _load_match_candidates(paths.speakers_dir / "speaker_matches.json")
    return SpeakerReviewSession(
        project_dir=paths.root,
        source_media=resolve_project_source_path(paths.root, manifest),
        speakers=_build_review_speakers(segments_by_speaker, mapping, matches),
        people_names=_load_people_names(store_dir),
        page_size=page_size,
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
    return decision or SpeakerReviewDecision(saved=False, mapping={})


def render_speaker_review_summary(session: SpeakerReviewSession) -> str:
    """
    Render a non-interactive summary of the review queue.

    Args:
        session: Speaker review inputs.

    Returns:
        Plain terminal text.
    """
    lines = [
        f"Speaker review queue: {session.project_dir}",
        f"Known people: {len(session.people_names)}",
    ]
    for speaker in session.speakers:
        lines.append(_summary_line(speaker))
    return "\n".join(lines)


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


def _load_match_candidates(path: Path) -> dict[int, SpeakerMatchCandidate]:
    """Load voiceprint match candidates if they exist."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: dict[int, SpeakerMatchCandidate] = {}
    for item in payload.get("matches", []):
        if isinstance(item, dict) and "speaker_id" in item:
            candidates[int(item["speaker_id"])] = _match_candidate(item)
    return candidates


def _match_candidate(item: dict[str, object]) -> SpeakerMatchCandidate:
    """Convert one raw match row into a TUI candidate."""
    score = item.get("score")
    return SpeakerMatchCandidate(
        name=str(item.get("name") or "unknown"),
        score=float(score) if score is not None else None,
        accepted=bool(item.get("accepted")),
    )


def _load_people_names(store_dir: Path | None) -> list[str]:
    """Load known people names from the global voiceprint registry."""
    db_path = get_voiceprint_db_path(store_dir)
    return [row.name for row in list_voiceprint_speakers(db_path)]


def _build_review_speakers(
    segments_by_speaker: dict[int, list[SentenceSegment]],
    mapping: dict[int, str],
    matches: dict[int, SpeakerMatchCandidate],
) -> list[ReviewSpeaker]:
    """Build mutable speaker review rows."""
    return [
        _review_speaker(speaker_id, segments, mapping, matches)
        for speaker_id, segments in sorted(segments_by_speaker.items())
    ]


def _review_speaker(
    speaker_id: int,
    segments: list[SentenceSegment],
    mapping: dict[int, str],
    matches: dict[int, SpeakerMatchCandidate],
) -> ReviewSpeaker:
    """Build one speaker review row."""
    label = speaker_id_to_label(speaker_id)
    match = matches.get(speaker_id)
    current_name = mapping.get(speaker_id) or _accepted_match_name(match) or label
    return ReviewSpeaker(speaker_id, label, segments, current_name, match)


def _accepted_match_name(match: SpeakerMatchCandidate | None) -> str | None:
    """Return a usable accepted match name."""
    if match is None or not match.accepted or match.name == "unknown":
        return None
    return match.name


def _speaker_status(speaker: ReviewSpeaker) -> str:
    """Return the review status for one speaker."""
    if _has_conflict(speaker):
        return "conflict"
    if speaker.current_name == speaker.label:
        return "review"
    if speaker.match and speaker.current_name == speaker.match.name:
        return "matched"
    return "confirmed"


def _has_conflict(speaker: ReviewSpeaker) -> bool:
    """Return whether the current name conflicts with an accepted match."""
    if speaker.match is None or not speaker.match.accepted:
        return False
    if speaker.match.name == "unknown" or speaker.current_name == speaker.label:
        return False
    return speaker.current_name != speaker.match.name


def _status_style(status: str) -> str:
    """Map a status to a Rich style."""
    styles = {"conflict": "bold red", "review": "yellow", "matched": "green"}
    return styles.get(status, "bold green")


def _status_icon(speaker: ReviewSpeaker) -> str:
    """Return a compact status marker."""
    status = _speaker_status(speaker)
    return {"conflict": "!", "review": "?", "matched": "~"}.get(status, "+")


def _match_lines(match: SpeakerMatchCandidate | None) -> list[str]:
    """Render the selected speaker's voiceprint match."""
    if match is None:
        return ["Match: -"]
    score = "-" if match.score is None else f"{match.score:.3f}"
    state = "accepted" if match.accepted else "review"
    return [f"Match: {escape(match.name)}", f"Score: {score} {state}"]


def _help_lines() -> list[str]:
    """Render the fixed keyboard help shown in the identity pane."""
    return [
        "",
        "[b]Keys[/b]",
        "j/k: previous/next speaker",
        "up/down: choose sample",
        "PageUp/PageDown: sample page",
        "bracket keys: sample page",
        "space: play selected sample",
        "a: accept voiceprint match",
        "/: edit or search name",
        "Tab: choose first suggestion",
        "i: keep anonymous",
        "s: save, q: quit",
    ]


def _identity_candidates(speaker: ReviewSpeaker, people_names: list[str]) -> list[str]:
    """Build deduplicated identity suggestions for one speaker."""
    candidates = [speaker.current_name]
    if speaker.match and speaker.match.name != "unknown":
        candidates.append(speaker.match.name)
    candidates.extend(people_names)
    return _dedupe_names(candidates)


def _dedupe_names(names: list[str]) -> list[str]:
    """Deduplicate names while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        normalized = name.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


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


def _summary_line(speaker: ReviewSpeaker) -> str:
    """Render one plain summary row."""
    status = _speaker_status(speaker)
    match = "-" if speaker.match is None else speaker.match.name
    return (
        f"{speaker.label} speaker_id={speaker.speaker_id} "
        f"status={status} name={speaker.current_name} match={match}"
    )


def _segment_time_range(segment: SentenceSegment) -> str:
    """Format a transcript segment time range."""
    start = format_ms_timestamp(segment.begin_time_ms)
    end = format_ms_timestamp(segment.end_time_ms)
    return f"{start}-{end}"


def _trim_sample_text(text: str, *, limit: int = 90) -> str:
    """Trim a transcript sample for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
