"""Chronological transcript view for the speaker review TUI.

The default speaker review groups segments by speaker. The timeline view shows
every transcript sentence in time order so reviewers can listen to the meeting
as it unfolded, spot mis-attributed sentences, and reassign them to a different
speaker without leaving the TUI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from app.models import SentenceSegment
from app.presentation.tui.i18n import tr
from app.presentation.tui.speaker_models import ReviewSpeaker
from app.utils import format_ms_timestamp


@dataclass(frozen=True, slots=True)
class TimelineRow:
    """One row in the chronological transcript view."""

    speaker_id: int
    label: str
    name: str
    segment: SentenceSegment


def build_timeline_rows(speakers: Sequence[ReviewSpeaker]) -> list[TimelineRow]:
    """Flatten speakers into a single time-sorted transcript list.

    Args:
        speakers: Current review speakers (mutable references; the rows hold
            the same segment objects so edits stay in sync).

    Returns:
        Sentences ordered by ``begin_time_ms`` then ``end_time_ms``.
    """
    rows: list[TimelineRow] = []
    for speaker in speakers:
        for segment in speaker.segments:
            rows.append(
                TimelineRow(
                    speaker_id=speaker.speaker_id,
                    label=speaker.label,
                    name=speaker.current_name,
                    segment=segment,
                )
            )
    rows.sort(key=lambda row: (row.segment.begin_time_ms, row.segment.end_time_ms))
    return rows


def render_timeline_pane(
    rows: Sequence[TimelineRow],
    *,
    selected_index: int,
    page_start: int,
    page_size: int,
    speaker_count: int,
    edited_keys: set[tuple[int | None, int, int]] = frozenset(),
    reassigned_keys: set[tuple[int | None, int, int]] = frozenset(),
    title: str | None = None,
) -> str:
    """Render the timeline view as Rich markup.

    Args:
        rows: All timeline rows.
        selected_index: Currently selected row index.
        page_start: First visible row index.
        page_size: Number of rows to render.
        speaker_count: Total active speaker count, displayed in the header.
        edited_keys: Sentence keys with staged transcript text edits.
        reassigned_keys: Sentence keys with staged speaker reassignments.
        title: Optional pane title; defaults to localized ``Timeline``.

    Returns:
        Markup string suitable for ``Static.update``.
    """
    pane_title = title or tr("Timeline", "时间轴")
    if not rows:
        return tr(
            f"[b]{pane_title}[/]\n[dim]No sentences in this transcript.[/]",
            f"[b]{pane_title}[/]\n[dim]当前转写没有可用句子。[/]",
        )
    header = tr(
        f"[b]{pane_title}[/] [dim]{len(rows)} sentences across {speaker_count} speakers[/]",
        f"[b]{pane_title}[/] [dim]共 {len(rows)} 条句子，覆盖 {speaker_count} 个 speaker[/]",
    )
    end = min(page_start + page_size, len(rows))
    lines = [header]
    for index in range(page_start, end):
        lines.append(
            _render_timeline_line(
                rows[index],
                index=index,
                selected=index == selected_index,
                edited=_segment_key(rows[index].segment) in edited_keys,
                reassigned=_segment_key(rows[index].segment) in reassigned_keys,
            )
        )
    page_count = max(1, (len(rows) + page_size - 1) // page_size)
    page_number = page_start // max(1, page_size) + 1
    footer = tr(
        f"Page {page_number}/{page_count}  Rows {page_start + 1}-{end}/{len(rows)}  "
        "j/k move | Space play | r reassign | e edit | t back to speakers",
        f"第 {page_number}/{page_count} 页  行 {page_start + 1}-{end}/{len(rows)}  "
        "j/k 移动 | Space 播放 | r 改 speaker | e 改文字 | t 切回分组",
    )
    lines.append("")
    lines.append(footer)
    return "\n".join(lines)


def _render_timeline_line(
    row: TimelineRow,
    *,
    index: int,
    selected: bool,
    edited: bool,
    reassigned: bool,
) -> str:
    """Render one timeline row as Rich markup."""
    marker = ">" if selected else " "
    time_range = _segment_time_range(row.segment)
    name = row.name or row.label
    speaker_tag = f"{row.label} {escape(name)}"
    text = _trim_text(row.segment.text)
    badges = []
    if reassigned:
        badges.append("[yellow]reassigned[/]")
    if edited:
        badges.append("[yellow]edited[/]")
    badge = (" " + " ".join(badges)) if badges else ""
    line = f"{marker} [cyan]{time_range}[/] [magenta]{speaker_tag}[/]: {escape(text)}{badge}"
    if selected:
        return f"[reverse]{line}[/]"
    return line


def _segment_time_range(segment: SentenceSegment) -> str:
    """Format a segment as ``HH:MM:SS.mmm-HH:MM:SS.mmm``."""
    start = format_ms_timestamp(segment.begin_time_ms)
    end = format_ms_timestamp(segment.end_time_ms)
    return f"{start}-{end}"


def _trim_text(text: str, *, limit: int = 90) -> str:
    """Trim a transcript text for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."


def _segment_key(segment: SentenceSegment) -> tuple[int | None, int, int]:
    """Return a stable identity key for a segment."""
    return segment.sentence_id, segment.begin_time_ms, segment.end_time_ms


def segment_key(segment: SentenceSegment) -> tuple[int | None, int, int]:
    """Return a stable identity key for a segment (public helper)."""
    return _segment_key(segment)


@dataclass(frozen=True, slots=True)
class SpeakerPickOption:
    """One row in the reassignment picker."""

    speaker_id: int
    label: str
    name: str
    ignored: bool


class SpeakerPickScreen(ModalScreen[int | None]):
    """Modal picker that returns a target speaker_id, or ``None`` on cancel."""

    CSS = """
    SpeakerPickScreen {
        align: center middle;
    }
    #speaker-pick-box {
        width: 72;
        height: auto;
        max-height: 86%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #speaker-pick-title {
        text-style: bold;
    }
    #speaker-pick-context {
        margin: 1 0;
        color: $text-muted;
    }
    #speaker-pick-list {
        min-height: 6;
        margin: 1 0;
    }
    #speaker-pick-status {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "next_option", "Next"),
        Binding("k", "previous_option", "Previous"),
        Binding("down", "next_option", "Next", show=False),
        Binding("up", "previous_option", "Previous", show=False),
        Binding("enter", "accept_option", "Choose"),
        Binding("escape", "cancel_pick", "Cancel", show=False),
        Binding("q", "cancel_pick", "Cancel", show=False),
    ]

    def __init__(
        self,
        *,
        sentence_text: str,
        current_speaker_id: int,
        options: Sequence[SpeakerPickOption],
    ) -> None:
        """Create the speaker reassignment modal.

        Args:
            sentence_text: Trimmed transcript text shown for context.
            current_speaker_id: The sentence's current speaker.
            options: Candidate speakers to choose from.
        """
        super().__init__()
        self.sentence_text = sentence_text
        self.current_speaker_id = current_speaker_id
        self.options = list(options)
        self.selected_index = self._initial_selection()

    def compose(self) -> ComposeResult:
        """Build the picker layout."""
        with Vertical(id="speaker-pick-box"):
            yield Static(
                tr("Reassign sentence to speaker", "把这句话改给哪位 speaker"),
                id="speaker-pick-title",
            )
            yield Static(self._context_text(), id="speaker-pick-context")
            yield Static(self._list_text(), id="speaker-pick-list")
            yield Static(
                tr(
                    "j/k or up/down move | Enter choose | Esc cancel",
                    "j/k 或 ↑/↓ 移动 | Enter 选择 | Esc 取消",
                ),
                id="speaker-pick-status",
            )

    def action_next_option(self) -> None:
        """Move selection down."""
        if not self.options:
            return
        self.selected_index = min(self.selected_index + 1, len(self.options) - 1)
        self._refresh_list()

    def action_previous_option(self) -> None:
        """Move selection up."""
        if not self.options:
            return
        self.selected_index = max(self.selected_index - 1, 0)
        self._refresh_list()

    def action_accept_option(self) -> None:
        """Return the selected speaker_id to the parent app."""
        if not self.options:
            self.dismiss(None)
            return
        choice = self.options[self.selected_index]
        if choice.speaker_id == self.current_speaker_id:
            self.dismiss(None)
            return
        self.dismiss(choice.speaker_id)

    def action_cancel_pick(self) -> None:
        """Close the modal without changes."""
        self.dismiss(None)

    def _refresh_list(self) -> None:
        """Re-render the option list."""
        self.query_one("#speaker-pick-list", Static).update(self._list_text())

    def _initial_selection(self) -> int:
        """Pick the first non-current option as the default highlighted row."""
        for index, option in enumerate(self.options):
            if option.speaker_id != self.current_speaker_id:
                return index
        return 0

    def _context_text(self) -> str:
        """Render the sentence preview shown above the option list."""
        text = _trim_text(self.sentence_text, limit=160)
        return tr(f"Sentence: {escape(text)}", f"句子：{escape(text)}")

    def _list_text(self) -> str:
        """Render the candidate speaker list."""
        if not self.options:
            return tr(
                "[dim]No other speakers available.[/]", "[dim]没有可选的 speaker。[/]"
            )
        lines: list[str] = []
        for index, option in enumerate(self.options):
            marker = ">" if index == self.selected_index else " "
            note: list[str] = []
            if option.speaker_id == self.current_speaker_id:
                note.append(tr("current", "当前"))
            if option.ignored:
                note.append(tr("ignored", "已忽略"))
            suffix = (" [dim](" + ", ".join(note) + ")[/]") if note else ""
            display = f"{marker} {option.label}  {escape(option.name)}{suffix}"
            lines.append(
                f"[reverse]{display}[/]" if index == self.selected_index else display
            )
        return "\n".join(lines)


def speaker_pick_options(speakers: Sequence[ReviewSpeaker]) -> list[SpeakerPickOption]:
    """Return picker options sorted by speaker_id."""
    return [
        SpeakerPickOption(
            speaker_id=speaker.speaker_id,
            label=speaker.label,
            name=speaker.current_name or speaker.label,
            ignored=bool(speaker.ignored),
        )
        for speaker in sorted(speakers, key=lambda item: item.speaker_id)
    ]


def capture_speaker_baseline(
    speakers: Sequence[ReviewSpeaker],
) -> dict[tuple[int | None, int, int], int]:
    """Snapshot the original speaker_id of each segment for diffing later.

    The segment's own ``speaker_id`` field is the persistent source of truth;
    capturing it keeps the baseline stable even when the same segment instance
    is shared across multiple speaker rows in tests or rebuilt sessions.
    """
    baseline: dict[tuple[int | None, int, int], int] = {}
    for speaker in speakers:
        for seg in speaker.segments:
            speaker_id = (
                seg.speaker_id if seg.speaker_id is not None else speaker.speaker_id
            )
            baseline[_segment_key(seg)] = speaker_id
    return baseline


def speaker_by_id(
    speakers: Sequence[ReviewSpeaker],
    speaker_id: int,
) -> ReviewSpeaker | None:
    """Return the speaker with ``speaker_id`` or ``None`` when missing."""
    for speaker in speakers:
        if speaker.speaker_id == speaker_id:
            return speaker
    return None


def move_segment_between_speakers(
    source: ReviewSpeaker,
    target: ReviewSpeaker,
    segment: SentenceSegment,
) -> bool:
    """Move ``segment`` from ``source.segments`` to ``target.segments``.

    Args:
        source: Speaker that currently owns the segment.
        target: Speaker that should own the segment after the move.
        segment: Segment to move.

    Returns:
        ``True`` when the segment was found and moved; ``False`` otherwise.
    """
    target_key = _segment_key(segment)
    for index, owned in enumerate(source.segments):
        if owned is segment or _segment_key(owned) == target_key:
            source.segments.pop(index)
            if source.selected_sample_index >= len(source.segments):
                source.selected_sample_index = max(0, len(source.segments) - 1)
            segment.speaker_id = target.speaker_id
            insert_at = _segment_insert_position(target.segments, segment)
            target.segments.insert(insert_at, segment)
            target.selected_sample_index = insert_at
            return True
    return False


def _segment_insert_position(
    segments: list[SentenceSegment],
    segment: SentenceSegment,
) -> int:
    """Return the time-sorted insertion index for ``segment`` in ``segments``."""
    target = (segment.begin_time_ms, segment.end_time_ms)
    for index, existing in enumerate(segments):
        if (existing.begin_time_ms, existing.end_time_ms) > target:
            return index
    return len(segments)


__all__ = [
    "SpeakerPickOption",
    "SpeakerPickScreen",
    "TimelineRow",
    "build_timeline_rows",
    "capture_speaker_baseline",
    "move_segment_between_speakers",
    "render_timeline_pane",
    "segment_key",
    "speaker_by_id",
    "speaker_pick_options",
]
