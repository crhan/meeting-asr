"""Modal identity selection for speaker review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from app.presentation.tui.inputs import ReadlineInput
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_people import KnownPerson, find_person_by_name
from app.presentation.tui.speaker_status import render_match_lines
from app.voiceprint_people import create_voiceprint_person
from app.voiceprint_store import get_voiceprint_db_path

IDENTITY_LIST_LIMIT = 14


@dataclass(frozen=True, slots=True)
class IdentitySelection:
    """Selected stable speaker identity."""

    person_id: int
    name: str
    created: bool = False


@dataclass(frozen=True, slots=True)
class ScoredPerson:
    """Known person with a speaker-match score for display."""

    person_id: int
    name: str
    score: float | None


class IdentityInput(ReadlineInput):
    """Search input for the identity selection modal."""

    BINDINGS = [
        Binding("escape", "cancel_identity", "Cancel", show=False),
        Binding("up", "previous_person", "Previous person", show=False),
        Binding("down", "next_person", "Next person", show=False),
        Binding("tab", "accept_person", "Accept person", show=False),
    ]

    def action_cancel_identity(self) -> None:
        """Cancel identity selection."""
        self.app.screen.dismiss(None)

    def action_previous_person(self) -> None:
        """Move to the previous visible person."""
        self.app.screen.action_previous_person()

    def action_next_person(self) -> None:
        """Move to the next visible person."""
        self.app.screen.action_next_person()

    def action_accept_person(self) -> None:
        """Accept the highlighted person."""
        self.app.screen.action_accept_person()


class IdentityEditScreen(ModalScreen[IdentitySelection | None]):
    """Modal picker for binding a speaker to a stable person id."""

    CSS = """
    IdentityEditScreen {
        align: center middle;
    }
    #identity-box {
        width: 96;
        height: auto;
        max-height: 88%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #identity-title {
        text-style: bold;
    }
    #identity-context {
        margin: 1 0;
    }
    #identity-search {
        height: 3;
    }
    #identity-list {
        min-height: 8;
        margin: 1 0;
    }
    #identity-status {
        height: 2;
        color: $text-muted;
    }
    """

    BINDINGS = [Binding("escape", "cancel_identity", "Cancel", show=False)]

    def __init__(
        self,
        *,
        speaker_label: str,
        current_name: str,
        current_person_id: int | None,
        match: SpeakerMatchCandidate | None,
        people: list[KnownPerson],
        store_dir: Path | None,
    ) -> None:
        """
        Create the identity selection modal.

        Args:
            speaker_label: Anonymous project speaker label.
            current_name: Current display name.
            current_person_id: Current stable person id if bound.
            match: Voiceprint match data for this speaker.
            people: Known global voiceprint people.
            store_dir: Optional voiceprint store directory.
        """
        super().__init__()
        self.speaker_label = speaker_label
        self.current_name = current_name
        self.current_person_id = current_person_id
        self.match = match
        self.people = list(people)
        self.store_dir = store_dir
        self.search_query = ""
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        """Build the identity selection modal."""
        with Vertical(id="identity-box"):
            yield Static(f"Edit Identity | {self.speaker_label}", id="identity-title")
            yield Static(self._context_text(), id="identity-context")
            yield IdentityInput(placeholder="Type to filter people, or +Name to create", id="identity-search")
            yield Static(id="identity-list")
            yield Static(self._status_text(), id="identity-status")

    def on_mount(self) -> None:
        """Focus search input and render the candidate list."""
        self.query_one("#identity-search", Input).focus()
        self._refresh()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter people while typing."""
        event.stop()
        self.search_query = event.value
        self.selected_index = 0
        self._refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Accept the typed or highlighted identity."""
        event.stop()
        self._submit(event.value.strip())

    def action_next_person(self) -> None:
        """Move highlighted person down."""
        self._move_selection(1)

    def action_previous_person(self) -> None:
        """Move highlighted person up."""
        self._move_selection(-1)

    def action_accept_person(self) -> None:
        """Accept the highlighted person."""
        person = self._selected_person()
        if person is not None:
            self.dismiss(IdentitySelection(person.person_id, person.name))

    def action_cancel_identity(self) -> None:
        """Cancel identity selection."""
        self.dismiss(None)

    def _submit(self, value: str) -> None:
        """Resolve typed input to an identity result."""
        if value.startswith("+"):
            self._create_person(value[1:].strip())
            return
        exact = find_person_by_name(value, self.people) if value else None
        if exact is not None:
            self.dismiss(IdentitySelection(exact.person_id, exact.name))
            return
        person = self._selected_person()
        if person is not None:
            self.dismiss(IdentitySelection(person.person_id, person.name))
            return
        self.query_one("#identity-status", Static).update(
            f"Unknown person: {escape(value)}. Use +Name to create."
        )

    def _create_person(self, name: str) -> None:
        """Create a new stable person explicitly."""
        if not name:
            self.query_one("#identity-status", Static).update("New person name is empty. Use +Name.")
            return
        duplicate = find_person_by_name(name, self.people)
        if duplicate is not None:
            self.query_one("#identity-status", Static).update(
                f"Person already exists: {escape(duplicate.name)} #{duplicate.person_id}."
            )
            return
        try:
            row = create_voiceprint_person(name, get_voiceprint_db_path(self.store_dir))
        except Exception as exc:  # noqa: BLE001
            self.query_one("#identity-status", Static).update(f"Failed to create person: {escape(str(exc))}")
            return
        self.dismiss(IdentitySelection(row.speaker_id, row.name, created=True))

    def _move_selection(self, delta: int) -> None:
        """Move highlighted person within the filtered list."""
        people = self._filtered_people()
        if not people:
            self.selected_index = 0
        else:
            self.selected_index = max(0, min(self.selected_index + delta, len(people) - 1))
        self._refresh()

    def _refresh(self) -> None:
        """Refresh list and status widgets."""
        self.query_one("#identity-list", Static).update(self._list_text())
        self.query_one("#identity-status", Static).update(self._status_text())

    def _list_text(self) -> str:
        """Render the sorted people list."""
        people = self._filtered_people()
        lines = [f"[b]People[/b] [dim]{len(people)}/{len(self.people)} shown, sorted by score[/dim]"]
        if not people:
            lines.append("- No matching known person")
        start, visible = _visible_window(people, self.selected_index)
        for offset, person in enumerate(visible):
            index = start + offset
            marker = ">" if index == self.selected_index else " "
            row = f"{marker} {escape(person.name):<24} {_score_text(person.score):<11} [dim]#{person.person_id}[/]"
            lines.append(f"[reverse]{row}[/]" if index == self.selected_index else row)
        if len(people) > IDENTITY_LIST_LIMIT:
            end = start + len(visible)
            lines.append(f"[dim]Showing {start + 1}-{end}/{len(people)}. Type to filter.[/]")
        return "\n".join(lines)

    def _context_text(self) -> str:
        """Render current identity and match context."""
        person = f"person #{self.current_person_id}" if self.current_person_id is not None else "no person id"
        lines = [f"Current: [green]{escape(self.current_name)}[/] ([dim]{person}[/])"]
        if self.match is not None:
            lines.extend(render_match_lines(self.match))
        return "\n".join(lines)

    def _status_text(self) -> str:
        """Render action hints."""
        query = self.search_query.strip()
        if query.startswith("+"):
            return f"Enter creates new stable person: {escape(query[1:].strip() or 'Name')}"
        return "Up/Down select | Enter choose highlighted/exact person | +Name create | Esc cancel"

    def _filtered_people(self) -> list[ScoredPerson]:
        """Return scored people filtered by search query."""
        query = self.search_query.strip().casefold()
        people = _scored_people(self.people, self.match)
        if not query or query.startswith("+"):
            return people
        return [person for person in people if query in person.name.casefold()]

    def _selected_person(self) -> ScoredPerson | None:
        """Return the highlighted scored person."""
        people = self._filtered_people()
        if not people:
            return None
        self.selected_index = max(0, min(self.selected_index, len(people) - 1))
        return people[self.selected_index]


def _scored_people(
    people: list[KnownPerson],
    match: SpeakerMatchCandidate | None,
) -> list[ScoredPerson]:
    """Build people sorted by descending voiceprint score."""
    score_by_person, score_by_name = _score_maps(match)
    rows = [
        ScoredPerson(
            person.person_id,
            person.name,
            _person_score(person, score_by_person, score_by_name),
        )
        for person in people
    ]
    rows.extend(_missing_match_people(rows, match))
    return sorted(rows, key=lambda item: (_score_rank(item.score), item.name.casefold(), item.person_id))


def _person_score(
    person: KnownPerson,
    score_by_person: dict[int, float],
    score_by_name: dict[str, float],
) -> float | None:
    """Return the match score for one person without treating 0.0 as missing."""
    if person.person_id in score_by_person:
        return score_by_person[person.person_id]
    return score_by_name.get(_normalize_name(person.name))


def _score_maps(match: SpeakerMatchCandidate | None) -> tuple[dict[int, float], dict[str, float]]:
    """Return best known scores by person id and normalized name."""
    scores_by_id = {}
    scores_by_name = {}
    if match is None:
        return scores_by_id, scores_by_name
    for candidate in match.candidates:
        _remember_score(scores_by_id, candidate.person_id, candidate.score)
        _remember_name_score(scores_by_name, candidate.name, candidate.score)
    _remember_score(scores_by_id, match.best_person_id, match.best_score)
    if match.best_name is not None:
        _remember_name_score(scores_by_name, match.best_name, match.best_score)
    return scores_by_id, scores_by_name


def _remember_score(scores: dict[int, float], person_id: int | None, score: float | None) -> None:
    """Store the best score for one person id."""
    if person_id is not None and score is not None:
        scores[person_id] = max(score, scores.get(person_id, score))


def _remember_name_score(scores: dict[str, float], name: str, score: float | None) -> None:
    """Store the best score for one normalized person name."""
    if name and score is not None:
        normalized = _normalize_name(name)
        scores[normalized] = max(score, scores.get(normalized, score))


def _missing_match_people(
    rows: list[ScoredPerson],
    match: SpeakerMatchCandidate | None,
) -> list[ScoredPerson]:
    """Add scored match candidates missing from the people table."""
    if match is None:
        return []
    known_ids = {row.person_id for row in rows}
    missing = []
    for candidate in match.candidates:
        if candidate.person_id is not None and candidate.person_id not in known_ids:
            missing.append(ScoredPerson(candidate.person_id, candidate.name, candidate.score))
            known_ids.add(candidate.person_id)
    return missing


def _score_rank(score: float | None) -> tuple[int, float]:
    """Sort scored people first and higher scores earlier."""
    if score is None:
        return (1, 0.0)
    return (0, -score)


def _score_text(score: float | None) -> str:
    """Format a score for the identity list."""
    return "score -" if score is None else f"score {score:.3f}"


def _visible_window(people: list[ScoredPerson], selected_index: int) -> tuple[int, list[ScoredPerson]]:
    """Return a visible list window around the selected row."""
    if len(people) <= IDENTITY_LIST_LIMIT:
        return 0, people
    last_start = max(0, len(people) - IDENTITY_LIST_LIMIT)
    start = max(0, min(selected_index - IDENTITY_LIST_LIMIT + 1, last_start))
    return start, people[start : start + IDENTITY_LIST_LIMIT]


def _normalize_name(name: str) -> str:
    """Normalize a display name for score lookup."""
    return " ".join(name.strip().split()).casefold()
