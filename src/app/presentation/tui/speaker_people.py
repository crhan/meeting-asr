"""Known person helpers for the speaker review TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_speakers

PEOPLE_LIST_LIMIT = 8


@dataclass(frozen=True, slots=True)
class KnownPerson:
    """Known voiceprint person shown in identity suggestions."""

    person_id: int
    name: str


def load_people(store_dir: Path | None) -> list[KnownPerson]:
    """
    Load known people from the global voiceprint registry.

    Args:
        store_dir: Optional voiceprint store directory.

    Returns:
        Stable person rows for the TUI.
    """
    db_path = get_voiceprint_db_path(store_dir)
    return [KnownPerson(row.speaker_id, row.name) for row in list_voiceprint_speakers(db_path)]


def load_existing_person_mapping(path: Path) -> dict[int, int]:
    """
    Load the current project speaker-to-person map if it exists.

    Args:
        path: Mapping JSON path.

    Returns:
        Project speaker id to voiceprint person id mapping.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): int(value) for key, value in payload.items()}


def identity_candidates(
    *,
    current_person_id: int | None,
    match_person_id: int | None,
    match_name: str | None,
    people: list[KnownPerson],
) -> list[KnownPerson]:
    """
    Build deduplicated identity suggestions for one project speaker.

    Args:
        current_person_id: Currently selected voiceprint person id.
        match_person_id: Candidate person id from voiceprint matching.
        match_name: Candidate display name from voiceprint matching.
        people: Known people loaded from the registry.

    Returns:
        Deduplicated suggestions ordered by relevance.
    """
    candidates: list[KnownPerson] = []
    current_person = find_person_by_id(current_person_id, people)
    if current_person is not None:
        candidates.append(current_person)
    if match_name and match_person_id is not None:
        candidates.append(KnownPerson(match_person_id, match_name))
    candidates.extend(people)
    return _dedupe_people(candidates)


def find_person_by_id(person_id: int | None, people: list[KnownPerson]) -> KnownPerson | None:
    """
    Find a known person by id.

    Args:
        person_id: Optional person id.
        people: Known people.

    Returns:
        Matching person or ``None``.
    """
    if person_id is None:
        return None
    for person in people:
        if person.person_id == person_id:
            return person
    return None


def find_person_by_name(name: str, people: list[KnownPerson]) -> KnownPerson | None:
    """
    Find a known person by normalized display name.

    Args:
        name: Display name typed by the user.
        people: Known people.

    Returns:
        Matching person or ``None``.
    """
    normalized = _normalize_person_name(name)
    for person in people:
        if _normalize_person_name(person.name) == normalized:
            return person
    return None


def optional_person_id(value: object) -> int | None:
    """
    Parse an optional positive person id from a JSON value.

    Args:
        value: JSON value.

    Returns:
        Positive person id, or ``None``.
    """
    if value is None:
        return None
    try:
        person_id = int(value)
    except (TypeError, ValueError):
        return None
    return person_id if person_id > 0 else None


def render_people_selection_lines(
    *,
    suggestions: list[KnownPerson],
    known_people: list[KnownPerson],
    selected_index: int,
    query: str,
) -> list[str]:
    """
    Render the TUI known-person selector.

    Args:
        suggestions: Filtered people shown for the current speaker.
        known_people: Complete known person list for duplicate checks.
        selected_index: Highlighted suggestion index.
        query: Current search input.

    Returns:
        Rich markup lines for the identity pane.
    """
    query_text = query.strip()
    lines = [f"[b]People[/b] [dim]{len(suggestions)}/{len(known_people)} shown[/dim]"]
    if not suggestions:
        lines.append("- No matching known person")
    start, visible = _visible_people_window(suggestions, selected_index)
    for offset, person in enumerate(visible):
        index = start + offset
        marker = ">" if index == selected_index else " "
        row = f"{marker} {escape(person.name)} [dim]#{person.person_id}[/]"
        lines.append(f"[reverse]{row}[/]" if index == selected_index else row)
    if len(suggestions) > PEOPLE_LIST_LIMIT:
        end = start + len(visible)
        lines.append(f"[dim]Showing {start + 1}-{end}/{len(suggestions)}. Type to filter.[/]")
    if query_text.startswith("+"):
        lines.append(_create_person_hint(query_text[1:].strip(), known_people))
    else:
        lines.append("[dim]Up/Down selects. Enter selects highlighted person. Use +Name to create.[/]")
    return lines


def _dedupe_people(people: list[KnownPerson]) -> list[KnownPerson]:
    """Deduplicate people by stable id while preserving order."""
    seen: set[int] = set()
    deduped: list[KnownPerson] = []
    for person in people:
        if person.person_id > 0 and person.person_id not in seen:
            seen.add(person.person_id)
            deduped.append(person)
    return deduped


def _normalize_person_name(name: str) -> str:
    """Normalize a person display name for exact selection."""
    return " ".join(name.strip().split()).casefold()


def _visible_people_window(
    suggestions: list[KnownPerson],
    selected_index: int,
) -> tuple[int, list[KnownPerson]]:
    """Return a scroll window that keeps the highlighted person visible."""
    if len(suggestions) <= PEOPLE_LIST_LIMIT:
        return 0, suggestions
    last_start = max(0, len(suggestions) - PEOPLE_LIST_LIMIT)
    start = max(0, min(selected_index - PEOPLE_LIST_LIMIT + 1, last_start))
    return start, suggestions[start : start + PEOPLE_LIST_LIMIT]


def _create_person_hint(name: str, people: list[KnownPerson]) -> str:
    """Render explicit create-person guidance."""
    if not name:
        return "[dim]+Name creates a new stable person after Enter.[/]"
    duplicate = find_person_by_name(name, people)
    if duplicate is not None:
        return f"[yellow]Person already exists: {escape(duplicate.name)} #{duplicate.person_id}[/]"
    return f"[yellow]Enter will create new person: {escape(name)}[/]"
