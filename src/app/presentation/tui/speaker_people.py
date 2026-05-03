"""Known person helpers for the speaker review TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_speakers


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


def _normalize_person_name(name: str) -> str:
    """Normalize a person display name for exact selection."""
    return " ".join(name.strip().split()).casefold()
