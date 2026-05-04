"""Lifecycle management for stable voiceprint people."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.voiceprint_ids import new_speaker_public_id
from app.voiceprint_store import (
    VoiceprintSpeakerRow,
    _configure_connection,
    _ensure_schema,
    _find_speaker,
    _normalize_name,
    _now_iso,
    _resolve_db_path,
    _speaker_by_id,
    _speaker_by_name,
)


def create_voiceprint_person(name: str, db_path: Path | None = None) -> VoiceprintSpeakerRow:
    """
    Create a new voiceprint person with a stable database id.

    Args:
        name: Display name for the person.
        db_path: Optional SQLite path.

    Returns:
        Created person row.

    Raises:
        ValueError: If the name is empty or already exists.
    """
    database_path = _resolve_db_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    normalized = _normalize_name(name)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        existing = _speaker_by_name(connection, name)
        if existing is not None:
            raise ValueError(f"Person already exists: {existing.name} (id {existing.public_id}).")
        cursor = connection.execute(
            """
            INSERT INTO voiceprint_speakers (public_id, name, normalized_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_speaker_public_id(connection), name.strip(), normalized, now, now),
        )
        created = _speaker_by_id(connection, int(cursor.lastrowid))
    if created is None:
        raise RuntimeError(f"Failed to create voiceprint person: {name}")
    return created


def get_voiceprint_person(person_ref: int | str, db_path: Path | None = None) -> VoiceprintSpeakerRow | None:
    """
    Load one voiceprint person by stable id.

    Args:
        person_ref: Voiceprint person public id, numeric id, or name.
        db_path: Optional SQLite path.

    Returns:
        Matching person row, or ``None``.
    """
    if isinstance(person_ref, int) and person_ref <= 0:
        return None
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return None
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        if isinstance(person_ref, int):
            return _speaker_by_id(connection, person_ref)
        return _find_speaker(connection, person_ref)


def rename_voiceprint_person(person_ref: int | str, name: str, db_path: Path | None = None) -> VoiceprintSpeakerRow:
    """
    Rename an existing voiceprint person by stable id.

    Args:
        person_ref: Voiceprint person public id, numeric id, or name.
        name: New display name.
        db_path: Optional SQLite path.

    Returns:
        Updated person row.

    Raises:
        LookupError: If the person id does not exist.
        ValueError: If the name is empty or already belongs to another person.
    """
    if isinstance(person_ref, int) and person_ref <= 0:
        raise LookupError(f"No voiceprint person found for id: {person_ref}")
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        raise LookupError(f"No voiceprint person found for id: {person_ref}")
    now = _now_iso()
    normalized = _normalize_name(name)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        existing = _speaker_by_id(connection, person_ref) if isinstance(person_ref, int) else _find_speaker(connection, person_ref)
        if existing is None:
            raise LookupError(f"No voiceprint person found for id: {person_ref}")
        duplicate = _speaker_by_name(connection, name)
        if duplicate is not None and duplicate.speaker_id != existing.speaker_id:
            raise ValueError(f"Person name already belongs to id {duplicate.public_id}: {duplicate.name}.")
        connection.execute(
            """
            UPDATE voiceprint_speakers
            SET name = ?, normalized_name = ?, updated_at = ?
            WHERE id = ?
            """,
            (name.strip(), normalized, now, existing.speaker_id),
        )
        updated = _speaker_by_id(connection, existing.speaker_id)
    if updated is None:
        raise RuntimeError(f"Failed to rename voiceprint person id {person_ref}")
    return updated
