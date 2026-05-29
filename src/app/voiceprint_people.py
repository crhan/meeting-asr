"""Lifecycle management for stable voiceprint people."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.voiceprint_ids import new_speaker_public_id
from app.voiceprint_store import (
    VoiceprintSpeakerRow,
    _configure_connection,
    _delete_empty_speaker,
    _ensure_schema,
    _find_speaker,
    _normalize_name,
    _now_iso,
    _resolve_db_path,
    _speaker_by_id,
    _speaker_by_name,
)


@dataclass(frozen=True)
class VoiceprintMergeResult:
    """Summary of a voiceprint person merge."""

    moved: int
    duplicates: int
    source_public_id: str
    source_name: str
    person: VoiceprintSpeakerRow


def create_voiceprint_person(
    name: str, db_path: Path | None = None
) -> VoiceprintSpeakerRow:
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
            raise ValueError(
                f"Person already exists: {existing.name} (id {existing.public_id})."
            )
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


def get_voiceprint_person(
    person_ref: int | str, db_path: Path | None = None
) -> VoiceprintSpeakerRow | None:
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


def rename_voiceprint_person(
    person_ref: int | str, name: str, db_path: Path | None = None
) -> VoiceprintSpeakerRow:
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
        existing = (
            _speaker_by_id(connection, person_ref)
            if isinstance(person_ref, int)
            else _find_speaker(connection, person_ref)
        )
        if existing is None:
            raise LookupError(f"No voiceprint person found for id: {person_ref}")
        duplicate = _speaker_by_name(connection, name)
        if duplicate is not None and duplicate.speaker_id != existing.speaker_id:
            raise ValueError(
                f"Person name already belongs to id {duplicate.public_id}: {duplicate.name}."
            )
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


def _resolve_person_ref(
    connection: sqlite3.Connection, person_ref: int | str
) -> VoiceprintSpeakerRow | None:
    """Resolve a person by positive numeric id, public id, or name."""
    if isinstance(person_ref, int):
        if person_ref <= 0:
            return None
        return _speaker_by_id(connection, person_ref)
    return _find_speaker(connection, person_ref)


def merge_voiceprint_people(
    from_ref: int | str,
    into_ref: int | str,
    db_path: Path | None = None,
) -> VoiceprintMergeResult:
    """
    Merge one voiceprint person's samples into another, then delete the source.

    Every sample under ``from_ref`` is moved onto ``into_ref``. A source sample
    whose audio (``clip_sha256``) already exists under the target is dropped as a
    duplicate instead of moved (its embedding is removed by cascade). The emptied
    source person is then deleted. There is no automatic undo, so confirm ids first.

    Args:
        from_ref: Source person public id, numeric id, or name (removed on success).
        into_ref: Target person public id, numeric id, or name (kept).
        db_path: Optional SQLite path.

    Returns:
        Merge summary with moved / duplicate counts and the kept person row.

    Raises:
        LookupError: If the store is missing or either person does not exist.
        ValueError: If source and target resolve to the same person.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        raise LookupError("Voiceprint store does not exist yet.")
    now = _now_iso()
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        source = _resolve_person_ref(connection, from_ref)
        target = _resolve_person_ref(connection, into_ref)
        if source is None:
            raise LookupError(f"No voiceprint person found for source: {from_ref}")
        if target is None:
            raise LookupError(f"No voiceprint person found for target: {into_ref}")
        if source.speaker_id == target.speaker_id:
            raise ValueError("Source and target voiceprint person are the same.")
        moved = 0
        duplicates = 0
        rows = connection.execute(
            "SELECT id, clip_sha256 FROM voiceprint_samples WHERE speaker_id = ?",
            (source.speaker_id,),
        ).fetchall()
        for row in rows:
            sample_id = int(row["id"])
            clip_sha256 = str(row["clip_sha256"])
            duplicate = connection.execute(
                "SELECT 1 FROM voiceprint_samples "
                "WHERE speaker_id = ? AND clip_sha256 = ? LIMIT 1",
                (target.speaker_id, clip_sha256),
            ).fetchone()
            if duplicate is not None:
                connection.execute(
                    "DELETE FROM voiceprint_samples WHERE id = ?", (sample_id,)
                )
                duplicates += 1
            else:
                connection.execute(
                    "UPDATE voiceprint_samples "
                    "SET speaker_id = ?, updated_at = ? WHERE id = ?",
                    (target.speaker_id, now, sample_id),
                )
                moved += 1
        connection.execute(
            "UPDATE voiceprint_speakers SET updated_at = ? WHERE id = ?",
            (now, target.speaker_id),
        )
        _delete_empty_speaker(connection, source.speaker_id)
        kept = _speaker_by_id(connection, target.speaker_id)
    if kept is None:
        raise RuntimeError("Failed to load merged voiceprint person.")
    return VoiceprintMergeResult(
        moved=moved,
        duplicates=duplicates,
        source_public_id=source.public_id,
        source_name=source.name,
        person=kept,
    )
