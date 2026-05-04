"""Voiceprint public id helpers."""

from __future__ import annotations

import sqlite3

from app.object_ids import new_prefixed_id

VOICEPRINT_PERSON_ID_PREFIX = "vpp-"
VOICEPRINT_SAMPLE_ID_PREFIX = "vps-"


def ensure_voiceprint_public_ids(connection: sqlite3.Connection) -> None:
    """
    Ensure voiceprint people and samples have stable public ids.

    Args:
        connection: SQLite connection.
    """
    _ensure_public_id_column(connection, "voiceprint_speakers")
    _ensure_public_id_column(connection, "voiceprint_samples")
    _backfill_public_ids(connection, "voiceprint_speakers", VOICEPRINT_PERSON_ID_PREFIX)
    _backfill_public_ids(connection, "voiceprint_samples", VOICEPRINT_SAMPLE_ID_PREFIX)


def new_speaker_public_id(connection: sqlite3.Connection) -> str:
    """
    Return a new voiceprint person public id.

    Args:
        connection: SQLite connection.

    Returns:
        Prefixed public id.
    """
    return _new_public_id(connection, "voiceprint_speakers", VOICEPRINT_PERSON_ID_PREFIX)


def new_sample_public_id(connection: sqlite3.Connection) -> str:
    """
    Return a new voiceprint sample public id.

    Args:
        connection: SQLite connection.

    Returns:
        Prefixed public id.
    """
    return _new_public_id(connection, "voiceprint_samples", VOICEPRINT_SAMPLE_ID_PREFIX)


def valid_person_public_id(public_id: str) -> bool:
    """
    Return whether a string is a voiceprint person public id.

    Args:
        public_id: Candidate public id.

    Returns:
        True when the value has the expected shape.
    """
    return _valid_public_id(public_id, VOICEPRINT_PERSON_ID_PREFIX)


def _ensure_public_id_column(connection: sqlite3.Connection, table: str) -> None:
    """Add the public_id column and index when upgrading an old database."""
    columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if "public_id" not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN public_id TEXT")
    connection.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_public_id ON {table}(public_id)")


def _backfill_public_ids(connection: sqlite3.Connection, table: str, prefix: str) -> None:
    """Backfill missing public ids for older local databases."""
    rows = connection.execute(f"SELECT id FROM {table} WHERE public_id IS NULL OR public_id = ''").fetchall()
    for row in rows:
        public_id = _new_public_id(connection, table, prefix)
        connection.execute(f"UPDATE {table} SET public_id = ? WHERE id = ?", (public_id, int(row["id"])))


def _new_public_id(connection: sqlite3.Connection, table: str, prefix: str) -> str:
    """Generate a collision-free public id for a voiceprint table."""
    return new_prefixed_id(
        prefix,
        lambda public_id: connection.execute(f"SELECT 1 FROM {table} WHERE public_id = ?", (public_id,)).fetchone()
        is not None,
    )


def _valid_public_id(public_id: str, prefix: str) -> bool:
    """Return whether a string is one of our prefixed public ids."""
    suffix = public_id.removeprefix(prefix)
    return public_id.startswith(prefix) and len(suffix) == 16 and all(
        character in "0123456789abcdef" for character in suffix
    )
