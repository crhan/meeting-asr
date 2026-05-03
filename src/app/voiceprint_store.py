"""SQLite registry for speaker voiceprint references."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import get_data_dir

SCHEMA_VERSION = 1
VOICEPRINT_STORE_DIR = "voiceprints"
VOICEPRINT_DB_FILENAME = "voiceprints.sqlite"
VOICEPRINT_CLIPS_DIR = "clips"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS voiceprint_speakers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprint_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker_id INTEGER NOT NULL REFERENCES voiceprint_speakers(id) ON DELETE CASCADE,
  project_id TEXT NOT NULL,
  project_path TEXT NOT NULL,
  project_speaker_id INTEGER NOT NULL,
  source_path TEXT NOT NULL,
  clip_path TEXT NOT NULL UNIQUE,
  clip_rel_path TEXT NOT NULL,
  clip_sha256 TEXT NOT NULL,
  source_begin_time_ms INTEGER NOT NULL,
  source_end_time_ms INTEGER NOT NULL,
  clip_begin_time_ms INTEGER NOT NULL,
  clip_end_time_ms INTEGER NOT NULL,
  transcript_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprint_embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id INTEGER NOT NULL REFERENCES voiceprint_samples(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(sample_id, model)
);

CREATE INDEX IF NOT EXISTS idx_voiceprint_samples_speaker
  ON voiceprint_samples(speaker_id);
CREATE INDEX IF NOT EXISTS idx_voiceprint_samples_project
  ON voiceprint_samples(project_id);
CREATE INDEX IF NOT EXISTS idx_voiceprint_samples_sha256
  ON voiceprint_samples(clip_sha256);
CREATE INDEX IF NOT EXISTS idx_voiceprint_embeddings_model
  ON voiceprint_embeddings(model);
"""

UPSERT_SAMPLE_SQL = """
INSERT INTO voiceprint_samples (
  speaker_id, project_id, project_path, project_speaker_id,
  source_path, clip_path, clip_rel_path, clip_sha256,
  source_begin_time_ms, source_end_time_ms, clip_begin_time_ms,
  clip_end_time_ms, transcript_text, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(clip_path) DO UPDATE SET
  speaker_id = excluded.speaker_id,
  project_id = excluded.project_id,
  project_path = excluded.project_path,
  project_speaker_id = excluded.project_speaker_id,
  source_path = excluded.source_path,
  clip_rel_path = excluded.clip_rel_path,
  clip_sha256 = excluded.clip_sha256,
  source_begin_time_ms = excluded.source_begin_time_ms,
  source_end_time_ms = excluded.source_end_time_ms,
  clip_begin_time_ms = excluded.clip_begin_time_ms,
  clip_end_time_ms = excluded.clip_end_time_ms,
  transcript_text = excluded.transcript_text,
  updated_at = excluded.updated_at
"""


@dataclass(frozen=True, slots=True)
class VoiceprintSampleRow:
    """Stored voiceprint sample row."""

    sample_id: int
    speaker_id: int
    speaker_name: str
    project_id: str
    project_speaker_id: int
    clip_path: Path
    clip_rel_path: str
    clip_sha256: str
    source_begin_time_ms: int
    source_end_time_ms: int
    transcript_text: str


@dataclass(frozen=True, slots=True)
class VoiceprintSpeakerRow:
    """Stored speaker summary row."""

    speaker_id: int
    name: str
    sample_count: int
    project_count: int
    embedded_sample_count: int
    embedding_model_count: int
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class VoiceprintEmbeddingRow:
    """Stored voiceprint embedding row."""

    sample_id: int
    speaker_id: int
    speaker_name: str
    clip_path: Path
    model: str
    vector: list[float]


@dataclass(frozen=True, slots=True)
class DeletedVoiceprintSample:
    """Deleted voiceprint sample result."""

    sample_id: int
    speaker_id: int
    speaker_name: str
    clip_path: Path
    clip_deleted: bool


@dataclass(frozen=True, slots=True)
class StoredVoiceprintSample:
    """Voiceprint sample passed to SQLite storage."""

    speaker_name: str
    project_id: str
    project_path: Path
    project_speaker_id: int
    source_path: Path
    clip_path: Path
    clip_rel_path: str
    source_begin_time_ms: int
    source_end_time_ms: int
    clip_begin_time_ms: int
    clip_end_time_ms: int
    transcript_text: str
    person_id: int | None = None


def get_default_voiceprint_db_path() -> Path:
    """
    Return the default global voiceprint database path.

    Returns:
        XDG-compliant SQLite path.
    """
    return get_default_voiceprint_store_dir() / VOICEPRINT_DB_FILENAME


def get_default_voiceprint_store_dir() -> Path:
    """
    Return the default global voiceprint store directory.

    Returns:
        XDG-compliant voiceprint store directory.
    """
    return get_data_dir() / VOICEPRINT_STORE_DIR


def get_voiceprint_clip_dir(store_dir: Path | None = None) -> Path:
    """
    Return the clip directory for a voiceprint store.

    Args:
        store_dir: Optional voiceprint store directory.

    Returns:
        Voiceprint clip directory.
    """
    return _resolve_store_dir(store_dir) / VOICEPRINT_CLIPS_DIR


def get_voiceprint_db_path(store_dir: Path | None = None) -> Path:
    """
    Return the SQLite path for a voiceprint store.

    Args:
        store_dir: Optional voiceprint store directory.

    Returns:
        SQLite database path.
    """
    return _resolve_store_dir(store_dir) / VOICEPRINT_DB_FILENAME


def store_voiceprint_samples(samples: list[StoredVoiceprintSample], db_path: Path | None = None) -> Path:
    """
    Store voiceprint sample metadata in SQLite.

    Args:
        samples: Samples to record.
        db_path: Optional SQLite path.

    Returns:
        SQLite database path.
    """
    database_path = _resolve_db_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        for sample in samples:
            _upsert_sample(connection, sample)
    return database_path


def list_voiceprint_speakers(db_path: Path | None = None) -> list[VoiceprintSpeakerRow]:
    """
    List speakers stored in the voiceprint registry.

    Args:
        db_path: Optional SQLite path.

    Returns:
        Speaker summary rows.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT speakers.id, speakers.name,
                   COUNT(DISTINCT samples.id) AS sample_count,
                   COUNT(DISTINCT samples.project_id) AS project_count,
                   COUNT(DISTINCT embeddings.sample_id) AS embedded_sample_count,
                   COUNT(DISTINCT embeddings.model) AS embedding_model_count,
                   MAX(samples.updated_at) AS updated_at
            FROM voiceprint_speakers AS speakers
            LEFT JOIN voiceprint_samples AS samples ON samples.speaker_id = speakers.id
            LEFT JOIN voiceprint_embeddings AS embeddings ON embeddings.sample_id = samples.id
            GROUP BY speakers.id
            ORDER BY speakers.name
            """
        ).fetchall()
    return [_speaker_row(row) for row in rows]


def list_voiceprint_samples(speaker: str, db_path: Path | None = None) -> list[VoiceprintSampleRow]:
    """
    List samples for one speaker name or id.

    Args:
        speaker: Speaker name or speaker id.
        db_path: Optional SQLite path.

    Returns:
        Voiceprint sample rows.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        speaker_row = _find_speaker(connection, speaker)
        if speaker_row is None:
            return []
        rows = connection.execute(
            """
            SELECT samples.id, speakers.id AS speaker_id, speakers.name,
                   samples.project_id, samples.project_speaker_id,
                   samples.clip_path, samples.clip_rel_path, samples.clip_sha256,
                   samples.source_begin_time_ms, samples.source_end_time_ms,
                   samples.transcript_text
            FROM voiceprint_samples AS samples
            JOIN voiceprint_speakers AS speakers ON speakers.id = samples.speaker_id
            WHERE speakers.id = ?
            ORDER BY samples.project_id, samples.source_begin_time_ms
            """,
            (speaker_row.speaker_id,),
        ).fetchall()
    return [_sample_row(row) for row in rows]


def list_all_voiceprint_samples(db_path: Path | None = None) -> list[VoiceprintSampleRow]:
    """
    List all stored voiceprint samples.

    Args:
        db_path: Optional SQLite path.

    Returns:
        All voiceprint sample rows.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT samples.id, speakers.id AS speaker_id, speakers.name,
                   samples.project_id, samples.project_speaker_id,
                   samples.clip_path, samples.clip_rel_path, samples.clip_sha256,
                   samples.source_begin_time_ms, samples.source_end_time_ms,
                   samples.transcript_text
            FROM voiceprint_samples AS samples
            JOIN voiceprint_speakers AS speakers ON speakers.id = samples.speaker_id
            ORDER BY speakers.name, samples.project_id, samples.source_begin_time_ms
            """
        ).fetchall()
    return [_sample_row(row) for row in rows]


def list_voiceprint_samples_for_project(project_id: str, db_path: Path | None = None) -> list[VoiceprintSampleRow]:
    """
    List voiceprint samples captured from one project.

    Args:
        project_id: Project id stored in the sample registry.
        db_path: Optional SQLite path.

    Returns:
        Matching voiceprint sample rows.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT samples.id, speakers.id AS speaker_id, speakers.name,
                   samples.project_id, samples.project_speaker_id,
                   samples.clip_path, samples.clip_rel_path, samples.clip_sha256,
                   samples.source_begin_time_ms, samples.source_end_time_ms,
                   samples.transcript_text
            FROM voiceprint_samples AS samples
            JOIN voiceprint_speakers AS speakers ON speakers.id = samples.speaker_id
            WHERE samples.project_id = ?
            ORDER BY samples.project_speaker_id, samples.source_begin_time_ms
            """,
            (project_id,),
        ).fetchall()
    return [_sample_row(row) for row in rows]


def list_embedded_sample_ids(model: str, db_path: Path | None = None) -> set[int]:
    """
    List sample ids that already have an embedding for a model.

    Args:
        model: Embedding model key.
        db_path: Optional SQLite path.

    Returns:
        Embedded sample ids.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return set()
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        rows = connection.execute("SELECT sample_id FROM voiceprint_embeddings WHERE model = ?", (model,)).fetchall()
    return {int(row["sample_id"]) for row in rows}


def upsert_voiceprint_embedding(sample_id: int, model: str, vector: list[float], db_path: Path | None = None) -> None:
    """
    Store one voiceprint embedding.

    Args:
        sample_id: Voiceprint sample id.
        model: Embedding model key.
        vector: Embedding vector.
        db_path: Optional SQLite path.
    """
    database_path = _resolve_db_path(db_path)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO voiceprint_embeddings (sample_id, model, dimension, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sample_id, model) DO UPDATE SET
              dimension = excluded.dimension,
              vector = excluded.vector,
              created_at = excluded.created_at
            """,
            (sample_id, model, len(vector), _pack_vector(vector), _now_iso()),
        )


def list_voiceprint_embeddings(model: str, db_path: Path | None = None) -> list[VoiceprintEmbeddingRow]:
    """
    List stored embeddings for a model.

    Args:
        model: Embedding model key.
        db_path: Optional SQLite path.

    Returns:
        Embedding rows.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        rows = connection.execute(_embedding_rows_sql(), (model,)).fetchall()
    return [_embedding_row(row) for row in rows]


def delete_voiceprint_sample(
    speaker: str,
    sample_number: int,
    *,
    db_path: Path | None = None,
    delete_clip: bool = True,
) -> DeletedVoiceprintSample:
    """
    Delete one numbered sample for a speaker.

    Args:
        speaker: Speaker name or speaker id.
        sample_number: One-based sample number from ``list_voiceprint_samples``.
        db_path: Optional SQLite path.
        delete_clip: Whether to delete the WAV file.

    Returns:
        Deleted sample summary.
    """
    rows = list_voiceprint_samples(speaker, db_path)
    row = _select_sample_number(rows, sample_number, speaker)
    database_path = _resolve_db_path(db_path)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        connection.execute("DELETE FROM voiceprint_samples WHERE id = ?", (row.sample_id,))
        _delete_empty_speaker(connection, row.speaker_id)
    return _deleted_sample(row, delete_clip)


def delete_voiceprint_speaker(
    speaker: str,
    *,
    db_path: Path | None = None,
    delete_clips: bool = True,
) -> list[DeletedVoiceprintSample]:
    """
    Delete one speaker and all of their samples.

    Args:
        speaker: Speaker name or speaker id.
        db_path: Optional SQLite path.
        delete_clips: Whether to delete WAV files.

    Returns:
        Deleted sample summaries.
    """
    rows = list_voiceprint_samples(speaker, db_path)
    if not rows:
        raise LookupError(f"No voiceprint samples found for: {speaker}")
    database_path = _resolve_db_path(db_path)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        connection.execute("DELETE FROM voiceprint_speakers WHERE id = ?", (rows[0].speaker_id,))
    return [_deleted_sample(row, delete_clips) for row in rows]


def _resolve_db_path(db_path: Path | None) -> Path:
    """
    Resolve the database path.

    Args:
        db_path: Optional SQLite path.

    Returns:
        Absolute database path.
    """
    return (db_path or get_default_voiceprint_db_path()).expanduser().resolve()


def _resolve_store_dir(store_dir: Path | None) -> Path:
    """
    Resolve a voiceprint store directory.

    Args:
        store_dir: Optional voiceprint store directory.

    Returns:
        Absolute store directory.
    """
    return (store_dir or get_default_voiceprint_store_dir()).expanduser().resolve()


def _configure_connection(connection: sqlite3.Connection) -> None:
    """
    Configure SQLite connection behavior.

    Args:
        connection: SQLite connection.
    """
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """
    Ensure the voiceprint SQLite schema exists.

    Args:
        connection: SQLite connection.
    """
    connection.executescript(SCHEMA_SQL)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _upsert_sample(connection: sqlite3.Connection, sample: StoredVoiceprintSample) -> None:
    """
    Upsert one voiceprint sample.

    Args:
        connection: SQLite connection.
        sample: Sample metadata to store.
    """
    now = _now_iso()
    speaker_id = _speaker_id_for_sample(connection, sample, now)
    connection.execute(UPSERT_SAMPLE_SQL, _sample_values(sample, speaker_id, now))


def _speaker_id_for_sample(
    connection: sqlite3.Connection,
    sample: StoredVoiceprintSample,
    now: str,
) -> int:
    """
    Resolve the person id for a stored sample.

    Args:
        connection: SQLite connection.
        sample: Sample metadata.
        now: Current timestamp.

    Returns:
        Stable voiceprint person id.
    """
    if sample.person_id is None:
        return _upsert_speaker(connection, sample.speaker_name, now)
    row = _speaker_by_id(connection, sample.person_id)
    if row is None:
        raise LookupError(f"No voiceprint person found for id: {sample.person_id}")
    return sample.person_id


def _sample_values(sample: StoredVoiceprintSample, speaker_id: int, now: str) -> tuple[object, ...]:
    """
    Build SQLite values for one voiceprint sample.

    Args:
        sample: Sample metadata.
        speaker_id: Stored speaker id.
        now: Current timestamp.

    Returns:
        Values for ``UPSERT_SAMPLE_SQL``.
    """
    return (
        speaker_id,
        sample.project_id,
        str(sample.project_path.expanduser().resolve()),
        sample.project_speaker_id,
        str(sample.source_path.expanduser().resolve()),
        str(sample.clip_path.expanduser().resolve()),
        sample.clip_rel_path,
        _sha256_file(sample.clip_path),
        sample.source_begin_time_ms,
        sample.source_end_time_ms,
        sample.clip_begin_time_ms,
        sample.clip_end_time_ms,
        sample.transcript_text,
        now,
        now,
    )


def _upsert_speaker(connection: sqlite3.Connection, name: str, now: str) -> int:
    """
    Upsert one speaker by normalized name.

    Args:
        connection: SQLite connection.
        name: Speaker display name.
        now: Current timestamp.

    Returns:
        Stored speaker id.
    """
    normalized = _normalize_name(name)
    connection.execute(
        """
        INSERT INTO voiceprint_speakers (name, normalized_name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(normalized_name) DO UPDATE SET
          name = excluded.name,
          updated_at = excluded.updated_at
        """,
        (name, normalized, now, now),
    )
    row = connection.execute(
        "SELECT id FROM voiceprint_speakers WHERE normalized_name = ?",
        (normalized,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to store voiceprint speaker: {name}")
    return int(row["id"])


def _find_speaker(connection: sqlite3.Connection, speaker: str) -> VoiceprintSpeakerRow | None:
    """
    Find a stored speaker by id first, then by normalized name.

    Args:
        connection: SQLite connection.
        speaker: Speaker name or speaker id.

    Returns:
        Matching speaker row, or ``None`` when absent.
    """
    speaker_id = _parse_speaker_id(speaker)
    if speaker_id is not None:
        row = _speaker_by_id(connection, speaker_id)
        if row is not None:
            return row
    return _speaker_by_name(connection, speaker)


def _speaker_by_id(connection: sqlite3.Connection, speaker_id: int) -> VoiceprintSpeakerRow | None:
    """
    Find one speaker by database id.

    Args:
        connection: SQLite connection.
        speaker_id: Speaker database id.

    Returns:
        Matching speaker row, or ``None`` when absent.
    """
    row = connection.execute(
        """
        SELECT speakers.id, speakers.name,
               COUNT(DISTINCT samples.id) AS sample_count,
               COUNT(DISTINCT samples.project_id) AS project_count,
               COUNT(DISTINCT embeddings.sample_id) AS embedded_sample_count,
               COUNT(DISTINCT embeddings.model) AS embedding_model_count,
               MAX(samples.updated_at) AS updated_at
        FROM voiceprint_speakers AS speakers
        LEFT JOIN voiceprint_samples AS samples ON samples.speaker_id = speakers.id
        LEFT JOIN voiceprint_embeddings AS embeddings ON embeddings.sample_id = samples.id
        WHERE speakers.id = ?
        GROUP BY speakers.id
        """,
        (speaker_id,),
    ).fetchone()
    return _optional_speaker_row(row)


def _speaker_by_name(connection: sqlite3.Connection, name: str) -> VoiceprintSpeakerRow | None:
    """
    Find one speaker by normalized display name.

    Args:
        connection: SQLite connection.
        name: Speaker display name.

    Returns:
        Matching speaker row, or ``None`` when absent.
    """
    row = connection.execute(
        """
        SELECT speakers.id, speakers.name,
               COUNT(DISTINCT samples.id) AS sample_count,
               COUNT(DISTINCT samples.project_id) AS project_count,
               COUNT(DISTINCT embeddings.sample_id) AS embedded_sample_count,
               COUNT(DISTINCT embeddings.model) AS embedding_model_count,
               MAX(samples.updated_at) AS updated_at
        FROM voiceprint_speakers AS speakers
        LEFT JOIN voiceprint_samples AS samples ON samples.speaker_id = speakers.id
        LEFT JOIN voiceprint_embeddings AS embeddings ON embeddings.sample_id = samples.id
        WHERE speakers.normalized_name = ?
        GROUP BY speakers.id
        """,
        (_normalize_name(name),),
    ).fetchone()
    return _optional_speaker_row(row)


def _optional_speaker_row(row: sqlite3.Row | None) -> VoiceprintSpeakerRow | None:
    """
    Convert an optional SQLite row to a speaker dataclass.

    Args:
        row: Optional SQLite row.

    Returns:
        Speaker row, or ``None``.
    """
    if row is None:
        return None
    return _speaker_row(row)


def _parse_speaker_id(speaker: str) -> int | None:
    """
    Parse a positive speaker id from user input.

    Args:
        speaker: Speaker CLI argument.

    Returns:
        Positive speaker id, or ``None`` when the input is not an id.
    """
    stripped = speaker.strip()
    if not stripped.isdecimal():
        return None
    speaker_id = int(stripped)
    return speaker_id if speaker_id > 0 else None


def _sample_row(row: sqlite3.Row) -> VoiceprintSampleRow:
    """
    Convert a SQLite row to a sample dataclass.

    Args:
        row: SQLite row.

    Returns:
        Voiceprint sample row.
    """
    return VoiceprintSampleRow(
        sample_id=int(row["id"]),
        speaker_id=int(row["speaker_id"]),
        speaker_name=str(row["name"]),
        project_id=str(row["project_id"]),
        project_speaker_id=int(row["project_speaker_id"]),
        clip_path=Path(str(row["clip_path"])),
        clip_rel_path=str(row["clip_rel_path"]),
        clip_sha256=str(row["clip_sha256"]),
        source_begin_time_ms=int(row["source_begin_time_ms"]),
        source_end_time_ms=int(row["source_end_time_ms"]),
        transcript_text=str(row["transcript_text"]),
    )


def _speaker_row(row: sqlite3.Row) -> VoiceprintSpeakerRow:
    """
    Convert a SQLite row to a speaker summary dataclass.

    Args:
        row: SQLite row.

    Returns:
        Voiceprint speaker summary row.
    """
    updated_at = row["updated_at"]
    return VoiceprintSpeakerRow(
        speaker_id=int(row["id"]),
        name=str(row["name"]),
        sample_count=int(row["sample_count"]),
        project_count=int(row["project_count"]),
        embedded_sample_count=int(row["embedded_sample_count"]),
        embedding_model_count=int(row["embedding_model_count"]),
        updated_at=str(updated_at) if updated_at is not None else None,
    )


def _embedding_rows_sql() -> str:
    """
    Return SQL for joined embedding rows.

    Returns:
        SQL query.
    """
    return """
        SELECT samples.id AS sample_id, speakers.id AS speaker_id, speakers.name, samples.clip_path,
               embeddings.model, embeddings.vector
        FROM voiceprint_embeddings AS embeddings
        JOIN voiceprint_samples AS samples ON samples.id = embeddings.sample_id
        JOIN voiceprint_speakers AS speakers ON speakers.id = samples.speaker_id
        WHERE embeddings.model = ?
        ORDER BY speakers.name, samples.project_id, samples.source_begin_time_ms
    """


def _embedding_row(row: sqlite3.Row) -> VoiceprintEmbeddingRow:
    """
    Convert a SQLite row to an embedding dataclass.

    Args:
        row: SQLite row.

    Returns:
        Voiceprint embedding row.
    """
    return VoiceprintEmbeddingRow(
        sample_id=int(row["sample_id"]),
        speaker_id=int(row["speaker_id"]),
        speaker_name=str(row["name"]),
        clip_path=Path(str(row["clip_path"])),
        model=str(row["model"]),
        vector=_unpack_vector(bytes(row["vector"])),
    )


def _pack_vector(vector: list[float]) -> bytes:
    """
    Pack a float vector for SQLite storage.

    Args:
        vector: Embedding vector.

    Returns:
        Binary float payload.
    """
    if not vector:
        raise ValueError("Embedding vector must not be empty.")
    return struct.pack(f"<{len(vector)}f", *[float(item) for item in vector])


def _unpack_vector(payload: bytes) -> list[float]:
    """
    Unpack a float vector from SQLite storage.

    Args:
        payload: Binary float payload.

    Returns:
        Embedding vector.
    """
    if len(payload) % 4 != 0:
        raise ValueError("Embedding vector payload is not float32-aligned.")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def _select_sample_number(rows: list[VoiceprintSampleRow], sample_number: int, name: str) -> VoiceprintSampleRow:
    """
    Select one sample by one-based display number.

    Args:
        rows: Samples in display order.
        sample_number: One-based sample number.
        name: Speaker name for error messages.

    Returns:
        Selected sample row.
    """
    if sample_number < 1 or sample_number > len(rows):
        raise IndexError(f"Sample {sample_number} is out of range for {name}. Available: {len(rows)}.")
    return rows[sample_number - 1]


def _delete_empty_speaker(connection: sqlite3.Connection, speaker_id: int) -> None:
    """
    Delete a speaker row if it no longer owns samples.

    Args:
        connection: SQLite connection.
        speaker_id: Speaker database id.
    """
    row = connection.execute(
        """
        SELECT speakers.id, COUNT(samples.id) AS sample_count
        FROM voiceprint_speakers AS speakers
        LEFT JOIN voiceprint_samples AS samples ON samples.speaker_id = speakers.id
        WHERE speakers.id = ?
        GROUP BY speakers.id
        """,
        (speaker_id,),
    ).fetchone()
    if row is not None and int(row["sample_count"]) == 0:
        connection.execute("DELETE FROM voiceprint_speakers WHERE id = ?", (int(row["id"]),))


def _deleted_sample(row: VoiceprintSampleRow, delete_clip: bool) -> DeletedVoiceprintSample:
    """
    Delete a clip file when requested and return a summary.

    Args:
        row: Deleted database row.
        delete_clip: Whether to delete the clip file.

    Returns:
        Deleted sample summary.
    """
    clip_deleted = _delete_clip_file(row.clip_path) if delete_clip else False
    return DeletedVoiceprintSample(row.sample_id, row.speaker_id, row.speaker_name, row.clip_path, clip_deleted)


def _delete_clip_file(path: Path) -> bool:
    """
    Delete one clip file without touching parent directories.

    Args:
        path: Clip path.

    Returns:
        Whether a file was deleted.
    """
    if not path.exists():
        return False
    path.unlink()
    return True


def _normalize_name(name: str) -> str:
    """
    Normalize a speaker name for identity matching.

    Args:
        name: Speaker name.

    Returns:
        Normalized name.
    """
    normalized = " ".join(name.strip().split()).casefold()
    if not normalized:
        raise ValueError("Speaker name must not be empty.")
    return normalized


def _sha256_file(path: Path) -> str:
    """
    Hash a voiceprint clip without loading it all into memory.

    Args:
        path: Clip path.

    Returns:
        SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    """
    Return the current local timestamp.

    Returns:
        ISO timestamp.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")
