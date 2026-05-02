"""Cross-project vocabulary correction store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.config import get_data_dir

LEXICON_SCHEMA_VERSION = 1
LEXICON_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO metadata(key, value)
VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS terms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_terms_category ON terms(category);

CREATE TABLE IF NOT EXISTS aliases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  term_id INTEGER NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  alias_type TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(term_id, alias, alias_type)
);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);

CREATE TABLE IF NOT EXISTS contexts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  term_id INTEGER NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
  wrong_text TEXT NOT NULL,
  corrected_text TEXT NOT NULL,
  left_context TEXT NOT NULL,
  right_context TEXT NOT NULL,
  speaker_name TEXT,
  project_id TEXT NOT NULL,
  sentence_id INTEGER,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_contexts_project ON contexts(project_id);
CREATE INDEX IF NOT EXISTS idx_contexts_term ON contexts(term_id);
"""


@dataclass(frozen=True, slots=True)
class LexiconContext:
    """One accepted vocabulary correction context."""

    canonical: str
    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str
    category: str
    speaker_name: str | None
    project_id: str
    sentence_id: int | None
    source: str


def default_lexicon_db_path() -> Path:
    """
    Return the default XDG lexicon database path.

    Returns:
        SQLite database path.
    """
    return get_data_dir() / "lexicon" / "lexicon.sqlite"


def record_lexicon_contexts(contexts: list[LexiconContext], *, db_path: Path | None = None) -> int:
    """
    Persist accepted vocabulary contexts.

    Args:
        contexts: Contexts inferred from user edits.
        db_path: Optional database path override.

    Returns:
        Number of context rows written.
    """
    if not contexts:
        return 0
    database_path = db_path or default_lexicon_db_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        for context in contexts:
            term_id = _upsert_term(connection, context.canonical, context.category)
            _upsert_alias(connection, term_id, context.wrong_text)
            _insert_context(connection, term_id, context)
    return len(contexts)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the lexicon schema if needed."""
    connection.executescript(LEXICON_SCHEMA_SQL)


def _upsert_term(connection: sqlite3.Connection, canonical: str, category: str) -> int:
    """Insert or refresh a canonical term."""
    connection.execute(
        """
        INSERT INTO terms(canonical, category)
        VALUES (?, ?)
        ON CONFLICT(canonical) DO UPDATE SET
          category = excluded.category,
          updated_at = CURRENT_TIMESTAMP
        """,
        (canonical, category),
    )
    row = connection.execute("SELECT id FROM terms WHERE canonical = ?", (canonical,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert lexicon term: {canonical}")
    return int(row[0])


def _upsert_alias(connection: sqlite3.Connection, term_id: int, alias: str) -> None:
    """Insert an ASR-error alias for a term."""
    connection.execute(
        """
        INSERT INTO aliases(term_id, alias, alias_type)
        VALUES (?, ?, 'asr_error')
        ON CONFLICT(term_id, alias, alias_type) DO UPDATE SET
          updated_at = CURRENT_TIMESTAMP
        """,
        (term_id, alias),
    )


def _insert_context(connection: sqlite3.Connection, term_id: int, context: LexiconContext) -> None:
    """Insert one accepted context row."""
    connection.execute(
        """
        INSERT INTO contexts(
          term_id, wrong_text, corrected_text, left_context, right_context,
          speaker_name, project_id, sentence_id, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            term_id,
            context.wrong_text,
            context.corrected_text,
            context.left_context,
            context.right_context,
            context.speaker_name,
            context.project_id,
            context.sentence_id,
            context.source,
        ),
    )
