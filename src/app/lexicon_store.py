"""Cross-project vocabulary correction store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import get_data_dir
from app.correction_hotwords import AsrHotword, hotwords_from_terms
from app.lexicon_models import (
    AsrVocabularyState,
    LexiconAlias,
    LexiconCorrectionRule,
    LexiconContext,
    LexiconContextRow,
    LexiconStats,
    LexiconTerm,
    LexiconTermDetail,
)
from app.object_ids import new_prefixed_id

LEXICON_SCHEMA_VERSION = 2
LEXICON_TERM_ID_PREFIX = "lex-"
LEXICON_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO metadata(key, value)
VALUES ('schema_version', '2');

CREATE TABLE IF NOT EXISTS terms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  public_id TEXT UNIQUE,
  canonical TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_terms_category ON terms(category);
CREATE INDEX IF NOT EXISTS idx_terms_status ON terms(status);

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

CREATE TABLE IF NOT EXISTS asr_hotword_vocabularies (
  target_model TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  vocabulary_hash TEXT NOT NULL,
  vocabulary_id TEXT NOT NULL,
  hotword_count INTEGER NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(target_model, endpoint)
);
"""


def default_lexicon_db_path() -> Path:
    """
    Return the default XDG lexicon database path.

    Returns:
        SQLite database path.
    """
    return get_data_dir() / "lexicon" / "lexicon.sqlite"


def record_lexicon_contexts(
    contexts: list[LexiconContext], *, db_path: Path | None = None
) -> int:
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


def upsert_lexicon_term(
    *,
    canonical: str,
    category: str,
    description: str = "",
    aliases: tuple[str, ...] = (),
    status: str = "active",
    db_path: Path | None = None,
) -> LexiconTermDetail:
    """
    Insert or update one canonical lexicon term.

    Args:
        canonical: Canonical term text.
        category: Term category, for example ``person`` or ``system``.
        description: Optional human note.
        aliases: Optional aliases or common ASR mistakes.
        status: Term status.
        db_path: Optional database path override.

    Returns:
        Saved term detail.
    """
    canonical = _require_text(canonical, "canonical")
    category = _require_text(category, "category")
    status = _validate_status(status)
    database_path = _database_path(db_path, create_parent=True)
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        term_id = _upsert_term(connection, canonical, category, description, status)
        for alias in aliases:
            cleaned_alias = alias.strip()
            if cleaned_alias:
                _upsert_alias(connection, term_id, cleaned_alias)
    return get_lexicon_term(canonical, db_path=database_path)


def list_lexicon_terms(
    *,
    db_path: Path | None = None,
    status: str = "active",
    category: str | None = None,
    query: str | None = None,
    limit: int = 100,
) -> list[LexiconTerm]:
    """
    List local lexicon terms.

    Args:
        db_path: Optional database path override.
        status: ``active``, ``inactive``, or ``all``.
        category: Optional category filter.
        query: Optional canonical or alias substring.
        limit: Maximum rows.

    Returns:
        Matching terms.
    """
    database_path = _database_path(db_path)
    if not database_path.exists():
        return []
    where, params = _term_filters(status=status, category=category, query=query)
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            f"""
            SELECT t.id, t.canonical, t.category, t.description, t.status,
                   t.public_id,
                   COUNT(DISTINCT a.id) AS alias_count,
                   COUNT(DISTINCT c.id) AS context_count,
                   t.created_at, t.updated_at
            FROM terms AS t
            LEFT JOIN aliases AS a ON a.term_id = t.id
            LEFT JOIN contexts AS c ON c.term_id = t.id
            {where}
            GROUP BY t.id
            ORDER BY t.updated_at DESC, t.canonical ASC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    return [_term_from_row(row) for row in rows]


def get_lexicon_term(
    term_ref: str | int,
    *,
    db_path: Path | None = None,
    context_limit: int = 20,
) -> LexiconTermDetail:
    """
    Return one term by id, canonical text, or alias.

    Args:
        term_ref: Term id, canonical text, or alias.
        db_path: Optional database path override.
        context_limit: Maximum context rows to include.

    Returns:
        Full term detail.
    """
    database_path = _database_path(db_path)
    if not database_path.exists():
        raise LookupError(f"Lexicon database does not exist: {database_path}")
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        term_id = _resolve_term_id(connection, term_ref)
        term = _load_term(connection, term_id)
        aliases = _load_aliases(connection, term_id)
        contexts = _load_contexts(connection, term_id, context_limit)
    return LexiconTermDetail(term, aliases, contexts)


def delete_lexicon_term(
    term_ref: str | int,
    *,
    db_path: Path | None = None,
    permanent: bool = False,
) -> LexiconTermDetail:
    """
    Deactivate or permanently delete a local lexicon term.

    Args:
        term_ref: Term id, canonical text, or alias.
        db_path: Optional database path override.
        permanent: Physically delete the row when true.

    Returns:
        Deleted or deactivated term detail.
    """
    database_path = _database_path(db_path)
    if not database_path.exists():
        raise LookupError(f"Lexicon database does not exist: {database_path}")
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        term_id = _resolve_term_id(connection, term_ref)
        detail = LexiconTermDetail(
            _load_term(connection, term_id), _load_aliases(connection, term_id), ()
        )
        if permanent:
            connection.execute("DELETE FROM terms WHERE id = ?", (term_id,))
        else:
            connection.execute(
                "UPDATE terms SET status = 'inactive', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (term_id,),
            )
    return detail


def lexicon_stats(*, db_path: Path | None = None) -> LexiconStats:
    """
    Return aggregate local lexicon statistics.

    Args:
        db_path: Optional database path override.

    Returns:
        Local lexicon statistics.
    """
    database_path = _database_path(db_path)
    if not database_path.exists():
        return LexiconStats(0, 0, 0, 0, 0, 0)
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        active = _count(
            connection, "SELECT COUNT(*) FROM terms WHERE status = 'active'"
        )
        inactive = _count(
            connection, "SELECT COUNT(*) FROM terms WHERE status != 'active'"
        )
        aliases = _count(connection, "SELECT COUNT(*) FROM aliases")
        contexts = _count(connection, "SELECT COUNT(*) FROM contexts")
        vocabularies = _count(
            connection, "SELECT COUNT(*) FROM asr_hotword_vocabularies"
        )
    hotwords = len(list_asr_hotwords(db_path=database_path))
    return LexiconStats(active, inactive, aliases, contexts, hotwords, vocabularies)


def export_lexicon_payload(
    *, db_path: Path | None = None, include_inactive: bool = True
) -> dict[str, Any]:
    """
    Build a portable local lexicon JSON payload.

    Args:
        db_path: Optional database path override.
        include_inactive: Include inactive terms when true.

    Returns:
        JSON-ready lexicon payload.
    """
    status = "all" if include_inactive else "active"
    terms = []
    for term in list_lexicon_terms(db_path=db_path, status=status, limit=10_000):
        detail = get_lexicon_term(term.canonical, db_path=db_path, context_limit=10_000)
        terms.append(_term_export_payload(detail))
    return {"schema_version": LEXICON_SCHEMA_VERSION, "terms": terms}


def import_lexicon_payload(
    payload: dict[str, Any], *, db_path: Path | None = None
) -> int:
    """
    Import a portable local lexicon JSON payload.

    Args:
        payload: Payload previously produced by ``export_lexicon_payload``.
        db_path: Optional database path override.

    Returns:
        Number of imported terms.
    """
    terms = payload.get("terms")
    if not isinstance(terms, list):
        raise ValueError("Lexicon import payload must contain a terms list.")
    database_path = _database_path(db_path, create_parent=True)
    imported = 0
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        for item in terms:
            if not isinstance(item, dict):
                continue
            term_id = _import_term(connection, item)
            _import_aliases(connection, term_id, item.get("aliases"))
            _import_contexts(connection, term_id, item.get("contexts"))
            imported += 1
    return imported


def list_asr_hotwords(
    *, db_path: Path | None = None, limit: int = 500
) -> list[AsrHotword]:
    """
    Build ASR hotwords from accepted cross-project lexicon terms.

    Args:
        db_path: Optional database path override.
        limit: Maximum number of terms to export.

    Returns:
        Hotwords suitable for DashScope ASR biasing.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT canonical, category
            FROM terms
            WHERE status = 'active'
            ORDER BY updated_at DESC, canonical ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return hotwords_from_terms([(str(row[0]), str(row[1])) for row in rows])


def list_lexicon_known_texts(*, db_path: Path | None = None) -> frozenset[str]:
    """
    Return the lowercased set of every active term's canonical text and aliases.

    Used by the polish guard to whitelist verified ASCII restorations: a polish
    that turns a homophone into a name already in the lexicon (底码 -> Dima) is a
    correction, not a hallucination.

    Args:
        db_path: Optional database path override.

    Returns:
        Lowercased canonical and alias strings for active terms.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return frozenset()
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        texts = {
            str(row[0]).lower()
            for row in connection.execute(
                "SELECT canonical FROM terms WHERE status = 'active'"
            )
        }
        texts.update(
            str(row[0]).lower()
            for row in connection.execute(
                "SELECT a.alias FROM aliases AS a "
                "JOIN terms AS t ON t.id = a.term_id "
                "WHERE t.status = 'active'"
            )
        )
    return frozenset(texts)


def list_lexicon_correction_rules(
    *, db_path: Path | None = None, limit: int = 1000
) -> list[LexiconCorrectionRule]:
    """
    Return active local correction rules from accepted lexicon knowledge.

    Args:
        db_path: Optional database path override.
        limit: Maximum rules to return.

    Returns:
        Replacement rules ordered for deterministic longest-match application.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        context_rows = connection.execute(
            """
            SELECT c.wrong_text, c.corrected_text, c.left_context, c.right_context,
                   t.canonical, t.category
            FROM contexts AS c
            JOIN terms AS t ON t.id = c.term_id
            WHERE t.status = 'active'
              AND c.wrong_text != ''
              AND c.corrected_text != ''
              AND c.wrong_text != c.corrected_text
            ORDER BY length(c.wrong_text) DESC, c.created_at DESC, c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        alias_rows = connection.execute(
            """
            SELECT a.alias, t.canonical, t.canonical, t.category
            FROM aliases AS a
            JOIN terms AS t ON t.id = a.term_id
            WHERE t.status = 'active'
              AND a.alias_type = 'asr_error'
              AND a.alias != ''
              AND a.alias != t.canonical
            ORDER BY length(a.alias) DESC, a.updated_at DESC, a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return _lexicon_correction_rules_from_rows(context_rows, alias_rows, limit)


def get_asr_vocabulary_state(
    *,
    target_model: str,
    endpoint: str,
    db_path: Path | None = None,
) -> AsrVocabularyState | None:
    """
    Return cached DashScope vocabulary state for a model and endpoint.

    Args:
        target_model: DashScope ASR model.
        endpoint: DashScope base endpoint.
        db_path: Optional database path override.

    Returns:
        Cached state, or None.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return None
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT target_model, endpoint, vocabulary_hash, vocabulary_id, hotword_count, updated_at
            FROM asr_hotword_vocabularies
            WHERE target_model = ? AND endpoint = ?
            """,
            (target_model, endpoint),
        ).fetchone()
    return _asr_state_from_row(row)


def list_asr_vocabulary_states(
    *, db_path: Path | None = None
) -> list[AsrVocabularyState]:
    """
    Return all cached DashScope vocabulary states.

    Args:
        db_path: Optional database path override.

    Returns:
        Cached states ordered by model and endpoint.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return []
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT target_model, endpoint, vocabulary_hash, vocabulary_id, hotword_count, updated_at
            FROM asr_hotword_vocabularies
            ORDER BY target_model ASC, endpoint ASC
            """
        ).fetchall()
    return [state for row in rows if (state := _asr_state_from_row(row)) is not None]


def save_asr_vocabulary_state(
    state: AsrVocabularyState,
    *,
    db_path: Path | None = None,
) -> None:
    """
    Persist cached DashScope vocabulary state.

    Args:
        state: Vocabulary state to save.
        db_path: Optional database path override.

    Returns:
        None.
    """
    database_path = db_path or default_lexicon_db_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO asr_hotword_vocabularies(
              target_model, endpoint, vocabulary_hash, vocabulary_id, hotword_count
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(target_model, endpoint) DO UPDATE SET
              vocabulary_hash = excluded.vocabulary_hash,
              vocabulary_id = excluded.vocabulary_id,
              hotword_count = excluded.hotword_count,
              updated_at = CURRENT_TIMESTAMP
            """,
            (
                state.target_model,
                state.endpoint,
                state.vocabulary_hash,
                state.vocabulary_id,
                state.hotword_count,
            ),
        )


def delete_asr_vocabulary_state(
    *,
    target_model: str,
    endpoint: str,
    vocabulary_id: str | None = None,
    db_path: Path | None = None,
) -> AsrVocabularyState | None:
    """
    Delete one cached DashScope vocabulary state.

    Args:
        target_model: DashScope ASR model.
        endpoint: DashScope base endpoint.
        vocabulary_id: Optional guard; when set, only delete matching ids.
        db_path: Optional database path override.

    Returns:
        The deleted state, or ``None`` when no matching state exists.
    """
    database_path = db_path or default_lexicon_db_path()
    if not database_path.exists():
        return None
    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT target_model, endpoint, vocabulary_hash, vocabulary_id, hotword_count, updated_at
            FROM asr_hotword_vocabularies
            WHERE target_model = ? AND endpoint = ?
            """,
            (target_model, endpoint),
        ).fetchone()
        state = _asr_state_from_row(row)
        if state is None:
            return None
        if vocabulary_id is not None and state.vocabulary_id != vocabulary_id:
            return None
        connection.execute(
            """
            DELETE FROM asr_hotword_vocabularies
            WHERE target_model = ? AND endpoint = ?
            """,
            (target_model, endpoint),
        )
    return state


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the lexicon schema if needed."""
    connection.executescript(LEXICON_SCHEMA_SQL)
    _migrate_terms_public_id(connection)
    connection.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(LEXICON_SCHEMA_VERSION),),
    )


def _migrate_terms_public_id(connection: sqlite3.Connection) -> None:
    """Ensure every lexicon term has a stable prefixed public id."""
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(terms)").fetchall()
    }
    if "public_id" not in columns:
        connection.execute("ALTER TABLE terms ADD COLUMN public_id TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_terms_public_id ON terms(public_id)"
    )
    rows = connection.execute(
        "SELECT id FROM terms WHERE public_id IS NULL OR public_id = ''"
    ).fetchall()
    for row in rows:
        connection.execute(
            "UPDATE terms SET public_id = ? WHERE id = ?",
            (_new_public_id(connection), int(row[0])),
        )


def _new_public_id(connection: sqlite3.Connection) -> str:
    """Generate a collision-free public id for one lexicon term."""
    return new_prefixed_id(
        LEXICON_TERM_ID_PREFIX,
        lambda public_id: (
            connection.execute(
                "SELECT 1 FROM terms WHERE public_id = ?", (public_id,)
            ).fetchone()
            is not None
        ),
    )


def _public_id_for_insert(connection: sqlite3.Connection, requested: str | None) -> str:
    """Return an insertable public id, preserving valid imported ids when possible."""
    public_id = (requested or "").strip().lower()
    if _valid_public_id(public_id):
        row = connection.execute(
            "SELECT 1 FROM terms WHERE public_id = ?", (public_id,)
        ).fetchone()
        if row is None:
            return public_id
    return _new_public_id(connection)


def _valid_public_id(public_id: str) -> bool:
    """Return whether a public id follows the lexicon term id format."""
    suffix = public_id.removeprefix(LEXICON_TERM_ID_PREFIX)
    return (
        public_id.startswith(LEXICON_TERM_ID_PREFIX)
        and len(suffix) == 16
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _asr_state_from_row(row) -> AsrVocabularyState | None:
    """Build a vocabulary state from one SQLite row."""
    if row is None:
        return None
    return AsrVocabularyState(
        target_model=str(row[0]),
        endpoint=str(row[1]),
        vocabulary_hash=str(row[2]),
        vocabulary_id=str(row[3]),
        hotword_count=int(row[4]),
        updated_at=str(row[5]) if len(row) > 5 and row[5] is not None else None,
    )


def _database_path(db_path: Path | None, *, create_parent: bool = False) -> Path:
    """Resolve the lexicon database path."""
    database_path = db_path or default_lexicon_db_path()
    if create_parent:
        database_path.parent.mkdir(parents=True, exist_ok=True)
    return database_path


def _require_text(value: str, field: str) -> str:
    """Return a stripped required text field."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Lexicon {field} is required.")
    return stripped


def _validate_status(status: str) -> str:
    """Return a normalized lexicon term status."""
    normalized = status.strip().lower()
    if normalized not in {"active", "inactive"}:
        raise ValueError("Lexicon term status must be active or inactive.")
    return normalized


def _term_filters(
    *,
    status: str,
    category: str | None,
    query: str | None,
) -> tuple[str, tuple[object, ...]]:
    """Build the WHERE clause for term listing."""
    clauses = []
    params: list[object] = []
    normalized_status = status.strip().lower()
    if normalized_status != "all":
        clauses.append("t.status = ?")
        params.append(_validate_status(normalized_status))
    if category and category.strip():
        clauses.append("t.category = ?")
        params.append(category.strip())
    if query and query.strip():
        clauses.append(
            """
            (
              t.canonical LIKE ?
              OR t.public_id LIKE ?
              OR EXISTS (
                SELECT 1 FROM aliases AS qa
                WHERE qa.term_id = t.id AND qa.alias LIKE ?
              )
            )
            """
        )
        pattern = f"%{query.strip()}%"
        params.extend([pattern, pattern, pattern])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, tuple(params)


def _resolve_term_id(connection: sqlite3.Connection, term_ref: str | int) -> int:
    """Resolve a term id from a public id, numeric id, canonical text, or alias."""
    ref_text = str(term_ref).strip()
    if not ref_text:
        raise LookupError("Lexicon term reference is empty.")
    public_ref = ref_text.lower()
    if _valid_public_id(public_ref):
        row = connection.execute(
            "SELECT id FROM terms WHERE public_id = ?", (public_ref,)
        ).fetchone()
        if row is not None:
            return int(row[0])
    if ref_text.isdecimal():
        row = connection.execute(
            "SELECT id FROM terms WHERE id = ?", (int(ref_text),)
        ).fetchone()
        if row is not None:
            return int(row[0])
    rows = connection.execute(
        """
        SELECT id FROM terms WHERE canonical = ?
        UNION
        SELECT term_id FROM aliases WHERE alias = ?
        """,
        (ref_text, ref_text),
    ).fetchall()
    if not rows:
        raise LookupError(f"Lexicon term not found: {ref_text}")
    if len(rows) > 1:
        raise LookupError(f"Lexicon term reference is ambiguous: {ref_text}")
    return int(rows[0][0])


def _load_term(connection: sqlite3.Connection, term_id: int) -> LexiconTerm:
    """Load one summarized term row."""
    row = connection.execute(
        """
        SELECT t.id, t.canonical, t.category, t.description, t.status,
               t.public_id,
               COUNT(DISTINCT a.id), COUNT(DISTINCT c.id),
               t.created_at, t.updated_at
        FROM terms AS t
        LEFT JOIN aliases AS a ON a.term_id = t.id
        LEFT JOIN contexts AS c ON c.term_id = t.id
        WHERE t.id = ?
        GROUP BY t.id
        """,
        (term_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"Lexicon term not found: {term_id}")
    return _term_from_row(row)


def _load_aliases(
    connection: sqlite3.Connection, term_id: int
) -> tuple[LexiconAlias, ...]:
    """Load aliases for one term."""
    rows = connection.execute(
        """
        SELECT alias, alias_type, created_at, updated_at
        FROM aliases
        WHERE term_id = ?
        ORDER BY alias_type ASC, alias ASC
        """,
        (term_id,),
    ).fetchall()
    return tuple(
        LexiconAlias(str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in rows
    )


def _load_contexts(
    connection: sqlite3.Connection, term_id: int, limit: int
) -> tuple[LexiconContextRow, ...]:
    """Load recent contexts for one term."""
    rows = connection.execute(
        """
        SELECT wrong_text, corrected_text, left_context, right_context,
               speaker_name, project_id, sentence_id, source, created_at
        FROM contexts
        WHERE term_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (term_id, limit),
    ).fetchall()
    return tuple(_context_from_row(row) for row in rows)


def _count(connection: sqlite3.Connection, query: str) -> int:
    """Return the integer result of one count query."""
    row = connection.execute(query).fetchone()
    return int(row[0]) if row is not None else 0


def _term_from_row(row) -> LexiconTerm:
    """Build one term summary from a SQLite row."""
    return LexiconTerm(
        term_id=int(row[0]),
        public_id=str(row[5]),
        canonical=str(row[1]),
        category=str(row[2]),
        description=str(row[3]),
        status=str(row[4]),
        alias_count=int(row[6]),
        context_count=int(row[7]),
        created_at=str(row[8]),
        updated_at=str(row[9]),
    )


def _context_from_row(row) -> LexiconContextRow:
    """Build one context row from SQLite data."""
    return LexiconContextRow(
        wrong_text=str(row[0]),
        corrected_text=str(row[1]),
        left_context=str(row[2]),
        right_context=str(row[3]),
        speaker_name=str(row[4]) if row[4] is not None else None,
        project_id=str(row[5]),
        sentence_id=int(row[6]) if row[6] is not None else None,
        source=str(row[7]),
        created_at=str(row[8]),
    )


def _term_export_payload(detail: LexiconTermDetail) -> dict[str, Any]:
    """Return a portable JSON payload for one term."""
    return {
        "public_id": detail.term.public_id,
        "canonical": detail.term.canonical,
        "category": detail.term.category,
        "description": detail.term.description,
        "status": detail.term.status,
        "aliases": [_alias_payload(alias) for alias in detail.aliases],
        "contexts": [_context_payload(context) for context in detail.contexts],
    }


def _alias_payload(alias: LexiconAlias) -> dict[str, str]:
    """Return JSON-ready alias data."""
    return {
        "alias": alias.alias,
        "alias_type": alias.alias_type,
        "created_at": alias.created_at,
        "updated_at": alias.updated_at,
    }


def _context_payload(context: LexiconContextRow) -> dict[str, object]:
    """Return JSON-ready correction context data."""
    return {
        "wrong_text": context.wrong_text,
        "corrected_text": context.corrected_text,
        "left_context": context.left_context,
        "right_context": context.right_context,
        "speaker_name": context.speaker_name,
        "project_id": context.project_id,
        "sentence_id": context.sentence_id,
        "source": context.source,
        "created_at": context.created_at,
    }


def _lexicon_correction_rules_from_rows(
    context_rows: list[tuple],
    alias_rows: list[tuple],
    limit: int,
) -> list[LexiconCorrectionRule]:
    """Merge context and alias rows into unique correction rules."""
    rules: list[LexiconCorrectionRule] = []
    seen: set[tuple[str, str]] = set()
    for row in context_rows:
        _append_correction_rule(
            rules,
            seen,
            wrong_text=str(row[0]),
            corrected_text=str(row[1]),
            left_context=str(row[2]),
            right_context=str(row[3]),
            canonical=str(row[4]),
            category=str(row[5]),
            source="context",
            limit=limit,
        )
    for row in alias_rows:
        _append_correction_rule(
            rules,
            seen,
            wrong_text=str(row[0]),
            corrected_text=str(row[1]),
            left_context="",
            right_context="",
            canonical=str(row[2]),
            category=str(row[3]),
            source="alias",
            limit=limit,
        )
    return sorted(
        rules,
        key=lambda rule: (-len(rule.wrong_text), rule.wrong_text, rule.corrected_text),
    )


def _append_correction_rule(
    rules: list[LexiconCorrectionRule],
    seen: set[tuple[str, str]],
    *,
    wrong_text: str,
    corrected_text: str,
    left_context: str,
    right_context: str,
    canonical: str,
    category: str,
    source: str,
    limit: int,
) -> None:
    """Append one unique correction rule when it is usable."""
    if (
        len(rules) >= limit
        or not wrong_text
        or not corrected_text
        or wrong_text == corrected_text
    ):
        return
    key = (wrong_text, corrected_text)
    if key in seen:
        return
    seen.add(key)
    rules.append(
        LexiconCorrectionRule(
            wrong_text=wrong_text,
            corrected_text=corrected_text,
            left_context=left_context,
            right_context=right_context,
            canonical=canonical,
            category=category,
            source=source,
        )
    )


def _import_term(connection: sqlite3.Connection, item: dict[str, Any]) -> int:
    """Import one term object."""
    return _upsert_term(
        connection,
        _require_text(str(item.get("canonical") or ""), "canonical"),
        _require_text(str(item.get("category") or "unknown"), "category"),
        str(item.get("description") or ""),
        _validate_status(str(item.get("status") or "active")),
        str(item.get("public_id") or ""),
    )


def _import_aliases(
    connection: sqlite3.Connection, term_id: int, aliases: object
) -> None:
    """Import aliases for one term."""
    if not isinstance(aliases, list):
        return
    for item in aliases:
        alias = item.get("alias") if isinstance(item, dict) else item
        alias_type = item.get("alias_type") if isinstance(item, dict) else "asr_error"
        if isinstance(alias, str) and alias.strip():
            _upsert_alias(
                connection, term_id, alias.strip(), str(alias_type or "asr_error")
            )


def _import_contexts(
    connection: sqlite3.Connection, term_id: int, contexts: object
) -> None:
    """Import contexts for one term."""
    if not isinstance(contexts, list):
        return
    for item in contexts:
        if isinstance(item, dict):
            _insert_imported_context(connection, term_id, item)


def _insert_imported_context(
    connection: sqlite3.Connection, term_id: int, item: dict[str, Any]
) -> None:
    """Insert one imported context row."""
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
            str(item.get("wrong_text") or ""),
            str(item.get("corrected_text") or ""),
            str(item.get("left_context") or ""),
            str(item.get("right_context") or ""),
            item.get("speaker_name"),
            str(item.get("project_id") or "import"),
            item.get("sentence_id"),
            str(item.get("source") or "import"),
        ),
    )


def _upsert_term(
    connection: sqlite3.Connection,
    canonical: str,
    category: str,
    description: str | None = None,
    status: str | None = None,
    public_id: str | None = None,
) -> int:
    """Insert or refresh a canonical term."""
    row = connection.execute(
        "SELECT id FROM terms WHERE canonical = ?", (canonical,)
    ).fetchone()
    if row is not None:
        _update_term(connection, int(row[0]), category, description, status)
        return int(row[0])
    connection.execute(
        """
        INSERT INTO terms(public_id, canonical, category, description, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            _public_id_for_insert(connection, public_id),
            canonical,
            category,
            description or "",
            status or "active",
        ),
    )
    row = connection.execute(
        "SELECT id FROM terms WHERE canonical = ?", (canonical,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert lexicon term: {canonical}")
    return int(row[0])


def _update_term(
    connection: sqlite3.Connection,
    term_id: int,
    category: str,
    description: str | None,
    status: str | None,
) -> None:
    """Update one term without clobbering fields not provided by callers."""
    fields = ["category = ?", "updated_at = CURRENT_TIMESTAMP"]
    params: list[object] = [category]
    if description is not None:
        fields.insert(-1, "description = ?")
        params.append(description)
    if status is not None:
        fields.insert(-1, "status = ?")
        params.append(status)
    params.append(term_id)
    connection.execute(
        f"UPDATE terms SET {', '.join(fields)} WHERE id = ?", tuple(params)
    )


def _upsert_alias(
    connection: sqlite3.Connection,
    term_id: int,
    alias: str,
    alias_type: str = "asr_error",
) -> None:
    """Insert or refresh an alias for a term."""
    connection.execute(
        """
        INSERT INTO aliases(term_id, alias, alias_type)
        VALUES (?, ?, ?)
        ON CONFLICT(term_id, alias, alias_type) DO UPDATE SET
          updated_at = CURRENT_TIMESTAMP
        """,
        (term_id, alias, alias_type),
    )


def _insert_context(
    connection: sqlite3.Connection, term_id: int, context: LexiconContext
) -> None:
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
