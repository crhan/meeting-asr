"""DashScope ASR hotword synchronization from the correction lexicon."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import dashscope
from dashscope.audio.asr import VocabularyService

from app.config import Settings
from app.correction_hotwords import (
    AsrHotword,
    dashscope_vocabulary,
    hotword_hash,
    write_hotword_artifact,
)
from app.lexicon_store import (
    AsrVocabularyState,
    default_lexicon_db_path,
    get_asr_vocabulary_state,
    list_asr_hotwords,
    save_asr_vocabulary_state,
)

DEFAULT_HOTWORD_PREFIX = "mtgasr"
PREFIX_RE = re.compile(r"^[a-z0-9]{1,9}$")


class VocabularyClient(Protocol):
    """Protocol for DashScope vocabulary clients."""

    def create_vocabulary(
        self, target_model: str, prefix: str, vocabulary: list[dict]
    ) -> str:
        """Create a vocabulary and return its id."""

    def update_vocabulary(self, vocabulary_id: str, vocabulary: list[dict]) -> None:
        """Update an existing vocabulary."""

    def list_vocabularies(
        self, prefix=None, page_index: int = 0, page_size: int = 10
    ) -> list[dict]:
        """List remote vocabularies."""

    def query_vocabulary(self, vocabulary_id: str):
        """Query one remote vocabulary."""

    def delete_vocabulary(self, vocabulary_id: str) -> None:
        """Delete one remote vocabulary."""


@dataclass(frozen=True, slots=True)
class AsrHotwordSyncSummary:
    """Result of synchronizing ASR hotwords with DashScope."""

    db_path: Path
    target_model: str
    endpoint: str
    vocabulary_id: str | None
    hotword_count: int
    vocabulary_hash: str | None
    changed: bool
    dry_run: bool
    artifact_path: Path | None


@dataclass(frozen=True, slots=True)
class AsrHotwordResolution:
    """Vocabulary id resolved for one ASR request."""

    vocabulary_id: str | None
    source: str
    hotword_count: int = 0
    vocabulary_hash: str | None = None
    error: str | None = None
    # The hotword table this resolution represents, carried so callers can
    # snapshot exactly what backed the request without re-reading the lexicon.
    hotwords: tuple[AsrHotword, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AsrHotwordStatus:
    """Local hotword table and cached DashScope vocabulary state."""

    db_path: Path
    target_model: str
    endpoint: str
    hotword_count: int
    vocabulary_hash: str | None
    cache_status: str
    cached_state: AsrVocabularyState | None


def resolve_asr_hotwords(
    *,
    mode: str | None,
    settings: Settings,
    target_model: str,
    db_path: Path | None = None,
) -> AsrHotwordResolution:
    """
    Resolve the vocabulary id to use for one ASR task.

    Args:
        mode: ``auto``, ``off``, or an explicit vocabulary id.
        settings: Runtime settings.
        target_model: DashScope ASR model.
        db_path: Optional lexicon database path.

    Returns:
        Resolved vocabulary id plus the hotword table it represents. ``off``
        carries an empty table; every other mode carries the active lexicon
        hotwords so callers can snapshot exactly what backed the request.
    """
    value = (mode or "auto").strip()
    if value.lower() in {"off", "false", "none", "0"}:
        return AsrHotwordResolution(None, "off")
    database_path = db_path or default_lexicon_db_path()
    hotwords = tuple(list_asr_hotwords(db_path=database_path))
    if value.lower() == "auto":
        resolution = _auto_hotword_resolution(
            settings=settings, target_model=target_model, db_path=database_path
        )
    else:
        resolution = AsrHotwordResolution(value, "explicit")
    return replace(resolution, hotwords=hotwords)


def write_asr_hotword_artifact(path: Path, resolution: AsrHotwordResolution) -> Path:
    """
    Persist the hotword table backing one resolution to a project artifact.

    Args:
        path: Destination ``corrections/asr_hotwords.json`` path.
        resolution: Resolved hotwords for the ASR request.

    Returns:
        Written artifact path.
    """
    return write_hotword_artifact(path, list(resolution.hotwords))


def get_asr_hotword_status(
    *,
    settings: Settings,
    target_model: str,
    db_path: Path | None = None,
    limit: int = 500,
) -> AsrHotwordStatus:
    """
    Return local hotword hash and cached DashScope vocabulary state.

    Args:
        settings: Runtime settings.
        target_model: DashScope ASR model.
        db_path: Optional lexicon database path.
        limit: Maximum lexicon terms to inspect.

    Returns:
        Hotword status for one model and endpoint.
    """
    database_path = db_path or default_lexicon_db_path()
    hotwords = list_asr_hotwords(db_path=database_path, limit=limit)
    table_hash = hotword_hash(hotwords) if hotwords else None
    endpoint = settings.dashscope_base_url
    state = get_asr_vocabulary_state(
        target_model=target_model, endpoint=endpoint, db_path=database_path
    )
    cache_status = _cache_status(
        hotword_count=len(hotwords), table_hash=table_hash, state=state
    )
    return AsrHotwordStatus(
        database_path,
        target_model,
        endpoint,
        len(hotwords),
        table_hash,
        cache_status,
        state,
    )


def sync_asr_hotwords(
    *,
    settings: Settings,
    target_model: str,
    db_path: Path | None = None,
    prefix: str = DEFAULT_HOTWORD_PREFIX,
    force: bool = False,
    dry_run: bool = False,
    output: Path | None = None,
    limit: int = 500,
    client: VocabularyClient | None = None,
) -> AsrHotwordSyncSummary:
    """
    Synchronize accepted correction hotwords to DashScope.

    Args:
        settings: Runtime settings.
        target_model: DashScope ASR model.
        db_path: Optional lexicon database path.
        prefix: DashScope vocabulary prefix.
        force: Force remote update even when hash is unchanged.
        dry_run: Write local artifact without remote changes.
        output: Optional hotword artifact output path.
        limit: Maximum lexicon terms to export.
        client: Optional test double for DashScope VocabularyService.

    Returns:
        Synchronization summary.
    """
    database_path = db_path or default_lexicon_db_path()
    _validate_prefix(prefix)
    hotwords = list_asr_hotwords(db_path=database_path, limit=limit)
    artifact_path = _write_optional_artifact(output, hotwords)
    if not hotwords:
        return _empty_summary(
            database_path, settings, target_model, dry_run, artifact_path
        )
    table_hash = hotword_hash(hotwords)
    endpoint = settings.dashscope_base_url
    state = get_asr_vocabulary_state(
        target_model=target_model, endpoint=endpoint, db_path=database_path
    )
    if state and state.vocabulary_hash == table_hash and not force:
        return _unchanged_summary(database_path, state, dry_run, artifact_path)
    if dry_run:
        return _dry_run_summary(
            database_path,
            settings,
            target_model,
            table_hash,
            hotwords,
            state,
            artifact_path,
        )
    vocabulary_id = _sync_remote_vocabulary(
        settings, target_model, prefix, hotwords, state, client
    )
    new_state = AsrVocabularyState(
        target_model, endpoint, table_hash, vocabulary_id, len(hotwords)
    )
    save_asr_vocabulary_state(new_state, db_path=database_path)
    return _changed_summary(database_path, new_state, artifact_path)


def list_remote_asr_vocabularies(
    *,
    settings: Settings,
    prefix: str | None = DEFAULT_HOTWORD_PREFIX,
    page_index: int = 0,
    page_size: int = 10,
    client: VocabularyClient | None = None,
) -> list[dict]:
    """
    List remote DashScope hotword vocabularies.

    Args:
        settings: Runtime settings.
        prefix: Optional vocabulary prefix filter.
        page_index: Remote page index.
        page_size: Remote page size.
        client: Optional test double for DashScope VocabularyService.

    Returns:
        Remote vocabulary metadata rows.
    """
    _configure_dashscope(settings)
    service = client or VocabularyService()
    rows = service.list_vocabularies(
        prefix=prefix, page_index=page_index, page_size=page_size
    )
    if not isinstance(rows, list):
        raise RuntimeError("DashScope list_vocabularies did not return a list.")
    return [_plain_payload(row) for row in rows]


def query_remote_asr_vocabulary(
    *,
    settings: Settings,
    vocabulary_id: str,
    client: VocabularyClient | None = None,
) -> dict:
    """
    Query one remote DashScope hotword vocabulary.

    Args:
        settings: Runtime settings.
        vocabulary_id: Remote DashScope vocabulary id.
        client: Optional test double for DashScope VocabularyService.

    Returns:
        Remote vocabulary payload.
    """
    _configure_dashscope(settings)
    service = client or VocabularyService()
    payload = service.query_vocabulary(vocabulary_id)
    normalized = _plain_payload(payload)
    if not isinstance(normalized, dict):
        raise RuntimeError("DashScope query_vocabulary did not return an object.")
    return normalized


def delete_remote_asr_vocabulary(
    *,
    settings: Settings,
    vocabulary_id: str,
    client: VocabularyClient | None = None,
) -> None:
    """
    Delete one remote DashScope hotword vocabulary.

    Args:
        settings: Runtime settings.
        vocabulary_id: Remote DashScope vocabulary id.
        client: Optional test double for DashScope VocabularyService.

    Returns:
        None.
    """
    _configure_dashscope(settings)
    service = client or VocabularyService()
    service.delete_vocabulary(vocabulary_id)


def _auto_hotword_resolution(
    *,
    settings: Settings,
    target_model: str,
    db_path: Path | None,
) -> AsrHotwordResolution:
    """Resolve automatic ASR hotwords through config or sync."""
    if settings.dashscope_asr_vocabulary_id:
        return AsrHotwordResolution(settings.dashscope_asr_vocabulary_id, "config")
    try:
        summary = sync_asr_hotwords(
            settings=settings, target_model=target_model, db_path=db_path
        )
    except Exception as exc:
        return AsrHotwordResolution(None, "auto-error", error=str(exc))
    return AsrHotwordResolution(
        summary.vocabulary_id, "auto", summary.hotword_count, summary.vocabulary_hash
    )


def _sync_remote_vocabulary(
    settings: Settings,
    target_model: str,
    prefix: str,
    hotwords: list[AsrHotword],
    state: AsrVocabularyState | None,
    client: VocabularyClient | None,
) -> str:
    """Create or update the remote DashScope vocabulary."""
    _configure_dashscope(settings)
    service = client or VocabularyService()
    vocabulary = dashscope_vocabulary(hotwords)
    if state is not None:
        try:
            service.update_vocabulary(state.vocabulary_id, vocabulary)
            return state.vocabulary_id
        except Exception:
            pass
    return service.create_vocabulary(
        target_model=target_model, prefix=prefix, vocabulary=vocabulary
    )


def _cache_status(
    *,
    hotword_count: int,
    table_hash: str | None,
    state: AsrVocabularyState | None,
) -> str:
    """Return whether the cached vocabulary id matches local hotwords."""
    if state is None:
        return "empty" if hotword_count == 0 else "missing"
    if table_hash is not None and state.vocabulary_hash == table_hash:
        return "current"
    return "stale"


def _validate_prefix(prefix: str) -> None:
    """Validate DashScope hotword vocabulary prefix."""
    if not PREFIX_RE.fullmatch(prefix):
        raise ValueError(
            "DashScope hotword prefix must be 1-9 lowercase letters or digits."
        )


def _configure_dashscope(settings: Settings) -> None:
    """Configure DashScope SDK globals for vocabulary APIs."""
    dashscope.api_key = settings.dashscope_api_key
    if settings.dashscope_base_url:
        for attr in ("base_http_api_url", "base_url"):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, settings.dashscope_base_url)


def _plain_payload(value):
    """Convert DashScope SDK return values into JSON-friendly containers."""
    if isinstance(value, dict):
        return {str(key): _plain_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_payload(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain_payload(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _write_optional_artifact(
    output: Path | None, hotwords: list[AsrHotword]
) -> Path | None:
    """Write a local hotword artifact when requested."""
    if output is None:
        return None
    return write_hotword_artifact(output, hotwords)


def _empty_summary(
    db_path: Path,
    settings: Settings,
    target_model: str,
    dry_run: bool,
    artifact_path: Path | None,
) -> AsrHotwordSyncSummary:
    """Build a summary when no hotwords exist."""
    return AsrHotwordSyncSummary(
        db_path,
        target_model,
        settings.dashscope_base_url,
        None,
        0,
        None,
        False,
        dry_run,
        artifact_path,
    )


def _unchanged_summary(
    db_path: Path,
    state: AsrVocabularyState,
    dry_run: bool,
    artifact_path: Path | None,
) -> AsrHotwordSyncSummary:
    """Build a summary for an unchanged remote vocabulary."""
    return AsrHotwordSyncSummary(
        db_path,
        state.target_model,
        state.endpoint,
        state.vocabulary_id,
        state.hotword_count,
        state.vocabulary_hash,
        False,
        dry_run,
        artifact_path,
    )


def _dry_run_summary(
    db_path: Path,
    settings: Settings,
    target_model: str,
    table_hash: str,
    hotwords: list[AsrHotword],
    state: AsrVocabularyState | None,
    artifact_path: Path | None,
) -> AsrHotwordSyncSummary:
    """Build a dry-run summary without remote changes."""
    vocabulary_id = state.vocabulary_id if state else None
    return AsrHotwordSyncSummary(
        db_path,
        target_model,
        settings.dashscope_base_url,
        vocabulary_id,
        len(hotwords),
        table_hash,
        False,
        True,
        artifact_path,
    )


def _changed_summary(
    db_path: Path,
    state: AsrVocabularyState,
    artifact_path: Path | None,
) -> AsrHotwordSyncSummary:
    """Build a summary for a changed remote vocabulary."""
    return AsrHotwordSyncSummary(
        db_path,
        state.target_model,
        state.endpoint,
        state.vocabulary_id,
        state.hotword_count,
        state.vocabulary_hash,
        True,
        False,
        artifact_path,
    )
