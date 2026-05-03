"""Lexicon and ASR hotword commands."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from app.asr_hotwords import (
    DEFAULT_HOTWORD_PREFIX,
    AsrHotwordStatus,
    delete_remote_asr_vocabulary,
    get_asr_hotword_status,
    list_remote_asr_vocabularies,
    query_remote_asr_vocabulary,
    sync_asr_hotwords,
)
from app.config import load_settings
from app.correction_hotwords import write_hotword_artifact
from app.lexicon_store import (
    AsrVocabularyState,
    default_lexicon_db_path,
    delete_asr_vocabulary_state,
    list_asr_hotwords,
)
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
hotwords_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(hotwords_app, name="hotwords", help="Export and sync ASR hotwords from accepted corrections.")


@hotwords_app.command("list")
def list_command(
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    limit: int = typer.Option(500, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """List local ASR hotwords derived from accepted corrections."""
    db_path = lexicon_db or default_lexicon_db_path()
    hotwords = run_with_cli_errors(lambda: list_asr_hotwords(db_path=db_path, limit=limit))
    if as_json:
        emit_json({"lexicon_db": db_path, "count": len(hotwords), "hotwords": [asdict(item) for item in hotwords]})
        return
    typer.echo(f"Lexicon DB: {db_path}")
    typer.echo(f"Hotwords: {len(hotwords)}")
    if not hotwords:
        typer.echo("No ASR hotwords.")
        return
    for index, hotword in enumerate(hotwords, start=1):
        typer.echo(f"{index}. {hotword.text} weight={hotword.weight} category={hotword.category}")


@hotwords_app.command("export")
def export_command(
    output: Optional[Path] = typer.Option(None, "--output", file_okay=True, dir_okay=False),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    limit: int = typer.Option(500, "--limit", min=1),
) -> None:
    """Export accepted correction knowledge as a DashScope ASR hotword table."""
    db_path = lexicon_db or default_lexicon_db_path()
    hotwords = run_with_cli_errors(lambda: list_asr_hotwords(db_path=db_path, limit=limit))
    output_path = output or db_path.parent / "asr_hotwords.json"
    written = run_with_cli_errors(lambda: write_hotword_artifact(output_path, hotwords))
    typer.echo("ASR hotwords exported.")
    typer.echo(f"Hotwords: {len(hotwords)}")
    typer.echo(f"Output: {written}")


@hotwords_app.command("status")
def status_command(
    target_model: str = typer.Option("fun-asr", "--target-model", help="DashScope ASR target model."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    limit: int = typer.Option(500, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show local hotword hash and cached DashScope vocabulary id."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=False))
    status = run_with_cli_errors(
        lambda: get_asr_hotword_status(settings=settings, target_model=target_model, db_path=lexicon_db, limit=limit)
    )
    if as_json:
        emit_json(_status_payload(status, configured_vocabulary_id=settings.dashscope_asr_vocabulary_id))
        return
    _echo_status(status, configured_vocabulary_id=settings.dashscope_asr_vocabulary_id)


@hotwords_app.command("sync")
def sync_command(
    target_model: str = typer.Option("fun-asr", "--target-model", help="DashScope ASR target model."),
    output: Optional[Path] = typer.Option(None, "--output", file_okay=True, dir_okay=False),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    prefix: str = typer.Option(DEFAULT_HOTWORD_PREFIX, "--prefix", help="DashScope vocabulary prefix."),
    force: bool = typer.Option(False, "--force", help="Force remote vocabulary update."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only render local hotword table."),
    limit: int = typer.Option(500, "--limit", min=1),
) -> None:
    """Sync accepted correction hotwords to DashScope and cache the vocabulary id."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=not dry_run))
    summary = run_with_cli_errors(
        lambda: sync_asr_hotwords(
            settings=settings,
            target_model=target_model,
            db_path=lexicon_db,
            prefix=prefix,
            force=force,
            dry_run=dry_run,
            output=output,
            limit=limit,
        )
    )
    _echo_sync_summary(summary)


@hotwords_app.command("clear-cache")
def clear_cache_command(
    target_model: str = typer.Option("fun-asr", "--target-model", help="DashScope ASR target model."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="DashScope base endpoint. Defaults to config."),
    vocabulary_id: Optional[str] = typer.Option(None, "--vocabulary-id", help="Only clear this cached vocabulary id."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
) -> None:
    """Clear one cached DashScope vocabulary id."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=False))
    resolved_endpoint = endpoint or settings.dashscope_base_url
    state = run_with_cli_errors(
        lambda: delete_asr_vocabulary_state(
            target_model=target_model,
            endpoint=resolved_endpoint,
            vocabulary_id=vocabulary_id,
            db_path=lexicon_db,
        )
    )
    if state is None:
        typer.echo("No matching ASR hotword cache entry.")
        return
    typer.echo("ASR hotword cache cleared.")
    typer.echo(f"Target model: {state.target_model}")
    typer.echo(f"Endpoint: {state.endpoint}")
    typer.echo(f"Vocabulary ID: {state.vocabulary_id}")


@hotwords_app.command("remote-list")
def remote_list_command(
    prefix: Optional[str] = typer.Option(DEFAULT_HOTWORD_PREFIX, "--prefix", help="DashScope vocabulary prefix filter."),
    page_index: int = typer.Option(0, "--page-index", min=0),
    page_size: int = typer.Option(10, "--page-size", min=1, max=100),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """List remote DashScope hotword vocabularies."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=True))
    rows = run_with_cli_errors(
        lambda: list_remote_asr_vocabularies(
            settings=settings,
            prefix=_empty_to_none(prefix),
            page_index=page_index,
            page_size=page_size,
        )
    )
    if as_json:
        emit_json({"prefix": prefix, "page_index": page_index, "page_size": page_size, "vocabularies": rows})
        return
    typer.echo(f"Remote ASR vocabularies: {len(rows)}")
    if not rows:
        typer.echo("No remote ASR vocabularies.")
        return
    for row in rows:
        typer.echo(_remote_row_line(row))


@hotwords_app.command("remote-show")
def remote_show_command(
    vocabulary_id: str = typer.Argument(..., help="DashScope vocabulary id."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show one remote DashScope hotword vocabulary."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=True))
    payload = run_with_cli_errors(lambda: query_remote_asr_vocabulary(settings=settings, vocabulary_id=vocabulary_id))
    if as_json:
        emit_json({"vocabulary_id": vocabulary_id, "remote": payload})
        return
    _echo_remote_vocabulary(vocabulary_id, payload)


@hotwords_app.command("remote-delete")
def remote_delete_command(
    vocabulary_id: str = typer.Argument(..., help="DashScope vocabulary id."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm remote deletion."),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Also clear the matching local cache entry."),
    target_model: str = typer.Option("fun-asr", "--target-model", help="DashScope ASR target model for cache clearing."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="DashScope base endpoint. Defaults to config."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
) -> None:
    """Delete one remote DashScope hotword vocabulary."""
    if not yes:
        run_with_cli_errors(lambda: _require_delete_confirmation(vocabulary_id))
    settings = run_with_cli_errors(lambda: load_settings(require_oss=False, require_dashscope=True))
    run_with_cli_errors(lambda: delete_remote_asr_vocabulary(settings=settings, vocabulary_id=vocabulary_id))
    typer.echo(f"Deleted remote ASR vocabulary: {vocabulary_id}")
    if clear_cache:
        _clear_deleted_remote_cache(settings, target_model, endpoint, vocabulary_id, lexicon_db)


def _echo_sync_summary(summary) -> None:
    """Print a hotword synchronization summary."""
    status = "dry run" if summary.dry_run else ("updated" if summary.changed else "unchanged")
    typer.echo(f"ASR hotword sync {status}.")
    typer.echo(f"Lexicon DB: {summary.db_path}")
    typer.echo(f"Target model: {summary.target_model}")
    typer.echo(f"Hotwords: {summary.hotword_count}")
    typer.echo(f"Vocabulary ID: {summary.vocabulary_id or '<none>'}")
    typer.echo(f"Hash: {summary.vocabulary_hash or '<none>'}")
    if summary.artifact_path:
        typer.echo(f"Artifact: {summary.artifact_path}")


def _echo_status(status: AsrHotwordStatus, *, configured_vocabulary_id: str | None) -> None:
    """Print local hotword and cache status."""
    typer.echo("ASR hotword status.")
    typer.echo(f"Lexicon DB: {status.db_path}")
    typer.echo(f"Target model: {status.target_model}")
    typer.echo(f"Endpoint: {status.endpoint}")
    typer.echo(f"Hotwords: {status.hotword_count}")
    typer.echo(f"Hash: {status.vocabulary_hash or '<none>'}")
    typer.echo(f"Cache: {status.cache_status}")
    typer.echo(f"Configured vocabulary ID: {configured_vocabulary_id or '<none>'}")
    if status.cached_state is None:
        typer.echo("Cached vocabulary ID: <none>")
        return
    typer.echo(f"Cached vocabulary ID: {status.cached_state.vocabulary_id}")
    typer.echo(f"Cached hotwords: {status.cached_state.hotword_count}")
    typer.echo(f"Cached hash: {status.cached_state.vocabulary_hash}")
    typer.echo(f"Cached updated: {status.cached_state.updated_at or '<unknown>'}")


def _status_payload(status: AsrHotwordStatus, *, configured_vocabulary_id: str | None) -> dict:
    """Build machine-readable hotword status output."""
    return {
        "lexicon_db": status.db_path,
        "target_model": status.target_model,
        "endpoint": status.endpoint,
        "hotword_count": status.hotword_count,
        "vocabulary_hash": status.vocabulary_hash,
        "cache_status": status.cache_status,
        "configured_vocabulary_id": configured_vocabulary_id,
        "cached_state": _state_payload(status.cached_state),
    }


def _state_payload(state: AsrVocabularyState | None) -> dict | None:
    """Build JSON output for a cached vocabulary state."""
    if state is None:
        return None
    return {
        "target_model": state.target_model,
        "endpoint": state.endpoint,
        "vocabulary_hash": state.vocabulary_hash,
        "vocabulary_id": state.vocabulary_id,
        "hotword_count": state.hotword_count,
        "updated_at": state.updated_at,
    }


def _remote_row_line(row: dict) -> str:
    """Render one remote vocabulary list row."""
    vocabulary_id = row.get("vocabulary_id") or "<unknown>"
    status = row.get("status") or "<unknown>"
    modified = row.get("gmt_modified") or row.get("updated_at") or "<unknown>"
    return f"{vocabulary_id} status={status} modified={modified}"


def _echo_remote_vocabulary(vocabulary_id: str, payload: dict) -> None:
    """Print one remote vocabulary payload."""
    typer.echo(f"Vocabulary ID: {vocabulary_id}")
    typer.echo(f"Status: {payload.get('status') or '<unknown>'}")
    typer.echo(f"Target model: {payload.get('target_model') or '<unknown>'}")
    vocabulary = payload.get("vocabulary")
    if not isinstance(vocabulary, list):
        typer.echo("Hotwords: <unknown>")
        return
    typer.echo(f"Hotwords: {len(vocabulary)}")
    for index, row in enumerate(vocabulary, start=1):
        if isinstance(row, dict):
            text = row.get("text") or "<unknown>"
            weight = row.get("weight") or "<unknown>"
            lang = row.get("lang")
            suffix = f" lang={lang}" if lang else ""
            typer.echo(f"{index}. {text} weight={weight}{suffix}")


def _require_delete_confirmation(vocabulary_id: str) -> None:
    """Require explicit confirmation for remote deletion."""
    raise ValueError(f"Refusing to delete remote ASR vocabulary {vocabulary_id}. Pass --yes to confirm.")


def _clear_deleted_remote_cache(
    settings,
    target_model: str,
    endpoint: str | None,
    vocabulary_id: str,
    lexicon_db: Path | None,
) -> None:
    """Clear the matching local cache entry after deleting a remote vocabulary."""
    resolved_endpoint = endpoint or settings.dashscope_base_url
    state = run_with_cli_errors(
        lambda: delete_asr_vocabulary_state(
            target_model=target_model,
            endpoint=resolved_endpoint,
            vocabulary_id=vocabulary_id,
            db_path=lexicon_db,
        )
    )
    if state is None:
        typer.echo("No matching ASR hotword cache entry.")
        return
    typer.echo("ASR hotword cache cleared.")


def _empty_to_none(value: str | None) -> str | None:
    """Normalize empty CLI strings to None."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
