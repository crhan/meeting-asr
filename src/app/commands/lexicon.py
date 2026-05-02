"""Lexicon and ASR hotword commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from app.asr_hotwords import DEFAULT_HOTWORD_PREFIX, sync_asr_hotwords
from app.config import load_settings
from app.correction_hotwords import write_hotword_artifact
from app.lexicon_store import default_lexicon_db_path, list_asr_hotwords
from app.presentation.cli.errors import run_with_cli_errors

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
hotwords_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(hotwords_app, name="hotwords", help="Export and sync ASR hotwords from accepted corrections.")


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
