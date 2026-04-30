"""Voiceprint registry commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from app.cli_errors import run_with_cli_errors
from app.utils import format_ms_timestamp
from app.voiceprint_store import (
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    list_voiceprint_speakers,
)
from app.voiceprints import VoiceprintCaptureSummary, capture_voiceprints

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("capture")
def capture_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    sample_count: int = typer.Option(3, "--sample-count", min=1, max=20),
    max_seconds: float = typer.Option(12.0, "--max-seconds", min=0.1),
    padding_seconds: float = typer.Option(0.5, "--padding-seconds", min=0.0),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Capture this project's named speakers into the global voiceprint store."""
    summary = run_with_cli_errors(
        lambda: capture_voiceprints(
            project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            dry_run=dry_run,
        )
    )
    _echo_capture_summary(summary)


@app.command("list")
def list_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
) -> None:
    """List speakers recorded in the global voiceprint registry."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_speakers(db_path))
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo("No voiceprints recorded.")
        return
    for row in rows:
        typer.echo(f"{row.name}: {row.sample_count} sample(s)")


@app.command("show")
def show_command(
    name: str = typer.Argument(..., metavar="NAME"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
) -> None:
    """Show voiceprint samples for one speaker name."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_samples(name, db_path))
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo(f"No voiceprint samples found for: {name}")
        raise typer.Exit(code=1)
    for row in rows:
        start = format_ms_timestamp(row.source_begin_time_ms)
        end = format_ms_timestamp(row.source_end_time_ms)
        typer.echo(f"{row.speaker_name} | {row.project_id} | speaker {row.project_speaker_id}")
        typer.echo(f"  clip: {row.clip_path}")
        typer.echo(f"  time: {start} - {end}")
        typer.echo(f"  sha256: {row.clip_sha256}")
        typer.echo(f"  text: {row.transcript_text}")


@app.command("path")
def path_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
) -> None:
    """Print the global voiceprint store paths."""
    db_path = get_voiceprint_db_path(store_dir)
    typer.echo(f"Store: {db_path.parent}")
    typer.echo(f"Database: {db_path}")
    typer.echo(f"Clips: {get_voiceprint_clip_dir(store_dir)}")


def _echo_capture_summary(summary: VoiceprintCaptureSummary) -> None:
    """
    Print capture results.

    Args:
        summary: Capture summary.
    """
    status = "Planned" if summary.dry_run else "Captured"
    typer.echo(f"{status} voiceprint samples: {summary.sample_count}")
    typer.echo(f"Store: {summary.store_dir}")
    typer.echo(f"Database: {summary.db_path}")
    typer.echo(f"Clips: {summary.clip_dir}")
    for speaker in summary.speakers:
        typer.echo(f"{speaker.name} (speaker {speaker.speaker_id}): {len(speaker.clips)} sample(s)")
        for clip in speaker.clips:
            typer.echo(f"  - {clip.path}")
