"""Voiceprint registry commands."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Optional

import typer

from app.cli_errors import run_with_cli_errors
from app.cli_ui import run_with_progress
from app.completion_helpers import complete_voiceprint_model, complete_voiceprint_provider
from app.utils import format_ms_timestamp
from app.voiceprint_store import (
    delete_voiceprint_sample,
    delete_voiceprint_speaker,
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    list_voiceprint_speakers,
    VoiceprintSampleRow,
)
from app.voiceprint_embedding import embed_voiceprint_samples
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
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Capture this project's named speakers into the global voiceprint store."""
    summary = run_with_progress(
        lambda reporter: capture_voiceprints(
            project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            dry_run=dry_run,
            progress=reporter,
        ),
        description="Capturing voiceprints",
        enabled=progress,
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
        typer.echo(f"[{row.speaker_id}] {row.name}: {row.sample_count} sample(s)")


@app.command("embed")
def embed_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    provider: Optional[str] = typer.Option(None, "--provider", autocompletion=complete_voiceprint_provider),
    endpoint: Optional[str] = typer.Option(None, "--endpoint"),
    model: Optional[str] = typer.Option(None, "--model", autocompletion=complete_voiceprint_model),
    rebuild: bool = typer.Option(False, "--rebuild"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Generate embeddings for stored voiceprint samples."""
    summary = run_with_progress(
        lambda reporter: embed_voiceprint_samples(
            store_dir=store_dir,
            provider=provider,
            endpoint=endpoint,
            model=model,
            rebuild=rebuild,
            progress=reporter,
        ),
        description="Embedding voiceprints",
        enabled=progress,
    )
    typer.echo(f"Database: {summary.db_path}")
    typer.echo(f"Provider: {summary.provider}")
    typer.echo(f"Model: {summary.model}")
    typer.echo(f"Embedded: {summary.embedded_count}")
    typer.echo(f"Skipped: {summary.skipped_count}")


@app.command("show")
def show_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
) -> None:
    """Show voiceprint samples for one speaker name or id."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_samples(speaker, db_path))
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo(f"No voiceprint samples found for: {speaker}")
        raise typer.Exit(code=1)
    for index, row in enumerate(rows, start=1):
        start = format_ms_timestamp(row.source_begin_time_ms)
        end = format_ms_timestamp(row.source_end_time_ms)
        typer.echo(f"[{index}] {row.speaker_name} | {row.project_id} | speaker {row.project_speaker_id}")
        typer.echo(f"  speaker_id: {row.speaker_id}")
        typer.echo(f"  sample_id: {row.sample_id}")
        typer.echo(f"  clip: {row.clip_path}")
        typer.echo(f"  time: {start} - {end}")
        typer.echo(f"  sha256: {row.clip_sha256}")
        typer.echo(f"  text: {row.transcript_text}")


@app.command("play")
def play_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    sample: int = typer.Option(1, "--sample", "-s", min=1),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Play one numbered voiceprint sample."""
    db_path = get_voiceprint_db_path(store_dir)
    row = run_with_cli_errors(lambda: _select_sample(speaker, sample, db_path))
    command = _play_command(row.clip_path)
    if dry_run:
        typer.echo(" ".join(command))
        return
    run_with_cli_errors(lambda: subprocess.run(command, check=True))


@app.command("delete-sample")
def delete_sample_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    sample: int = typer.Option(..., "--sample", "-s", min=1),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    keep_clip: bool = typer.Option(False, "--keep-clip"),
) -> None:
    """Delete one numbered voiceprint sample and its WAV file."""
    db_path = get_voiceprint_db_path(store_dir)
    deleted = run_with_cli_errors(
        lambda: delete_voiceprint_sample(speaker, sample, db_path=db_path, delete_clip=not keep_clip)
    )
    _echo_deleted_sample(deleted.clip_path, deleted.clip_deleted, kept=keep_clip)


@app.command("delete-speaker")
def delete_speaker_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    keep_clips: bool = typer.Option(False, "--keep-clips"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete one speaker and all of their voiceprint samples."""
    db_path = get_voiceprint_db_path(store_dir)
    if not yes and not typer.confirm(f"Delete all voiceprint samples for {_speaker_label(speaker, db_path)}?"):
        raise typer.Exit(code=1)
    deleted = run_with_cli_errors(
        lambda: delete_voiceprint_speaker(speaker, db_path=db_path, delete_clips=not keep_clips)
    )
    typer.echo(f"Deleted speaker: {deleted[0].speaker_name} (id {deleted[0].speaker_id})")
    for item in deleted:
        _echo_deleted_sample(item.clip_path, item.clip_deleted, kept=keep_clips)


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


def _select_sample(speaker: str, sample: int, db_path: Path) -> VoiceprintSampleRow:
    """
    Select one sample for CLI playback.

    Args:
        speaker: Speaker name or speaker id.
        sample: One-based sample number.
        db_path: SQLite database path.

    Returns:
        Selected sample row.
    """
    rows = list_voiceprint_samples(speaker, db_path)
    if sample < 1 or sample > len(rows):
        raise IndexError(f"Sample {sample} is out of range for {speaker}. Available: {len(rows)}.")
    return rows[sample - 1]


def _speaker_label(speaker: str, db_path: Path) -> str:
    """
    Return a confirmation label for a speaker reference.

    Args:
        speaker: Speaker name or speaker id.
        db_path: SQLite database path.

    Returns:
        Human-readable confirmation label.
    """
    rows = run_with_cli_errors(lambda: list_voiceprint_samples(speaker, db_path))
    if not rows:
        return speaker
    return f"{rows[0].speaker_name} (id {rows[0].speaker_id})"


def _play_command(path: Path) -> list[str]:
    """
    Build a local playback command for one clip.

    Args:
        path: Clip path.

    Returns:
        Playback command.
    """
    if not path.exists():
        raise FileNotFoundError(f"Voiceprint clip does not exist: {path}")
    player = shutil.which("afplay")
    if player:
        return [player, str(path)]
    return [shutil.which("open") or "open", str(path)]


def _echo_deleted_sample(path: Path, clip_deleted: bool, *, kept: bool = False) -> None:
    """
    Print deletion result for one sample.

    Args:
        path: Clip path.
        clip_deleted: Whether the clip file was deleted.
        kept: Whether the user requested keeping the file.
    """
    status = "kept" if kept else ("deleted" if clip_deleted else "not found")
    typer.echo(f"Deleted sample: {path}")
    typer.echo(f"  clip file: {status}")
