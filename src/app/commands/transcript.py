"""Transcript viewing commands."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import subprocess

import typer

from app.cli_errors import run_with_cli_errors
from app.project_manager import project_paths

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


class TranscriptKind(str, Enum):
    """Transcript artifact variants."""

    auto = "auto"
    plain = "plain"
    speakers = "speakers"
    named = "named"
    srt = "srt"
    raw = "raw"
    sentences = "sentences"


@app.command("list")
def list_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
) -> None:
    """List transcript artifacts for a project."""
    paths = project_paths(project_dir)
    typer.echo(f"Project: {paths.root}")
    for kind in TranscriptKind:
        if kind == TranscriptKind.auto:
            continue
        path = _resolve_transcript_path(paths.root, kind, required=False)
        status = str(path) if path else "-"
        typer.echo(f"{kind.value}: {status}")


@app.command("path")
def path_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Print one transcript artifact path."""
    path = run_with_cli_errors(lambda: _resolve_transcript_path(project_dir, kind, required=True))
    typer.echo(path)


@app.command("show")
def show_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Print one transcript artifact."""
    path = run_with_cli_errors(lambda: _resolve_transcript_path(project_dir, kind, required=True))
    typer.echo(path.read_text(encoding="utf-8"), nl=False)


@app.command("open")
def open_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Open one transcript artifact with the OS default application."""
    path = run_with_cli_errors(lambda: _resolve_transcript_path(project_dir, kind, required=True))
    run_with_cli_errors(lambda: subprocess.run(["open", str(path)], check=True))
    typer.echo(f"Opened: {path}")


def _resolve_transcript_path(project_dir: Path, kind: TranscriptKind, *, required: bool) -> Path | None:
    """
    Resolve a transcript artifact path.

    Args:
        project_dir: Project root.
        kind: Artifact kind.
        required: Raise when no matching file exists.

    Returns:
        Existing artifact path, or ``None`` when optional and missing.
    """
    paths = project_paths(project_dir)
    candidates = _transcript_candidates(paths.root, kind)
    for path in candidates:
        if path.exists():
            return path
    if not required:
        return None
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Transcript artifact does not exist. Expected one of: {expected}")


def _transcript_candidates(project_dir: Path, kind: TranscriptKind) -> list[Path]:
    """
    Return candidate paths for a transcript kind.

    Args:
        project_dir: Project root.
        kind: Artifact kind.

    Returns:
        Candidate paths in preference order.
    """
    paths = project_paths(project_dir)
    if kind == TranscriptKind.auto:
        return [
            paths.exports_dir / "transcript_named.txt",
            paths.exports_dir / "transcript_speakers.txt",
            paths.exports_dir / "transcript.txt",
        ]
    if kind == TranscriptKind.plain:
        return [paths.exports_dir / "transcript.txt"]
    if kind == TranscriptKind.speakers:
        return [paths.exports_dir / "transcript_speakers.txt"]
    if kind == TranscriptKind.named:
        return [paths.exports_dir / "transcript_named.txt"]
    if kind == TranscriptKind.srt:
        return [paths.exports_dir / "subtitle_named.srt", paths.exports_dir / "subtitle.srt"]
    if kind == TranscriptKind.raw:
        return [paths.asr_dir / "raw_result.json"]
    return [paths.asr_dir / "sentences.json"]
