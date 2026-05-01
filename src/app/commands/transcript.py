"""Transcript viewing commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table
import typer

from app.cli_errors import run_with_cli_errors
from app.project_manager import project_paths, resolve_project_ref

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


@dataclass(frozen=True, slots=True)
class TranscriptArtifactRow:
    """One transcript artifact row for list output."""

    kind: TranscriptKind
    path: Path | None
    candidates: list[Path]


@app.command("list")
def list_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
) -> None:
    """List transcript artifacts for a project."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    paths = project_paths(resolved_project_dir)
    rows = _transcript_artifact_rows(paths.root)
    _echo_transcript_artifact_rows(paths.root, rows)


@app.command("path")
def path_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Print one transcript artifact path."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    path = run_with_cli_errors(lambda: _resolve_transcript_path(resolved_project_dir, kind, required=True))
    typer.echo(path)


@app.command("show")
def show_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Print one transcript artifact."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    path = run_with_cli_errors(lambda: _resolve_transcript_path(resolved_project_dir, kind, required=True))
    typer.echo(path.read_text(encoding="utf-8"), nl=False)


@app.command("open")
def open_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    kind: TranscriptKind = typer.Option(TranscriptKind.auto, "--kind", "-k"),
) -> None:
    """Open one transcript artifact with the OS default application."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    path = run_with_cli_errors(lambda: _resolve_transcript_path(resolved_project_dir, kind, required=True))
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
    if path := _first_existing_path(candidates):
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


def _transcript_artifact_rows(project_dir: Path) -> list[TranscriptArtifactRow]:
    """
    Build transcript artifact rows in display order.

    Args:
        project_dir: Project root.

    Returns:
        Artifact rows with resolved availability.
    """
    rows = []
    for kind in TranscriptKind:
        if kind == TranscriptKind.auto:
            continue
        candidates = _transcript_candidates(project_dir, kind)
        rows.append(
            TranscriptArtifactRow(
                kind=kind,
                path=_first_existing_path(candidates),
                candidates=candidates,
            )
        )
    return rows


def _first_existing_path(candidates: list[Path]) -> Path | None:
    """
    Return the first existing candidate path.

    Args:
        candidates: Candidate paths in preference order.

    Returns:
        First existing path, or ``None`` when all candidates are absent.
    """
    for path in candidates:
        if path.exists():
            return path
    return None


def _echo_transcript_artifact_rows(project_dir: Path, rows: list[TranscriptArtifactRow]) -> None:
    """
    Print transcript artifacts as a compact summary table.

    Args:
        project_dir: Project root.
        rows: Artifact rows to display.
    """
    available_count = sum(row.path is not None for row in rows)
    typer.echo(f"Project: {project_dir}")
    typer.echo(f"Artifacts: {available_count}/{len(rows)} available")
    _transcript_table_console().print(_transcript_artifact_table(project_dir, rows))


def _transcript_artifact_table(project_dir: Path, rows: list[TranscriptArtifactRow]) -> Table:
    """
    Build the transcript artifact table.

    Args:
        project_dir: Project root.
        rows: Artifact rows to display.

    Returns:
        Rich table ready to print.
    """
    table = Table(box=box.ASCII, show_edge=False, pad_edge=False)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Location")
    for row in rows:
        table.add_row(row.kind.value, _artifact_status(row), _artifact_location(project_dir, row))
    return table


def _artifact_status(row: TranscriptArtifactRow) -> str:
    """Return a short artifact availability label."""
    return "available" if row.path else "missing"


def _artifact_location(project_dir: Path, row: TranscriptArtifactRow) -> str:
    """
    Return the display location for one artifact row.

    Args:
        project_dir: Project root.
        row: Artifact row.

    Returns:
        Existing path or expected candidates, relative to the project where possible.
    """
    if row.path:
        return _relative_display_path(project_dir, row.path)
    expected = " or ".join(_relative_display_path(project_dir, path) for path in row.candidates)
    return f"expected: {expected}"


def _relative_display_path(project_dir: Path, path: Path) -> str:
    """
    Format a path relative to the project root when possible.

    Args:
        project_dir: Project root.
        path: Path to display.

    Returns:
        Project-relative path, falling back to the absolute path.
    """
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def _transcript_table_console() -> Console:
    """
    Build the stdout console used for transcript tables.

    Returns:
        Rich console instance.
    """
    return Console(highlight=False, color_system=None, width=120)
