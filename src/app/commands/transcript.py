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

from app.core.project_refs import resolve_project_ref
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json
from app.presentation.cli.output import cli_console
from app.presentation.cli.plain import echo_plain_table
from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper
from app.project_manager import project_paths

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


class TranscriptKind(str, Enum):
    """Transcript artifact variants."""

    auto = "auto"
    plain = "plain"
    speakers = "speakers"
    named = "named"
    corrected = "corrected"
    named_corrected = "named-corrected"
    srt = "srt"
    srt_corrected = "srt-corrected"
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
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    plain: bool = typer.Option(False, "--plain", help="Print stable tab-separated output."),
) -> None:
    """List transcript artifacts for a project."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    paths = project_paths(resolved_project_dir)
    rows = _transcript_artifact_rows(paths.root)
    if as_json:
        emit_json(_transcript_artifacts_payload(paths.root, rows))
        return
    if plain:
        _echo_transcript_artifact_rows_plain(paths.root, rows)
        return
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
    if kind == TranscriptKind.corrected:
        return [paths.exports_dir / "transcript_named_corrected.txt", paths.exports_dir / "transcript_corrected.txt"]
    if kind == TranscriptKind.named_corrected:
        return [paths.exports_dir / "transcript_named_corrected.txt"]
    if kind == TranscriptKind.srt:
        return [paths.exports_dir / "subtitle_named.srt", paths.exports_dir / "subtitle.srt"]
    if kind == TranscriptKind.srt_corrected:
        return [paths.exports_dir / "subtitle_named_corrected.srt", paths.exports_dir / "subtitle_corrected.srt"]
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
        path = _first_existing_path(candidates)
        if _is_optional_corrected_kind(kind) and path is None:
            continue
        rows.append(
            TranscriptArtifactRow(
                kind=kind,
                path=path,
                candidates=candidates,
            )
        )
    return rows


def _is_optional_corrected_kind(kind: TranscriptKind) -> bool:
    """Return whether a transcript kind should only show after correction exists."""
    return kind in {TranscriptKind.corrected, TranscriptKind.named_corrected, TranscriptKind.srt_corrected}


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


def _echo_transcript_artifact_rows_plain(project_dir: Path, rows: list[TranscriptArtifactRow]) -> None:
    """
    Print transcript artifacts as stable tab-separated values.

    Args:
        project_dir: Project root.
        rows: Artifact rows to display.
    """
    plain_rows = [
        (row.kind.value, "available" if row.path else "missing", _plain_artifact_path(project_dir, row))
        for row in rows
    ]
    echo_plain_table(("kind", "status", "path"), plain_rows)


def _transcript_artifacts_payload(project_dir: Path, rows: list[TranscriptArtifactRow]) -> dict[str, object]:
    """
    Build a machine-readable transcript artifact payload.

    Args:
        project_dir: Project root.
        rows: Artifact rows to serialize.

    Returns:
        JSON-ready artifact payload.
    """
    available_count = sum(row.path is not None for row in rows)
    return {
        "project": project_dir,
        "count": len(rows),
        "available_count": available_count,
        "artifacts": [_transcript_artifact_payload(row) for row in rows],
    }


def _transcript_artifact_payload(row: TranscriptArtifactRow) -> dict[str, object]:
    """
    Build one transcript artifact JSON row.

    Args:
        row: Artifact row to serialize.

    Returns:
        JSON-ready artifact row.
    """
    return {
        "kind": row.kind.value,
        "available": row.path is not None,
        "path": row.path,
        "candidates": row.candidates,
    }


def _transcript_artifact_table(project_dir: Path, rows: list[TranscriptArtifactRow]) -> Table:
    """
    Build the transcript artifact table.

    Args:
        project_dir: Project root.
        rows: Artifact rows to display.

    Returns:
        Rich table ready to print.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("Kind", no_wrap=True, style="bold cyan")
    table.add_column("Status", no_wrap=True)
    table.add_column("Location")
    for row in rows:
        table.add_row(row.kind.value, _artifact_status(row), _artifact_location(project_dir, row))
    return table


def _artifact_status(row: TranscriptArtifactRow) -> str:
    """Return a short artifact availability label."""
    return "[green]available[/]" if row.path else "[red]missing[/]"


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


def _plain_artifact_path(project_dir: Path, row: TranscriptArtifactRow) -> str:
    """Return a plain path or expected candidate list for one artifact."""
    if row.path:
        return _relative_display_path(project_dir, row.path)
    return " or ".join(_relative_display_path(project_dir, path) for path in row.candidates)


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
    return cli_console(width=140)
