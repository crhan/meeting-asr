"""Global path discovery command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer

from app.config import (
    get_cache_dir,
    get_config_path,
    get_data_dir,
    get_default_projects_dir,
)
from app.lexicon_store import default_lexicon_db_path
from app.presentation.cli.json_output import emit_json
from app.project_trash import get_project_trash_dir
from app.voiceprint_store import get_voiceprint_clip_dir, get_voiceprint_db_path


@dataclass(frozen=True, slots=True)
class PathRow:
    """One Meeting-ASR state path."""

    key: str
    path: Path
    description: str


def command(
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show Meeting-ASR config, data, cache, and store paths."""
    rows = _path_rows()
    if as_json:
        emit_json({"paths": [_row_payload(row) for row in rows]})
        return
    _echo_path_rows(rows)


def _path_rows() -> list[PathRow]:
    """Return the known global paths in display order."""
    return [
        PathRow("config", get_config_path(), "Global config file"),
        PathRow("data", get_data_dir(), "Global data directory"),
        PathRow("cache", get_cache_dir(), "Global cache directory"),
        PathRow("projects", get_default_projects_dir(), "Project store"),
        PathRow("project_trash", get_project_trash_dir(), "Deleted project trash"),
        PathRow(
            "voiceprint_db", get_voiceprint_db_path(), "Voiceprint SQLite database"
        ),
        PathRow("voiceprint_clips", get_voiceprint_clip_dir(), "Voiceprint WAV clips"),
        PathRow(
            "lexicon_db",
            default_lexicon_db_path(),
            "Correction lexicon SQLite database",
        ),
    ]


def _row_payload(row: PathRow) -> dict[str, str]:
    """Convert one path row to a JSON object."""
    return {
        "key": row.key,
        "path": str(row.path),
        "description": row.description,
    }


def _echo_path_rows(rows: list[PathRow]) -> None:
    """Print known state paths as copyable lines."""
    typer.echo("Meeting-ASR paths")
    for row in rows:
        typer.echo(f"{row.key}: {row.path}")
        typer.echo(f"  {row.description}")
