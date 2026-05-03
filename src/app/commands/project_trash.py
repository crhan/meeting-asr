"""Project trash CLI commands."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table
import typer

from app.core.project_models import (
    ProjectPurgeSummary,
    ProjectRestoreSummary,
    ProjectTrashCleanupSummary,
    TrashedProjectListItem,
)
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json
from app.presentation.cli.output import cli_console
from app.presentation.cli.plain import echo_plain_table
from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper
from app.project_trash import (
    cleanup_project_trash,
    list_trashed_projects,
    purge_trashed_project,
    restore_trashed_project,
)

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command("list")
def list_command(
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    plain: bool = typer.Option(False, "--plain", help="Print stable tab-separated output."),
) -> None:
    """List projects currently stored in Meeting-ASR trash."""
    result = run_with_cli_errors(list_trashed_projects)
    if as_json:
        emit_json(_project_trash_payload(result.trash_dir, result.projects))
        return
    if plain:
        _echo_project_trash_list_plain(result.projects)
        return
    _echo_project_trash_list(result.trash_dir, result.projects)


@app.command("restore")
def restore_command(
    trash_ref: str = typer.Argument(..., metavar="TRASH"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
) -> None:
    """Restore a trashed project by trash path, id, directory, or title."""
    summary = run_with_cli_errors(
        lambda: restore_trashed_project(trash_ref, projects_dir=projects_dir, project_dir=project_dir)
    )
    _echo_project_restored(summary)


@app.command("purge")
def purge_command(
    trash_ref: str = typer.Argument(..., metavar="TRASH"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation."),
) -> None:
    """Permanently delete one project from Meeting-ASR trash."""
    if not yes and not typer.confirm(f"permanently delete trashed project {trash_ref}?"):
        typer.echo("Project purge cancelled.")
        return
    summary = run_with_cli_errors(lambda: purge_trashed_project(trash_ref))
    _echo_project_purged(summary)


@app.command("cleanup")
def cleanup_command(
    older_than_days: int = typer.Option(30, "--older-than-days", min=0),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation."),
) -> None:
    """Permanently delete trashed projects older than N days."""
    if not yes and not typer.confirm(f"permanently delete trashed projects older than {older_than_days} day(s)?"):
        typer.echo("Project trash cleanup cancelled.")
        return
    summary = run_with_cli_errors(lambda: cleanup_project_trash(older_than_days=older_than_days))
    _echo_project_trash_cleanup(summary)


def _echo_project_trash_list(trash_dir: Path, projects: list[TrashedProjectListItem]) -> None:
    """
    Print trashed project rows.

    Args:
        trash_dir: Project trash directory.
        projects: Trashed project rows.

    Returns:
        None.
    """
    typer.echo(f"Project trash: {trash_dir}")
    if not projects:
        typer.echo("No trashed projects found.")
        return
    typer.echo("Use Project ID or Trash Dir with restore/purge.")
    _project_table_console().print(_project_trash_table(projects))


def _echo_project_trash_list_plain(projects: list[TrashedProjectListItem]) -> None:
    """
    Print trashed project rows as stable tab-separated values.

    Args:
        projects: Trashed project rows.
    """
    rows = [
        (
            project.project_id,
            project.status,
            _project_list_timestamp(project.trashed_at),
            project.title,
            project.trash_dir.name,
        )
        for project in projects
    ]
    echo_plain_table(("project_id", "status", "trashed", "title", "trash_dir"), rows)


def _project_trash_table(projects: list[TrashedProjectListItem]) -> Table:
    """
    Build the project trash table.

    Args:
        projects: Trash rows to display.

    Returns:
        Rich table ready to print.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("Project ID", no_wrap=True, overflow="ellipsis", max_width=28, style="bold cyan")
    table.add_column("Status", no_wrap=True)
    table.add_column("Trashed", no_wrap=True)
    table.add_column("Title")
    table.add_column("Trash Dir", no_wrap=True, overflow="ellipsis", max_width=38)
    for project in projects:
        table.add_row(
            project.project_id,
            _project_status_text(project.status),
            _project_list_timestamp(project.trashed_at),
            project.title,
            project.trash_dir.name,
        )
    return table


def _project_trash_payload(trash_dir: Path, projects: list[TrashedProjectListItem]) -> dict[str, object]:
    """
    Build a machine-readable project trash payload.

    Args:
        trash_dir: Project trash directory.
        projects: Trashed project rows.

    Returns:
        JSON-ready trash list payload.
    """
    return {
        "trash_dir": trash_dir,
        "count": len(projects),
        "projects": [_trashed_project_payload(project) for project in projects],
    }


def _trashed_project_payload(project: TrashedProjectListItem) -> dict[str, object]:
    """
    Build one trashed project JSON row.

    Args:
        project: Trashed project row.

    Returns:
        JSON-ready project row.
    """
    return {
        "project_id": project.project_id,
        "title": project.title,
        "status": project.status,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "trashed_at": project.trashed_at,
        "trash_dir": project.trash_dir,
        "restore_project_dir": project.restore_project_dir,
        "trash_directory": project.trash_dir.name,
    }


def _echo_project_restored(summary: ProjectRestoreSummary) -> None:
    """
    Print project restore output.

    Args:
        summary: Restore summary.

    Returns:
        None.
    """
    project_ref = summary.manifest.project_id
    typer.echo("Project restored.")
    typer.echo(f"Project: {summary.project_dir}")
    typer.echo(f"Project ID: {summary.manifest.project_id}")
    typer.echo(f"Title: {summary.manifest.title}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  meeting-asr project review {shlex.quote(project_ref)}")
    typer.echo(f"  meeting-asr project transcript show {shlex.quote(project_ref)}")


def _echo_project_purged(summary: ProjectPurgeSummary) -> None:
    """
    Print project purge output.

    Args:
        summary: Purge summary.

    Returns:
        None.
    """
    typer.echo("Trashed project permanently deleted.")
    typer.echo(f"Trash: {summary.trash_dir}")
    typer.echo(f"Project ID: {summary.manifest.project_id}")
    typer.echo(f"Title: {summary.manifest.title}")


def _echo_project_trash_cleanup(summary: ProjectTrashCleanupSummary) -> None:
    """
    Print project trash cleanup output.

    Args:
        summary: Cleanup summary.

    Returns:
        None.
    """
    typer.echo("Project trash cleanup complete.")
    typer.echo(f"Trash: {summary.trash_dir}")
    typer.echo(f"Removed: {len(summary.removed)}")
    for item in summary.removed:
        typer.echo(f"  - {item.manifest.title} ({item.manifest.project_id})")

def _project_status_text(status: str) -> str:
    """Return a styled project status for table display."""
    styles = {
        "created": "yellow",
        "prepared": "yellow",
        "transcribed": "cyan",
        "named": "green",
        "corrected": "green",
        "voiceprinted": "green",
    }
    return f"[{styles.get(status, 'white')}]{status}[/]"


def _project_table_console() -> Console:
    """Build the stdout console used for project tables."""
    return cli_console(width=140)


def _project_list_timestamp(value: str) -> str:
    """Return a compact timestamp for project list rows."""
    return value[:19]
