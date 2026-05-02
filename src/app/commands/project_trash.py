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
    ProjectManifest,
    ProjectPurgeSummary,
    ProjectRestoreSummary,
    ProjectTrashCleanupSummary,
    TrashedProjectListItem,
)
from app.core.project_refs import list_projects
from app.presentation.cli.errors import run_with_cli_errors
from app.project_trash import (
    cleanup_project_trash,
    list_trashed_projects,
    purge_trashed_project,
    restore_trashed_project,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("list")
def list_command() -> None:
    """List projects currently stored in Meeting-ASR trash."""
    result = run_with_cli_errors(list_trashed_projects)
    _echo_project_trash_list(result.trash_dir, result.projects)


@app.command("restore")
def restore_command(
    trash_ref: str = typer.Argument(..., metavar="TRASH"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
) -> None:
    """Restore a trashed project by trash number, path, id, or title."""
    summary = run_with_cli_errors(
        lambda: restore_trashed_project(trash_ref, projects_dir=projects_dir, project_dir=project_dir)
    )
    _echo_project_restored(summary, projects_dir)


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
    typer.echo("Use No. with restore/purge, e.g. `meeting-asr project trash restore 1`.")
    _project_table_console().print(_project_trash_table(projects))


def _project_trash_table(projects: list[TrashedProjectListItem]) -> Table:
    """
    Build the project trash table.

    Args:
        projects: Numbered trash rows to display.

    Returns:
        Rich table ready to print.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("No.", justify="right", no_wrap=True, style="bold cyan")
    table.add_column("Trashed", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Title")
    table.add_column("Project ID", no_wrap=True)
    table.add_column("Restore To", no_wrap=True)
    table.add_column("Trash Dir", no_wrap=True)
    for project in projects:
        table.add_row(
            str(project.number),
            _project_list_timestamp(project.trashed_at),
            _project_status_text(project.status),
            project.title,
            project.project_id,
            project.restore_project_dir.name,
            project.trash_dir.name,
        )
    return table


def _echo_project_restored(summary: ProjectRestoreSummary, projects_dir: Path | None) -> None:
    """
    Print project restore output.

    Args:
        summary: Restore summary.
        projects_dir: Optional projects parent used by the command.

    Returns:
        None.
    """
    project_ref = _project_cli_ref(summary.project_dir, summary.manifest, projects_dir)
    typer.echo("Project restored.")
    typer.echo(f"Project: {summary.project_dir}")
    if project_ref != summary.manifest.project_id:
        typer.echo(f"Project No.: {project_ref}")
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


def _project_cli_ref(project_dir: Path, manifest: ProjectManifest, projects_dir: Path | None) -> str:
    """
    Return the shortest safe project reference for follow-up commands.

    Args:
        project_dir: Resolved project root.
        manifest: Project manifest.
        projects_dir: Optional projects parent used by the current command.

    Returns:
        Project list number when available, otherwise the stable project id.
    """
    resolved = project_dir.expanduser().resolve()
    for project in list_projects(projects_dir).projects:
        if project.project_dir == resolved:
            return str(project.number)
    return manifest.project_id


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
    return Console(highlight=False, color_system="auto", width=140)


def _project_list_timestamp(value: str) -> str:
    """Return a compact timestamp for project list rows."""
    return value[:19]
