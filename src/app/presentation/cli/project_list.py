"""Project list rendering for human and plain CLI output."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
import typer

from app.core.project_models import ProjectListItem
from app.core.project_workflow import ProjectWorkflowSummary, load_project_workflow_summary
from app.presentation.cli.output import cli_console
from app.presentation.cli.plain import echo_plain_table


def render_project_list(projects_dir: Path, projects: list[ProjectListItem], *, plain: bool = False) -> None:
    """
    Render project list rows.

    Args:
        projects_dir: Resolved projects parent directory.
        projects: Project rows to render.
        plain: Whether to print stable tab-separated output.

    Returns:
        None.
    """
    if plain:
        _echo_project_list_plain(projects)
        return
    _echo_project_list(projects_dir, projects)


def _echo_project_list(projects_dir: Path, projects: list[ProjectListItem]) -> None:
    """Print project list rows for humans."""
    typer.echo(f"Projects: {projects_dir}")
    if not projects:
        typer.echo("No projects found.")
        return
    typer.echo("Use Project ID or Directory with PROJECT commands.")
    typer.echo("Start with: meeting-asr project show PROJECT_ID")
    _project_table_console().print(_project_list_table(projects))


def _echo_project_list_plain(projects: list[ProjectListItem]) -> None:
    """Print project rows as stable tab-separated values."""
    rows = []
    for project in projects:
        workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
        rows.append((project.project_id, workflow.state, _project_list_timestamp(project.updated_at), project.title))
    echo_plain_table(("project_id", "state", "updated", "title"), rows)


def _project_list_table(projects: list[ProjectListItem]) -> Table:
    """Build the project list table."""
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("Project ID", no_wrap=True, style="bold cyan")
    table.add_column("State", no_wrap=True)
    table.add_column("Updated", no_wrap=True)
    table.add_column("Title")
    for project in projects:
        workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
        table.add_row(
            project.project_id,
            _project_workflow_state_text(workflow),
            _project_list_timestamp(project.updated_at),
            project.title,
        )
    return table


def _project_workflow_state_text(workflow: ProjectWorkflowSummary) -> str:
    """Return a styled workflow state for table display."""
    styles = {
        "created": "yellow",
        "prepared": "yellow",
        "transcribed": "cyan",
        "completed": "green",
        "corrected": "green",
        "broken": "red",
    }
    return f"[{styles.get(workflow.state_key, 'white')}]{workflow.state}[/]"


def _project_table_console() -> Console:
    """Build the stdout console used for project tables."""
    return cli_console(width=140)


def _project_list_timestamp(value: str) -> str:
    """Return a compact timestamp for project list rows."""
    return value[:16].replace("T", " ")
