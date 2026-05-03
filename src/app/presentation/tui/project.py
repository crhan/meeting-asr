"""Textual UI for selecting and entering meeting projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static

from app.core.project_models import ProjectListItem
from app.core.project_refs import list_projects
from app.core.project_workflow import load_project_workflow_summary, project_outputs_text

SHORTCUT_HELP = """\
[b]Project List Shortcuts[/b]

[b]Navigation[/b]
j/k or up/down       Select previous/next project
enter                Open selected project review TUI
?                    Show this help
q                    Quit

[b]Project Reference[/b]
You can also skip this list and run:
meeting-asr project review PROJECT_ID
meeting-asr project review PROJECT_PATH
meeting-asr project review PROJECT_TITLE
"""


@dataclass(frozen=True, slots=True)
class ProjectPickerSession:
    """Inputs needed by the project picker TUI."""

    projects_dir: Path
    projects: list[ProjectListItem]


class ProjectPickerHelpScreen(ModalScreen[None]):
    """Modal shortcut help for the project picker TUI."""

    CSS = """
    ProjectPickerHelpScreen {
        align: center middle;
    }
    #project-picker-help {
        width: 78;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("q", "close_help", "Close"),
        Binding("?", "close_help", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Build the help popup."""
        yield Static(SHORTCUT_HELP, id="project-picker-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class ProjectPickerApp(App[Path | None]):
    """Keyboard-first TUI for choosing a project."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #overview {
        border: round $accent;
        height: 5;
        padding: 0 1;
    }
    #projects {
        border: round $accent;
        height: 1fr;
        padding: 0 1;
    }
    #detail {
        border: round $accent;
        height: 9;
        padding: 0 1;
    }
    #status {
        height: 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "down", "Down"),
        Binding("k", "up", "Up"),
        Binding("down", "down", "Down", show=False),
        Binding("up", "up", "Up", show=False),
        Binding("enter", "open_project", "Open"),
        Binding("?", "show_shortcuts", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, session: ProjectPickerSession) -> None:
        """
        Create the project picker app.

        Args:
            session: Project list inputs.
        """
        super().__init__()
        self.session = session
        self.selected_project_index = 0

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        yield Static(id="projects")
        yield Static(id="detail")
        yield Static("Use ↑/↓ or j/k, Enter opens review, ? help, q quit", id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Render the initial project list."""
        self._refresh()

    def action_down(self) -> None:
        """Select the next project."""
        self._move_project(1)

    def action_up(self) -> None:
        """Select the previous project."""
        self._move_project(-1)

    def action_open_project(self) -> None:
        """Exit with the selected project path."""
        project = self._project()
        if project is None:
            self.query_one("#status", Static).update("No project selected.")
            return
        self.exit(project.project_dir)

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.push_screen(ProjectPickerHelpScreen())

    def action_quit(self) -> None:
        """Exit without selecting a project."""
        self.exit(None)

    def _move_project(self, delta: int) -> None:
        """Move the selected project index."""
        total = len(self.session.projects)
        if total == 0:
            return
        self.selected_project_index = (self.selected_project_index + delta) % total
        self._refresh()

    def _refresh(self) -> None:
        """Refresh all project picker panes."""
        self.query_one("#overview", Static).update(self._overview_pane())
        self.query_one("#projects", Static).update(self._project_list_pane())
        self.query_one("#detail", Static).update(self._detail_pane())

    def _overview_pane(self) -> str:
        """Render project list summary."""
        selected = self._project()
        selected_text = "-" if selected is None else f"{selected.project_id} | {selected.title}"
        return "\n".join(
            [
                f"[b]Projects[/b] {escape(str(self.session.projects_dir))}",
                f"[b]Count[/b]    {len(self.session.projects)}",
                f"[b]Selected[/b] {escape(selected_text)}",
            ]
        )

    def _project_list_pane(self) -> str:
        """Render selectable project rows."""
        lines = ["[b]Project List[/b]"]
        if not self.session.projects:
            lines.append("[yellow]No projects found.[/]")
            return "\n".join(lines)
        for index, project in enumerate(self.session.projects):
            marker = ">" if index == self.selected_project_index else " "
            workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
            row = (
                f"{marker} {project.project_id} | {project.updated_at[:19]} | "
                f"{workflow.state} | {project.title}"
            )
            lines.append(f"[reverse]{escape(row)}[/]" if marker == ">" else escape(row))
        return "\n".join(lines)

    def _detail_pane(self) -> str:
        """Render detail for the selected project."""
        project = self._project()
        if project is None:
            return "[b]Detail[/b]\nNo project selected."
        workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
        return "\n".join(
            [
                "[b]Detail[/b]",
                f"Project ID: {escape(project.project_id)}",
                f"Title: {escape(project.title)}",
                f"State: {escape(workflow.state)}",
                f"Next: {escape(workflow.next_command_short)}",
                f"Artifacts: {escape(project_outputs_text(workflow.outputs))}",
                f"Path: {escape(str(project.project_dir))}",
                f"Open: meeting-asr project review {escape(project.project_id)}",
            ]
        )

    def _project(self) -> ProjectListItem | None:
        """Return the selected project, if any."""
        if not self.session.projects:
            return None
        return self.session.projects[self.selected_project_index]


def load_project_picker_session(projects_dir: Path | None = None) -> ProjectPickerSession:
    """
    Load projects for the picker TUI.

    Args:
        projects_dir: Optional projects parent directory.

    Returns:
        Project picker session.
    """
    result = list_projects(projects_dir)
    return ProjectPickerSession(projects_dir=result.projects_dir, projects=result.projects)


def run_project_picker_tui(session: ProjectPickerSession) -> Path | None:
    """
    Run the project picker TUI.

    Args:
        session: Project picker inputs.

    Returns:
        Selected project path, or ``None`` when cancelled.
    """
    return ProjectPickerApp(session).run()


def render_project_picker_summary(session: ProjectPickerSession) -> str:
    """
    Render a non-interactive project picker summary.

    Args:
        session: Project picker inputs.

    Returns:
        Plain terminal text.
    """
    lines = [f"Projects: {session.projects_dir}", f"Count: {len(session.projects)}"]
    if not session.projects:
        lines.append("No projects found.")
        return "\n".join(lines)
    for project in session.projects:
        workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
        lines.append(
            f"{project.project_id} | {workflow.state} | {workflow.next_command_short} | "
            f"{project_outputs_text(workflow.outputs)} | {project.title} | {project.project_dir}"
        )
    return "\n".join(lines)
