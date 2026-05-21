"""Textual UI for selecting and entering meeting projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Header, Static

from app.core.project_models import ProjectListItem
from app.core.project_refs import list_projects
from app.core.project_workflow import (
    load_project_workflow_summary,
    project_outputs_text,
)
from app.presentation.tui.i18n import tr
from app.presentation.time_format import format_local_minute


def shortcut_help() -> str:
    """Return localized project-picker shortcut help."""
    return tr(
        """\
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
""",
        """\
[b]项目列表快捷键[/b]

[b]导航[/b]
j/k 或 ↑/↓           选择上一个/下一个项目
enter                打开选中项目的 Review TUI
?                    显示帮助
q                    退出

[b]项目引用[/b]
也可以跳过列表，直接运行：
meeting-asr project review PROJECT_ID
meeting-asr project review PROJECT_PATH
meeting-asr project review PROJECT_TITLE
""",
    )


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
        yield Static(shortcut_help(), id="project-picker-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class _ProjectPickerBase:
    """Shared controller for standalone and embedded project pickers."""

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
        Binding("escape", "quit", "Back", show=False),
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
        yield Static(
            tr(
                "Use ↑/↓ or j/k, Enter opens review, ? help, q quit",
                "使用 ↑/↓ 或 j/k 选择，Enter 打开 review，? 帮助，q 退出",
            ),
            id="status",
        )

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
            self.query_one("#status", Static).update(
                tr("No project selected.", "未选择项目。")
            )
            return
        self._finish(project.project_dir)

    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcut help."""
        self.app.push_screen(ProjectPickerHelpScreen())

    def action_quit(self) -> None:
        """Exit without selecting a project."""
        self._finish(None)

    def _finish(self, project_dir: Path | None) -> None:
        """Return the selected project to the active Textual host."""
        raise NotImplementedError

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
        selected_text = (
            "-" if selected is None else f"{selected.project_id} | {selected.title}"
        )
        return "\n".join(
            [
                f"{tr('[b]Projects[/b]', '[b]项目[/b]')} {escape(str(self.session.projects_dir))}",
                f"{tr('[b]Count[/b]', '[b]数量[/b]')}    {len(self.session.projects)}",
                f"{tr('[b]Selected[/b]', '[b]当前[/b]')} {escape(selected_text)}",
            ]
        )

    def _project_list_pane(self) -> str:
        """Render selectable project rows."""
        lines = [tr("[b]Project List[/b]", "[b]项目列表[/b]")]
        if not self.session.projects:
            lines.append(
                tr("[yellow]No projects found.[/]", "[yellow]没有找到项目。[/]")
            )
            return "\n".join(lines)
        for index, project in enumerate(self.session.projects):
            marker = ">" if index == self.selected_project_index else " "
            workflow = load_project_workflow_summary(
                project.project_dir, project_ref=project.project_id
            )
            row = (
                f"{marker} {project.project_id} | {format_local_minute(project.updated_at)} | "
                f"{workflow.state} | {project.title}"
            )
            lines.append(f"[reverse]{escape(row)}[/]" if marker == ">" else escape(row))
        return "\n".join(lines)

    def _detail_pane(self) -> str:
        """Render detail for the selected project."""
        project = self._project()
        if project is None:
            return tr(
                "[b]Detail[/b]\nNo project selected.", "[b]详情[/b]\n未选择项目。"
            )
        workflow = load_project_workflow_summary(
            project.project_dir, project_ref=project.project_id
        )
        return "\n".join(
            [
                tr("[b]Detail[/b]", "[b]详情[/b]"),
                tr(
                    f"Project ID: {escape(project.project_id)}",
                    f"项目 ID：{escape(project.project_id)}",
                ),
                tr(f"Title: {escape(project.title)}", f"标题：{escape(project.title)}"),
                tr(
                    f"State: {escape(workflow.state)}",
                    f"状态：{escape(workflow.state)}",
                ),
                tr(
                    f"Next: {escape(workflow.next_command_short)}",
                    f"下一步：{escape(workflow.next_command_short)}",
                ),
                tr(
                    f"Artifacts: {escape(project_outputs_text(workflow.outputs))}",
                    f"产物：{escape(project_outputs_text(workflow.outputs))}",
                ),
                tr(
                    f"Path: {escape(str(project.project_dir))}",
                    f"路径：{escape(str(project.project_dir))}",
                ),
                tr(
                    f"Open: meeting-asr project review {escape(project.project_id)}",
                    f"打开：meeting-asr project review {escape(project.project_id)}",
                ),
            ]
        )

    def _project(self) -> ProjectListItem | None:
        """Return the selected project, if any."""
        if not self.session.projects:
            return None
        return self.session.projects[self.selected_project_index]


class ProjectPickerApp(_ProjectPickerBase, App[Path | None]):
    """Standalone keyboard-first TUI for choosing a project."""

    CSS = _ProjectPickerBase.CSS
    BINDINGS = _ProjectPickerBase.BINDINGS

    def _finish(self, project_dir: Path | None) -> None:
        """Exit the standalone app with the selected project."""
        self.exit(project_dir)


class ProjectPickerScreen(_ProjectPickerBase, ModalScreen[Path | None]):
    """Embeddable project picker used inside project review."""

    CSS = _ProjectPickerBase.CSS
    BINDINGS = _ProjectPickerBase.BINDINGS

    def _finish(self, project_dir: Path | None) -> None:
        """Dismiss the embedded picker with the selected project."""
        self.dismiss(project_dir)


def load_project_picker_session(
    projects_dir: Path | None = None,
) -> ProjectPickerSession:
    """
    Load projects for the picker TUI.

    Args:
        projects_dir: Optional projects parent directory.

    Returns:
        Project picker session.
    """
    result = list_projects(projects_dir)
    return ProjectPickerSession(
        projects_dir=result.projects_dir, projects=result.projects
    )


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
        workflow = load_project_workflow_summary(
            project.project_dir, project_ref=project.project_id
        )
        lines.append(
            f"{project.project_id} | {workflow.state} | {workflow.next_command_short} | "
            f"{project_outputs_text(workflow.outputs)} | {project.title} | {project.project_dir}"
        )
    return "\n".join(lines)
