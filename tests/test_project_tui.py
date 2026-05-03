"""Tests for the project picker TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from app.project_manager import create_project, load_manifest
from app.project_tui import (
    ProjectPickerApp,
    ProjectPickerHelpScreen,
    load_project_picker_session,
    render_project_picker_summary,
)


def test_project_picker_session_lists_history(tmp_path: Path) -> None:
    """Project picker data should come from the same project store as project list."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Selector Demo")
    manifest = load_manifest(project_dir)

    session = load_project_picker_session(projects_dir)
    summary = render_project_picker_summary(session)

    assert session.projects_dir == projects_dir.resolve()
    assert [project.project_id for project in session.projects] == [manifest.project_id]
    assert manifest.project_id in summary
    assert "Selector Demo" in summary


def test_project_picker_tui_returns_selected_project(tmp_path: Path) -> None:
    """Enter should return the selected project root to the caller."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Selector Demo")
    session = load_project_picker_session(projects_dir)
    app = ProjectPickerApp(session)

    async def scenario() -> None:
        async with app.run_test() as pilot:
            assert "Selector Demo" in app._project_list_pane()
            assert session.projects[0].project_id in app._project_list_pane()
            assert "List No." not in app._detail_pane()
            assert "meeting-asr project review" in app._detail_pane()
            await pilot.press("enter")

    asyncio.run(scenario())

    assert app.return_value == project_dir.resolve()


def test_project_picker_tui_question_mark_shows_help(tmp_path: Path) -> None:
    """The ? key should open and close a shortcut help modal."""
    session = load_project_picker_session(_sample_project_parent(tmp_path))

    async def scenario() -> None:
        async with ProjectPickerApp(session).run_test() as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_screen = pilot.app.screen
            help_text = str(help_screen.query_one("#project-picker-help", Static).render())

            assert isinstance(help_screen, ProjectPickerHelpScreen)
            assert "Project List Shortcuts" in help_text
            assert "project review PROJECT_ID" in help_text
            assert "project review PROJECT_PATH" in help_text

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, ProjectPickerHelpScreen)

    asyncio.run(scenario())


def _sample_project_parent(tmp_path: Path) -> Path:
    """Create a sample project and return its parent directory."""
    projects_dir = tmp_path / "projects"
    _sample_project(tmp_path, projects_dir=projects_dir, title="Selector Demo")
    return projects_dir


def _sample_project(tmp_path: Path, *, projects_dir: Path, title: str) -> Path:
    """Create a minimal project for picker tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = projects_dir / "project"
    create_project(
        source,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir
