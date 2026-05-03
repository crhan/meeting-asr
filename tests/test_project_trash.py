"""Tests for project trash lifecycle commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project, load_manifest

runner = CliRunner()


def test_project_trash_restore_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A safely deleted project should be restorable from trash."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir, "Review Me")
    manifest = load_manifest(project_dir)

    delete_result = runner.invoke(app, ["project", "delete", manifest.project_id, "--projects-dir", str(projects_dir), "--yes"])
    trash_list_result = runner.invoke(app, ["project", "trash", "list"])
    trash_plain_result = runner.invoke(app, ["project", "trash", "list", "--plain"])
    trash_json_result = runner.invoke(app, ["project", "trash", "list", "--json"])
    trash_payload = json.loads(trash_json_result.output)
    project_missing_after_delete = not project_dir.exists()
    restore_result = runner.invoke(
        app,
        ["project", "trash", "restore", manifest.project_id, "--projects-dir", str(projects_dir)],
    )
    project_list_result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir)])

    assert delete_result.exit_code == 0
    assert "Project moved to trash." in delete_result.output
    assert "meeting-asr project trash restore" in delete_result.output
    assert project_missing_after_delete
    assert trash_list_result.exit_code == 0
    assert "Project ID" in trash_list_result.output
    assert "No." not in trash_list_result.output
    assert "Review Me" in trash_list_result.output
    assert trash_plain_result.exit_code == 0
    assert trash_plain_result.output.splitlines()[0] == "project_id\tstatus\ttrashed\ttitle\ttrash_dir"
    assert f"{manifest.project_id}\tcreated\t" in trash_plain_result.output
    assert "Review Me" in trash_plain_result.output
    assert "╭" not in trash_plain_result.output
    assert trash_json_result.exit_code == 0
    assert trash_payload["count"] == 1
    assert "number" not in trash_payload["projects"][0]
    assert trash_payload["projects"][0]["project_id"] == manifest.project_id
    assert trash_payload["projects"][0]["title"] == "Review Me"
    assert trash_payload["projects"][0]["restore_project_dir"] == str(project_dir.resolve())
    assert restore_result.exit_code == 0
    assert "Project restored." in restore_result.output
    assert project_dir.exists()
    assert "Review Me" in project_list_result.output


def test_project_trash_purge_removes_trashed_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Purge should physically remove a project that is already in trash."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir, "Purge Me")
    manifest = load_manifest(project_dir)

    delete_result = runner.invoke(app, ["project", "delete", manifest.project_id, "--projects-dir", str(projects_dir), "--yes"])
    purge_result = runner.invoke(app, ["project", "trash", "purge", manifest.project_id, "--yes"])
    trash_list_result = runner.invoke(app, ["project", "trash", "list"])

    assert delete_result.exit_code == 0
    assert purge_result.exit_code == 0
    assert "Trashed project permanently deleted." in purge_result.output
    assert "Purge Me" in purge_result.output
    assert "No trashed projects found." in trash_list_result.output


def test_project_trash_cleanup_can_remove_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cleanup with age zero should empty the trash explicitly."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir, "Cleanup Me")
    manifest = load_manifest(project_dir)

    delete_result = runner.invoke(app, ["project", "delete", manifest.project_id, "--projects-dir", str(projects_dir), "--yes"])
    cleanup_result = runner.invoke(app, ["project", "trash", "cleanup", "--older-than-days", "0", "--yes"])
    trash_list_result = runner.invoke(app, ["project", "trash", "list"])

    assert delete_result.exit_code == 0
    assert cleanup_result.exit_code == 0
    assert "Removed: 1" in cleanup_result.output
    assert "Cleanup Me" in cleanup_result.output
    assert "No trashed projects found." in trash_list_result.output


def _sample_project(tmp_path: Path, projects_dir: Path, title: str) -> Path:
    """
    Create a small project for trash tests.

    Args:
        tmp_path: Pytest temporary directory.
        projects_dir: Parent project directory.
        title: Project title.

    Returns:
        Created project directory.
    """
    source = tmp_path / f"{title}.mp4"
    source.write_bytes(f"fake video {title}".encode("utf-8"))
    project_dir = projects_dir / title.replace(" ", "-").lower()
    create_project(
        source,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir
