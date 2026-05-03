"""Tests for root command boundaries."""

from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_root_version_option_prints_version() -> None:
    """Root CLI should expose a stable --version option."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("meeting-asr ")


def test_root_paths_command_prints_json(monkeypatch, tmp_path) -> None:
    """Global state paths should be discoverable for humans and scripts."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = runner.invoke(app, ["paths", "--json"])

    assert result.exit_code == 0
    assert '"key": "projects"' in result.output
    assert str(tmp_path / "data" / "meeting-asr" / "projects") in result.output


def test_root_help_command_prints_root_and_nested_help() -> None:
    """Git-like help command should expose root and subcommand help."""
    root_result = runner.invoke(app, ["help"])
    nested_result = runner.invoke(app, ["help", "project", "list"])

    assert root_result.exit_code == 0
    assert "Quick start:" in root_result.output
    assert "meeting-asr project run <video>" in root_result.output
    assert nested_result.exit_code == 0
    assert "Usage: meeting-asr project list" in nested_result.output
    assert "--plain" in nested_result.output


def test_top_level_audio_command_is_not_registered() -> None:
    """Audio preparation should stay under project workflows."""
    result = runner.invoke(app, ["audio", "extract"])

    assert result.exit_code != 0
    assert "No such command" in result.output
