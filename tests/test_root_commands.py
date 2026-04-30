"""Tests for root command boundaries."""

from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_top_level_audio_command_is_not_registered() -> None:
    """Audio preparation should stay under project workflows."""
    result = runner.invoke(app, ["audio", "extract"])

    assert result.exit_code != 0
    assert "No such command" in result.output
