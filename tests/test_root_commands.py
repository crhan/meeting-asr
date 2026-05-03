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
    assert "Usage:" in nested_result.output
    assert "meeting-asr project list [OPTIONS]" in nested_result.output
    assert "--plain" in nested_result.output


def test_root_help_command_can_render_chinese(monkeypatch) -> None:
    """Help command should render Meeting-ASR-owned text in Chinese."""
    option_result = runner.invoke(app, ["--lang", "zh", "help", "project", "list"])
    monkeypatch.setenv("MEETING_ASR_LANG", "zh")
    env_result = runner.invoke(app, ["help"])

    assert option_result.exit_code == 0
    assert "用法:" in option_result.output
    assert "列出默认或指定项目库里的项目。" in option_result.output
    assert "--plain                      输出稳定的制表符分隔文本。" in option_result.output
    assert env_result.exit_code == 0
    assert "快速开始:" in env_result.output
    assert "命令:" in env_result.output


def test_root_lang_rejects_invalid_value_without_traceback() -> None:
    """Invalid language should fail as a CLI parameter error."""
    result = runner.invoke(app, ["--lang", "bad", "help"])

    assert result.exit_code != 0
    assert "Invalid value for --lang" in result.output
    assert "Traceback" not in result.output


def test_top_level_audio_command_is_not_registered() -> None:
    """Audio preparation should stay under project workflows."""
    result = runner.invoke(app, ["audio", "extract"])

    assert result.exit_code != 0
    assert "No such command" in result.output
