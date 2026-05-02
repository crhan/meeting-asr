"""Tests for shared CLI output settings."""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.presentation.cli.output import (
    cli_console,
    configure_cli_output,
    should_disable_color,
    should_enable_verbose_logs,
)
from app.utils import configure_logging

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_cli_output() -> None:
    """
    Reset process-wide output settings around each test.

    Yields:
        None.
    """
    configure_cli_output(no_color=False)
    configure_logging(verbose=False)
    yield
    configure_cli_output(no_color=False)
    configure_logging(verbose=False)


def test_cli_output_uses_color_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default human output should keep Rich color support enabled."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    assert should_disable_color() is False
    assert cli_console().no_color is False


def test_cli_output_honors_no_color_option(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root --no-color option should disable Rich colors globally."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    configure_cli_output(no_color=True)

    assert should_disable_color() is True
    assert cli_console().no_color is True


def test_cli_output_honors_no_color_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """NO_COLOR should disable styled output even without a CLI flag."""
    monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.setenv("TERM", "xterm-256color")

    assert should_disable_color() is True


def test_cli_output_disables_color_for_dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dumb terminals should get plain Rich output."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert should_disable_color() is True


def test_root_no_color_option_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The root command should accept --no-color before subcommands."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = runner.invoke(app, ["--no-color", "paths", "--json"])

    assert result.exit_code == 0
    assert should_disable_color() is True


def test_root_verbose_option_enables_debug_logging(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The root --verbose option should enable process-wide diagnostic logging."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = runner.invoke(app, ["--verbose", "paths", "--json"])

    assert result.exit_code == 0
    assert should_enable_verbose_logs() is True
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
