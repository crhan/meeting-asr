"""Shared output settings for human-facing CLI rendering."""

from __future__ import annotations

import os

from rich.console import Console

_NO_COLOR_REQUESTED = False


def configure_cli_output(*, no_color: bool = False) -> None:
    """
    Configure process-wide human output preferences.

    Args:
        no_color: Whether the user explicitly disabled colored Rich output.

    Returns:
        None.
    """
    global _NO_COLOR_REQUESTED
    _NO_COLOR_REQUESTED = no_color


def should_disable_color() -> bool:
    """
    Return whether Rich color output should be disabled.

    Returns:
        True when root options or terminal environment request plain output.
    """
    return _NO_COLOR_REQUESTED or "NO_COLOR" in os.environ or os.environ.get("TERM", "").lower() == "dumb"


def cli_console(*, stderr: bool = False, width: int | None = None) -> Console:
    """
    Build a Rich console using the shared CLI output contract.

    Args:
        stderr: Whether the console should write to stderr.
        width: Optional fixed width for deterministic table rendering.

    Returns:
        Configured Rich console.
    """
    return Console(
        stderr=stderr,
        highlight=False,
        color_system="auto",
        no_color=should_disable_color(),
        width=width,
    )
