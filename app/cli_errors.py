"""Small helpers for readable CLI failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import click
import typer

T = TypeVar("T")


def run_with_cli_errors(operation: Callable[[], T]) -> T:
    """
    Run an operation and convert unexpected exceptions into readable CLI errors.

    Args:
        operation: Callable to execute.

    Returns:
        The callable result.
    """
    try:
        return operation()
    except (click.ClickException, typer.Exit):
        raise
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
