"""Stable plain-text table rendering for script-friendly CLI output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import typer


def echo_plain_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """
    Print a tab-separated table without color or box drawing.

    Args:
        headers: Column names.
        rows: Row values.

    Returns:
        None.
    """
    typer.echo(_plain_line(headers))
    for row in rows:
        typer.echo(_plain_line(row))


def plain_cell(value: object) -> str:
    """
    Normalize one value for tab-separated CLI output.

    Args:
        value: Cell value.

    Returns:
        Plain one-line cell text.
    """
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _plain_line(values: Sequence[object]) -> str:
    """Return one tab-separated plain output line."""
    return "\t".join(plain_cell(value) for value in values)
