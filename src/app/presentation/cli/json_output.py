"""Helpers for stable machine-readable CLI output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer


def emit_json(payload: dict[str, Any]) -> None:
    """
    Print one JSON payload to stdout.

    Args:
        payload: JSON-compatible command result.

    Returns:
        None.
    """
    typer.echo(json.dumps(_normalize(payload), ensure_ascii=False, indent=2))


def _normalize(value: Any) -> Any:
    """Convert paths and containers into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    return value
