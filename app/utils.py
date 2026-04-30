"""General filesystem, logging, and retry utilities."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def configure_logging(verbose: bool = False) -> None:
    """
    Configure standard logging for CLI commands.

    Args:
        verbose: Enable DEBUG logging when true.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


def ensure_directory(path: Path) -> Path:
    """
    Create a directory if needed.

    Args:
        path: Directory path.

    Returns:
        The created path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_text(path: Path, content: str) -> Path:
    """
    Write text after creating parent directories.

    Args:
        path: Output file path.
        content: Text content.

    Returns:
        Written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def safe_write_json(path: Path, payload: object) -> Path:
    """
    Write JSON after creating parent directories.

    Args:
        path: Output file path.
        payload: JSON-serializable object.

    Returns:
        Written path.
    """
    return safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def retry(operation: Callable[[], T], *, attempts: int = 3, delay_seconds: float = 1.0) -> T:
    """
    Retry a callable a small fixed number of times.

    Args:
        operation: Callable to run.
        attempts: Maximum attempts.
        delay_seconds: Initial delay between attempts.

    Returns:
        Callable result.
    """
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if index + 1 == attempts:
                break
            time.sleep(delay_seconds * (index + 1))
    raise RuntimeError(f"Operation failed after {attempts} attempts: {last_error}") from last_error


def format_ms_timestamp(ms: int) -> str:
    """
    Format milliseconds as HH:MM:SS.mmm.

    Args:
        ms: Milliseconds.

    Returns:
        Human-readable timestamp.
    """
    value = max(0, int(ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
