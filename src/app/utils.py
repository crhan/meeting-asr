"""General filesystem, logging, and retry utilities."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import requests

T = TypeVar("T")

NOISY_LOGGERS = (
    "dashscope",
    "httpx",
    "httpcore",
    "urllib3",
    "speechbrain",
    "speechbrain.utils.fetching",
    "huggingface_hub",
    "hyperpyyaml",
)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure standard logging for CLI commands.

    Args:
        verbose: Enable DEBUG logging when true.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", force=True)
    _set_noisy_logger_levels(level)


@contextmanager
def suppress_noisy_dependency_info_logs() -> Iterator[None]:
    """
    Hide noisy third-party INFO logs during dependency initialization.

    Some libraries reset their own loggers while loading models. Temporarily
    disabling INFO keeps progress rendering clean without hiding warnings.

    Yields:
        None.
    """
    disabled_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(disabled_level)
        _set_noisy_logger_levels(logging.WARNING)


def _set_noisy_logger_levels(level: int) -> None:
    """
    Set dependency logger levels in one place.

    Args:
        level: Logging level to apply to noisy dependencies.

    Returns:
        None.
    """
    for logger_name in _iter_noisy_logger_names():
        logging.getLogger(logger_name).setLevel(level)


def _iter_noisy_logger_names() -> set[str]:
    """
    Return configured noisy loggers plus already-created descendants.

    Returns:
        Logger names to suppress.
    """
    logger_names = set(NOISY_LOGGERS)
    for logger_name in logging.Logger.manager.loggerDict:
        if _is_noisy_logger_name(logger_name):
            logger_names.add(logger_name)
    return logger_names


def _is_noisy_logger_name(logger_name: str) -> bool:
    """
    Return whether a logger belongs to a known noisy dependency.

    Args:
        logger_name: Logger name.

    Returns:
        True when the logger should be treated as noisy.
    """
    return any(
        logger_name == noisy or logger_name.startswith(f"{noisy}.")
        for noisy in NOISY_LOGGERS
    )


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
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
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
    return safe_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def retry(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 1.0,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    """
    Retry a callable only for transient failures.

    Args:
        operation: Callable to run.
        attempts: Maximum attempts.
        delay_seconds: Initial delay between attempts.
        retryable: Optional predicate for retryable exceptions.

    Returns:
        Callable result.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1.")
    should_retry = retryable or is_retryable_exception
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            if not should_retry(exc):
                raise
            last_error = exc
            if index + 1 == attempts:
                break
            time.sleep(delay_seconds * (index + 1))
    raise RuntimeError(
        f"Operation failed after {attempts} retryable attempts: {last_error}"
    ) from last_error


def is_retryable_exception(exc: Exception) -> bool:
    """
    Return whether an exception is likely to be fixed by retrying.

    Args:
        exc: Exception raised by an operation.

    Returns:
        True for transient network/service failures.
    """
    if isinstance(
        exc, (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError)
    ):
        return True
    if isinstance(exc, requests.HTTPError):
        return _status_code_is_retryable(_response_status(exc.response))
    if isinstance(exc, requests.RequestException):
        return _status_code_is_retryable(
            _response_status(getattr(exc, "response", None))
        )
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if _status_code_is_retryable(_coerce_int(status)):
        return True
    return _message_looks_retryable(str(exc))


def _response_status(response: object | None) -> int | None:
    """
    Read an HTTP status code from a response-like object.

    Args:
        response: Response object or None.

    Returns:
        Integer status code when available.
    """
    if response is None:
        return None
    return _coerce_int(getattr(response, "status_code", None))


def _coerce_int(value: object | None) -> int | None:
    """
    Convert a value to int when possible.

    Args:
        value: Candidate value.

    Returns:
        Integer or None.
    """
    try:
        return int(value) if value is not None else None
    except TypeError, ValueError:
        return None


def _status_code_is_retryable(status_code: int | None) -> bool:
    """
    Return whether an HTTP status code is transient.

    Args:
        status_code: HTTP status code.

    Returns:
        True for retryable status codes.
    """
    return status_code in {408, 425, 429, 500, 502, 503, 504}


def _message_looks_retryable(message: str) -> bool:
    """
    Match common transient service failure text.

    Args:
        message: Exception message.

    Returns:
        True when the text suggests retrying may help.
    """
    lowered = message.lower()
    retryable_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
        "too many requests",
        "rate limit",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "internal server error",
        "http 408",
        "http 425",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    return any(marker in lowered for marker in retryable_markers)


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
