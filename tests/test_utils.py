"""Tests for shared utility helpers."""

from __future__ import annotations

import logging
import sys

import pytest
import requests

from app.utils import configure_logging, is_retryable_exception, retry, suppress_noisy_dependency_info_logs


def test_retry_retries_timeout_then_returns_value() -> None:
    """Transient request timeouts should be retried."""
    calls = 0

    def operation() -> str:
        """
        Fail once with a timeout, then succeed.

        Returns:
            Success marker.
        """
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.Timeout("timed out")
        return "ok"

    assert retry(operation, attempts=3, delay_seconds=0) == "ok"
    assert calls == 2


def test_retry_does_not_retry_configuration_errors() -> None:
    """Configuration errors should fail immediately instead of sleeping and retrying."""
    calls = 0

    def operation() -> str:
        """
        Always raise a configuration-shaped error.

        Returns:
            Unused success marker.
        """
        nonlocal calls
        calls += 1
        raise ValueError("Missing required config: dashscope.api_key")

    with pytest.raises(ValueError, match="dashscope.api_key"):
        retry(operation, attempts=3, delay_seconds=0)
    assert calls == 1


def test_retry_exhausts_retryable_http_status() -> None:
    """Retryable HTTP status codes should be retried before a wrapper error is raised."""
    response = requests.Response()
    response.status_code = 503
    calls = 0

    def operation() -> str:
        """
        Always raise a retryable HTTP error.

        Returns:
            Unused success marker.
        """
        nonlocal calls
        calls += 1
        raise requests.HTTPError("Service unavailable", response=response)

    with pytest.raises(RuntimeError, match="2 retryable attempts"):
        retry(operation, attempts=2, delay_seconds=0)
    assert calls == 2


def test_is_retryable_exception_rejects_bad_request() -> None:
    """HTTP 400 is a caller/config problem, not a transient retry problem."""
    response = requests.Response()
    response.status_code = 400

    assert not is_retryable_exception(requests.HTTPError("Bad request", response=response))


def test_configure_logging_suppresses_noisy_dependency_info() -> None:
    """Default logging should not let dependency INFO messages break progress UI."""
    configure_logging()

    assert logging.getLogger("speechbrain.utils.fetching").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("huggingface_hub").getEffectiveLevel() == logging.WARNING


def test_suppress_noisy_dependency_info_logs_blocks_logger_resets(capsys: pytest.CaptureFixture[str]) -> None:
    """Dependency INFO should stay hidden even when the library resets its logger."""
    logger = logging.getLogger("speechbrain.utils.fetching")
    child_logger = logging.getLogger("speechbrain.core")
    original_level = logger.level
    original_child_level = child_logger.level
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.handlers = [stream_handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        with suppress_noisy_dependency_info_logs():
            logger.setLevel(logging.INFO)
            child_logger.setLevel(logging.INFO)
            logger.info("Fetch hyperparams.yaml")
            logger.warning("real warning")
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate
        logger.setLevel(original_level)
        child_logger.setLevel(original_child_level)

    captured = capsys.readouterr()
    assert "Fetch hyperparams.yaml" not in captured.err
    assert "real warning" in captured.err
    assert logger.getEffectiveLevel() >= logging.WARNING
    assert child_logger.getEffectiveLevel() >= logging.WARNING
