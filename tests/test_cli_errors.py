"""Tests for user-facing CLI error advice."""

from __future__ import annotations

import pytest
import requests
import typer

from app.presentation.cli.errors import build_cli_error_advice, run_with_cli_errors


def test_missing_dashscope_config_suggests_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    """Missing DashScope config should point at the default doctor command."""
    with pytest.raises(typer.Exit):
        run_with_cli_errors(
            lambda: _raise(
                ValueError(
                    "Missing required config: dashscope.api_key. "
                    "Run `meeting-asr config set dashscope.api_key <value>`."
                )
            )
        )

    captured = capsys.readouterr()
    assert "Error: Missing required config: dashscope.api_key" in captured.err
    assert "Next step: run `meeting-asr doctor`." in captured.err
    assert "Agent prompt:" in captured.err


def test_missing_oss_config_suggests_oss_probe(capsys: pytest.CaptureFixture[str]) -> None:
    """Missing OSS config should point at the OSS upload probe."""
    with pytest.raises(typer.Exit):
        run_with_cli_errors(lambda: _raise(ValueError("Missing required OSS config: oss.bucket_name")))

    captured = capsys.readouterr()
    assert "Diagnosis: The command needs OSS config" in captured.err
    assert "Next step: run `meeting-asr doctor --oss-upload-probe`." in captured.err


def test_voiceprint_dependency_error_suggests_strict_voiceprint_doctor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing local voiceprint dependencies should point at strict voiceprint doctor."""
    with pytest.raises(typer.Exit):
        run_with_cli_errors(
            lambda: _raise(
                RuntimeError("local-speechbrain voiceprint embedding requires standard dependencies.")
            )
        )

    captured = capsys.readouterr()
    assert "voiceprint embedding dependencies" in captured.err
    assert "meeting-asr doctor --require-voiceprint-embedding" in captured.err


def test_retry_exhausted_error_mentions_retry_before_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    """Exhausted transient errors should say retry already happened."""
    with pytest.raises(typer.Exit):
        run_with_cli_errors(lambda: _raise(_retry_exhausted_error()))

    captured = capsys.readouterr()
    assert "Retry: this looked transient and was already retried before failing." in captured.err
    assert "transient network or service failure" in captured.err


def test_regular_missing_source_file_has_no_doctor_advice() -> None:
    """A bad user path is not a doctor problem."""
    advice = build_cli_error_advice(FileNotFoundError("OSS upload source does not exist: /tmp/missing.wav"))

    assert advice is None


def _retry_exhausted_error() -> RuntimeError:
    """
    Build a retry-wrapper error with the original request error as its cause.

    Returns:
        RuntimeError shaped like ``app.utils.retry`` final failure.
    """
    cause = requests.Timeout("timed out")
    error = RuntimeError("Operation failed after 3 retryable attempts: timed out")
    error.__cause__ = cause
    return error


def _raise(exc: Exception) -> None:
    """
    Raise an exception from inside a lambda-friendly helper.

    Args:
        exc: Exception to raise.

    Returns:
        None.
    """
    raise exc
