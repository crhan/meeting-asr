"""Small helpers for readable CLI failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import click
import typer

T = TypeVar("T")
AdviceRule = tuple[Callable[[str], bool], str, str, str]


@dataclass(frozen=True, slots=True)
class CliErrorAdvice:
    """Actionable advice for a failed CLI command."""

    problem: str
    doctor_command: str
    detail: str
    retry_exhausted: bool = False


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
        _echo_cli_error(exc)
        raise typer.Exit(code=1) from exc


def build_cli_error_advice(exc: Exception) -> CliErrorAdvice | None:
    """
    Build doctor guidance for configuration, environment, and exhausted transient errors.

    Args:
        exc: Exception raised by the failed command.

    Returns:
        Advice when the CLI can suggest a useful next step.
    """
    messages = _exception_messages(exc)
    lowered = "\n".join(messages).lower()
    retry_exhausted = _looks_like_retry_exhausted(lowered)
    for matcher, problem, doctor_command, detail in _ADVICE_RULES:
        if not matcher(lowered):
            continue
        return CliErrorAdvice(
            problem,
            doctor_command,
            detail,
            retry_exhausted=retry_exhausted,
        )
    if retry_exhausted:
        return CliErrorAdvice(
            "transient network or service failure",
            "meeting-asr doctor",
            "The operation was retried because it looked transient, but it still failed.",
            retry_exhausted=True,
        )
    return None


def _echo_cli_error(exc: Exception) -> None:
    """
    Print a readable CLI error and optional repair advice.

    Args:
        exc: Exception raised by the failed command.

    Returns:
        None.
    """
    typer.echo(f"Error: {exc}", err=True)
    advice = build_cli_error_advice(exc)
    if advice is None:
        return
    if advice.retry_exhausted:
        typer.echo("Retry: this looked transient and was already retried before failing.", err=True)
    typer.echo(f"Problem: {advice.problem}", err=True)
    typer.echo(f"Diagnosis: {advice.detail}", err=True)
    typer.echo(f"Next step: run `{advice.doctor_command}`.", err=True)
    typer.echo(
        "Agent prompt: Fix the meeting-asr failure above. "
        f"First run `{advice.doctor_command}`, apply the emitted Repair Prompt, "
        "then rerun the failed command. Do not print or commit secrets.",
        err=True,
    )


def _exception_messages(exc: Exception) -> list[str]:
    """
    Collect messages from an exception chain.

    Args:
        exc: Root exception.

    Returns:
        Non-empty message list.
    """
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return messages or [exc.__class__.__name__]


def _looks_like_retry_exhausted(lowered_message: str) -> bool:
    """
    Return whether retry already happened and failed.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True when retry wrapper exhausted all attempts.
    """
    return "retryable attempts" in lowered_message


def _looks_like_voiceprint_dependency(lowered_message: str) -> bool:
    """
    Return whether the error is a local voiceprint dependency problem.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for missing SpeechBrain/Torch dependencies.
    """
    return "local-speechbrain voiceprint embedding requires optional dependencies" in lowered_message


def _looks_like_voiceprint_config(lowered_message: str) -> bool:
    """
    Return whether the error is a voiceprint provider configuration problem.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for provider or endpoint configuration failures.
    """
    markers = (
        "voiceprint.embedding_endpoint",
        "voiceprint.embedding_provider",
        "unsupported voiceprint embedding provider",
        "bailian voiceprint embedding failed",
    )
    return any(marker in lowered_message for marker in markers)


def _looks_like_oss_config_or_access(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by OSS configuration or access.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for OSS config, auth, bucket, or signed URL failures.
    """
    markers = (
        "missing required oss config",
        "oss.access_key_id",
        "oss.access_key_secret",
        "oss.bucket_name",
        "oss.region",
        "oss.endpoint",
        "oss request failed",
        "signaturedoesnotmatch",
        "accessdenied",
        "invalidaccesskeyid",
        "nosuchbucket",
        "file_403_forbidden",
    )
    return any(marker in lowered_message for marker in markers)


def _looks_like_dashscope_config_or_access(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by DashScope configuration or access.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for missing API key or authorization failures.
    """
    if "dashscope.api_key" in lowered_message:
        return True
    if "dashscope" not in lowered_message:
        return False
    auth_markers = ("http 401", "http 403", "unauthorized", "forbidden", "invalid api key")
    return any(marker in lowered_message for marker in auth_markers)


def _looks_like_local_environment(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by missing local tools.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for ffmpeg or preview player problems.
    """
    markers = (
        "ffmpeg was not found",
        "no supported subtitle preview player found",
        "no supported audio preview player found",
    )
    return any(marker in lowered_message for marker in markers)


_ADVICE_RULES: tuple[AdviceRule, ...] = (
    (
        _looks_like_voiceprint_dependency,
        "voiceprint embedding dependencies",
        "meeting-asr doctor --require-voiceprint-embedding",
        "The local voiceprint provider is missing optional dependencies.",
    ),
    (
        _looks_like_voiceprint_config,
        "voiceprint embedding configuration",
        "meeting-asr doctor --require-oss --require-voiceprint-embedding",
        "The configured voiceprint provider cannot run with the current settings.",
    ),
    (
        _looks_like_oss_config_or_access,
        "OSS configuration or access",
        "meeting-asr doctor --oss-upload-probe",
        "The command needs OSS config, bucket access, or signed URL verification.",
    ),
    (
        _looks_like_dashscope_config_or_access,
        "DashScope configuration or access",
        "meeting-asr doctor",
        "The command needs a valid DashScope API key and service access.",
    ),
    (
        _looks_like_local_environment,
        "local environment",
        "meeting-asr doctor",
        "A required local dependency or preview tool is missing.",
    ),
)
