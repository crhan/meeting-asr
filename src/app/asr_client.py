"""DashScope Fun-ASR asynchronous task wrapper."""

from __future__ import annotations

from typing import Any

import dashscope
import requests
from dashscope.audio.asr import Transcription

from app.config import Settings
from app.utils import retry


def submit_transcription(
    *,
    settings: Settings,
    file_url: str,
    model: str,
    language_hints: list[str],
    speaker_count: int | None,
    timestamp_alignment_enabled: bool,
    disfluency_removal_enabled: bool,
) -> Any:
    """
    Submit a DashScope asynchronous transcription task.

    Args:
        settings: Runtime settings.
        file_url: Public or signed HTTPS audio URL.
        model: DashScope ASR model.
        language_hints: Language hints.
        speaker_count: Optional diarization hint.
        timestamp_alignment_enabled: Enable timestamp alignment.
        disfluency_removal_enabled: Remove disfluencies.

    Returns:
        DashScope task response.
    """
    _configure_dashscope(settings)
    kwargs: dict[str, Any] = {
        "model": model,
        "file_urls": [file_url],
        "diarization_enabled": True,
        "timestamp_alignment_enabled": timestamp_alignment_enabled,
        "disfluency_removal_enabled": disfluency_removal_enabled,
    }
    if language_hints:
        kwargs["language_hints"] = language_hints
    if speaker_count is not None:
        kwargs["speaker_count"] = speaker_count
    response = Transcription.async_call(**kwargs)
    _raise_for_task_error(response, stage="submit")
    return response


def wait_transcription(*, settings: Settings, task: Any) -> Any:
    """
    Wait for a DashScope transcription task.

    Args:
        settings: Runtime settings.
        task: Submission response.

    Returns:
        DashScope wait response.
    """
    _configure_dashscope(settings)
    response = Transcription.wait(task=task)
    _raise_for_task_error(response, stage="wait")
    _check_subtasks(response)
    return response


def download_transcription_json(wait_response: Any) -> dict:
    """
    Download the JSON pointed to by ``transcription_url``.

    Args:
        wait_response: Completed DashScope response.

    Returns:
        Downloaded JSON object.
    """
    url = _extract_transcription_url(wait_response)

    def _download() -> dict:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("transcription_url did not return a JSON object.")
        return payload

    return retry(_download, attempts=3, delay_seconds=1.0)


def _configure_dashscope(settings: Settings) -> None:
    """Set DashScope API key and optional base URL."""
    dashscope.api_key = settings.dashscope_api_key
    if settings.dashscope_base_url:
        for attr in ("base_http_api_url", "base_url"):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, settings.dashscope_base_url)


def _raise_for_task_error(response: Any, *, stage: str) -> None:
    """Raise when DashScope reports a failed request."""
    status_code = getattr(response, "status_code", None)
    if status_code and int(status_code) >= 400:
        message = getattr(response, "message", None) or getattr(response, "code", None) or response
        raise RuntimeError(f"DashScope {stage} failed: {message}")


def _check_subtasks(response: Any) -> None:
    """Raise when any subtask status is failed."""
    output = getattr(response, "output", None)
    if isinstance(output, dict):
        subtasks = output.get("results") or output.get("subtasks") or []
    else:
        subtasks = getattr(output, "results", None) or getattr(output, "subtasks", None) or []
    for index, subtask in enumerate(subtasks):
        status = _get_field(subtask, "subtask_status") or _get_field(subtask, "status")
        if status and str(status).upper() not in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
            raise RuntimeError(f"DashScope subtask {index} failed: {subtask}")


def _extract_transcription_url(response: Any) -> str:
    """Extract transcription_url from a completed response."""
    output = getattr(response, "output", None)
    containers = [output] if output is not None else []
    if isinstance(output, dict):
        containers.extend(output.get("results") or [])
    for container in containers:
        url = _get_field(container, "transcription_url")
        if url:
            return str(url)
    raise RuntimeError("DashScope task completed but transcription_url is missing.")


def _get_field(value: Any, key: str) -> Any:
    """Read a field from dict or object."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
