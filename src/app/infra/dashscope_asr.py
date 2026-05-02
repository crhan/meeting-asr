"""DashScope asynchronous ASR task wrapper."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dashscope
import requests
from dashscope.audio.asr import Transcription

from app.asr_models import is_qwen_filetrans_model
from app.config import Settings
from app.utils import retry

PollCallback = Callable[["TranscriptionPollEvent"], None]

SUCCESS_STATUSES = {"SUCCEEDED", "SUCCESS", "COMPLETED"}
FAILED_STATUSES = {"FAILED", "FAILURE", "CANCELED", "CANCELLED", "UNKNOWN"}


@dataclass(frozen=True, slots=True)
class TranscriptionPollEvent:
    """One DashScope transcription polling event."""

    status: str | None
    elapsed_seconds: float
    wait_seconds: float


@dataclass(frozen=True, slots=True)
class SubmittedTranscriptionTask:
    """One submitted transcription task with backend routing metadata."""

    backend: str
    task_id: str
    response: Any


def submit_transcription(
    *,
    settings: Settings,
    file_url: str,
    model: str,
    language_hints: list[str],
    speaker_count: int | None,
    vocabulary_id: str | None,
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
        vocabulary_id: Optional DashScope custom vocabulary ID.
        timestamp_alignment_enabled: Enable timestamp alignment.
        disfluency_removal_enabled: Remove disfluencies.

    Returns:
        DashScope task response.
    """
    if is_qwen_filetrans_model(model):
        return _submit_qwen_filetrans(
            settings=settings,
            file_url=file_url,
            model=model,
            language_hints=language_hints,
        )
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
    if vocabulary_id:
        kwargs["vocabulary_id"] = vocabulary_id
    def _submit() -> Any:
        response = Transcription.async_call(**kwargs)
        _raise_for_task_error(response, stage="submit")
        return response

    return retry(_submit, attempts=3, delay_seconds=1.0)


def _submit_qwen_filetrans(
    *,
    settings: Settings,
    file_url: str,
    model: str,
    language_hints: list[str],
) -> SubmittedTranscriptionTask:
    """
    Submit a Qwen-ASR asynchronous file transcription task.

    Args:
        settings: Runtime settings.
        file_url: Public or signed HTTPS audio URL.
        model: Qwen-ASR file transcription model.
        language_hints: Optional language hints. Qwen accepts one language.

    Returns:
        Submitted task wrapper.
    """
    headers = _qwen_headers(settings)
    parameters: dict[str, Any] = {
        "channel_id": [0],
        "enable_itn": False,
        "enable_words": True,
    }
    if len(language_hints) == 1:
        parameters["language"] = language_hints[0]
    payload = {
        "model": model,
        "input": {"file_url": file_url},
        "parameters": parameters,
    }

    def _submit() -> SubmittedTranscriptionTask:
        response = requests.post(_qwen_submit_url(settings), headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("Qwen-ASR submit response was not a JSON object.")
        task_id = _extract_task_id_from_payload(body)
        return SubmittedTranscriptionTask("qwen_filetrans", task_id, body)

    return retry(_submit, attempts=3, delay_seconds=1.0)


def wait_transcription(
    *,
    settings: Settings,
    task: Any,
    poll_callback: PollCallback | None = None,
) -> Any:
    """
    Wait for a DashScope transcription task.

    Args:
        settings: Runtime settings.
        task: Submission response.
        poll_callback: Optional callback invoked after each status fetch.

    Returns:
        DashScope wait response.
    """
    _configure_dashscope(settings)
    wait_seconds = 1.0
    max_wait_seconds = 5.0
    increment_steps = 3
    step = 0
    started_at = time.monotonic()
    while True:
        step += 1
        if wait_seconds < max_wait_seconds and step % increment_steps == 0:
            wait_seconds = min(wait_seconds * 2, max_wait_seconds)
        response = _fetch_transcription_status(settings, task)
        _raise_for_task_error(response, stage="wait")
        status = _extract_task_status(response)
        _raise_for_failed_status(status, response)
        _emit_poll_event(poll_callback, status, started_at, wait_seconds)
        if _is_success_status(status) or _response_has_no_output(response):
            _check_subtasks(response)
            return response
        time.sleep(wait_seconds)


def _fetch_transcription_status(settings: Settings, task: Any) -> Any:
    """
    Fetch one DashScope transcription status with transient retry.

    Args:
        settings: Runtime settings.
        task: Submission response or task id.

    Returns:
        DashScope status response.
    """
    if isinstance(task, SubmittedTranscriptionTask) and task.backend == "qwen_filetrans":
        return _fetch_qwen_filetrans(settings, task.task_id)

    def _fetch() -> Any:
        response = Transcription.fetch(task=task)
        _raise_for_task_error(response, stage="wait")
        return response

    return retry(_fetch, attempts=3, delay_seconds=2.0)


def _fetch_qwen_filetrans(settings: Settings, task_id: str) -> dict:
    """
    Fetch one Qwen-ASR asynchronous task status.

    Args:
        settings: Runtime settings.
        task_id: DashScope task id.

    Returns:
        Response JSON object.
    """
    headers = _qwen_headers(settings)

    def _fetch() -> dict:
        response = requests.get(_qwen_task_url(settings, task_id), headers=headers, timeout=30)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("Qwen-ASR task response was not a JSON object.")
        return body

    return retry(_fetch, attempts=3, delay_seconds=2.0)


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
        raise RuntimeError(f"DashScope {stage} failed: HTTP {status_code} {message}")


def _check_subtasks(response: Any) -> None:
    """Raise when any subtask status is failed."""
    output = _response_output(response)
    if isinstance(output, dict):
        subtasks = output.get("results") or output.get("subtasks") or []
    else:
        subtasks = getattr(output, "results", None) or getattr(output, "subtasks", None) or []
    for index, subtask in enumerate(subtasks):
        status = _get_field(subtask, "subtask_status") or _get_field(subtask, "status")
        if status and str(status).upper() not in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
            raise RuntimeError(f"DashScope subtask {index} failed: {subtask}")


def _extract_task_status(response: Any) -> str | None:
    """Extract a normalized task status from a DashScope response."""
    output = _response_output(response)
    status = _get_field(output, "task_status") or _get_field(output, "status")
    return str(status).upper() if status else None


def _raise_for_failed_status(status: str | None, response: Any) -> None:
    """Raise when the fetched task status is terminal failure."""
    if status in FAILED_STATUSES:
        raise RuntimeError(f"DashScope transcription task failed with status {status}: {response}")


def _is_success_status(status: str | None) -> bool:
    """Return whether a task status means success."""
    return status in SUCCESS_STATUSES


def _response_has_no_output(response: Any) -> bool:
    """Return whether DashScope returned no output payload."""
    return _response_output(response) is None


def _emit_poll_event(
    poll_callback: PollCallback | None,
    status: str | None,
    started_at: float,
    wait_seconds: float,
) -> None:
    """Emit one polling event when a callback is available."""
    if poll_callback is None:
        return
    poll_callback(
        TranscriptionPollEvent(
            status=status,
            elapsed_seconds=time.monotonic() - started_at,
            wait_seconds=wait_seconds,
        )
    )


def _extract_transcription_url(response: Any) -> str:
    """Extract transcription_url from a completed response."""
    output = _response_output(response)
    containers = [output] if output is not None else []
    if isinstance(output, dict):
        result = output.get("result")
        if isinstance(result, dict):
            containers.append(result)
        containers.extend(output.get("results") or [])
    for container in containers:
        url = _get_field(container, "transcription_url")
        if url:
            return str(url)
    raise RuntimeError("DashScope task completed but transcription_url is missing.")


def _extract_task_id_from_payload(payload: dict[str, Any]) -> str:
    """
    Extract task id from a DashScope REST response payload.

    Args:
        payload: Response payload.

    Returns:
        Task id.
    """
    output = payload.get("output")
    task_id = _get_field(output, "task_id")
    if not task_id:
        raise RuntimeError("Qwen-ASR task submission succeeded but task_id is missing.")
    return str(task_id)


def _qwen_submit_url(settings: Settings) -> str:
    """
    Build the Qwen-ASR async submit URL.

    Args:
        settings: Runtime settings.

    Returns:
        Submit endpoint URL.
    """
    return f"{_base_http_url(settings)}/services/audio/asr/transcription"


def _qwen_task_url(settings: Settings, task_id: str) -> str:
    """
    Build the Qwen-ASR task query URL.

    Args:
        settings: Runtime settings.
        task_id: DashScope task id.

    Returns:
        Task query endpoint URL.
    """
    return f"{_base_http_url(settings)}/tasks/{task_id}"


def _qwen_headers(settings: Settings) -> dict[str, str]:
    """
    Build Qwen-ASR REST headers.

    Args:
        settings: Runtime settings.

    Returns:
        HTTP request headers.
    """
    return {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }


def _base_http_url(settings: Settings) -> str:
    """
    Return DashScope base HTTP URL without a trailing slash.

    Args:
        settings: Runtime settings.

    Returns:
        Base URL.
    """
    return settings.dashscope_base_url.rstrip("/")


def _response_output(response: Any) -> Any:
    """
    Read the output field from a dict or SDK response object.

    Args:
        response: Response payload or object.

    Returns:
        Output field when present.
    """
    return _get_field(response, "output")


def _get_field(value: Any, key: str) -> Any:
    """Read a field from dict or object."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
