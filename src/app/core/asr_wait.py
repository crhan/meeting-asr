"""ASR wait ETA baseline helpers."""

from __future__ import annotations

import logging
import math

from app.config import Settings
from app.core.progress import CliProgressReporter, emit_progress
from app.infra.dashscope_asr import TranscriptionPollEvent
from app.core.asr_metrics import (
    ASR_SERVICE,
    AsrWaitEstimate,
    AsrWaitObservation,
    estimate_asr_wait_seconds,
    record_asr_wait_observation,
)

LOGGER = logging.getLogger(__name__)


def estimate_dashscope_wait(
    settings: Settings,
    *,
    model: str,
    audio_duration_seconds: float | None,
) -> AsrWaitEstimate | None:
    """
    Estimate DashScope ASR wait time from the persisted baseline.

    Args:
        settings: Runtime settings.
        model: ASR model.
        audio_duration_seconds: Audio duration in seconds.

    Returns:
        Wait estimate when a baseline exists.
    """
    if audio_duration_seconds is None:
        return None
    try:
        return estimate_asr_wait_seconds(
            provider="dashscope",
            service=ASR_SERVICE,
            model=model,
            endpoint=settings.dashscope_base_url,
            audio_duration_seconds=audio_duration_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to estimate ASR wait baseline: %s", exc)
        return None


def record_dashscope_wait(
    settings: Settings,
    *,
    project_id: str,
    model: str,
    task_id: str,
    audio_duration_seconds: float | None,
    wait_seconds: float,
    status: str,
) -> None:
    """
    Record one DashScope ASR wait observation without affecting transcription.

    Args:
        settings: Runtime settings.
        project_id: Project id.
        model: ASR model.
        task_id: DashScope task id.
        audio_duration_seconds: Audio duration in seconds.
        wait_seconds: Observed wait duration.
        status: Observation status.

    Returns:
        None.
    """
    if audio_duration_seconds is None:
        return
    try:
        record_asr_wait_observation(
            AsrWaitObservation(
                provider="dashscope",
                service=ASR_SERVICE,
                model=model,
                endpoint=settings.dashscope_base_url,
                project_id=project_id,
                task_id=task_id,
                audio_duration_seconds=audio_duration_seconds,
                wait_seconds=max(1.0, wait_seconds),
                status=status,
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to record ASR wait observation: %s", exc)


def emit_dashscope_wait_poll(
    progress: CliProgressReporter | None,
    *,
    task_id: str,
    estimate: AsrWaitEstimate | None,
    event: TranscriptionPollEvent,
) -> None:
    """
    Emit progress for one DashScope polling event.

    Args:
        progress: Optional progress reporter.
        task_id: DashScope task id.
        estimate: Optional wait estimate.
        event: Polling event.

    Returns:
        None.
    """
    total = asr_wait_total(estimate, elapsed_seconds=event.elapsed_seconds)
    completed = None
    if total is not None:
        completed = min(max(0, math.floor(event.elapsed_seconds)), max(0, total - 1))
    emit_progress(
        progress,
        asr_wait_description(task_id, estimate, event.status),
        total=total,
        completed=completed,
    )


def asr_wait_total(
    estimate: AsrWaitEstimate | None, elapsed_seconds: float = 0.0
) -> int | None:
    """
    Return progress total for an estimated ASR wait.

    Args:
        estimate: Optional wait estimate.
        elapsed_seconds: Already elapsed wait seconds.

    Returns:
        Integer progress total or None when no baseline exists.
    """
    if estimate is None:
        return None
    return max(
        1, math.ceil(estimate.estimated_seconds), math.floor(elapsed_seconds) + 1
    )


def asr_wait_description(
    task_id: str, estimate: AsrWaitEstimate | None, status: str | None
) -> str:
    """
    Build the DashScope wait progress description.

    Args:
        task_id: DashScope task id.
        estimate: Optional wait estimate.
        status: Optional fetched task status.

    Returns:
        Human-readable progress description.
    """
    parts = [f"Waiting for DashScope ASR ({task_id})"]
    if status:
        parts.append(status)
    if estimate is None:
        parts.append("baseline: collecting")
    else:
        parts.append(f"ETA ~{_format_duration_short(estimate.estimated_seconds)}")
        parts.append(f"{estimate.confidence} n={estimate.sample_count}")
    return " | ".join(parts)


def _format_duration_short(seconds: float) -> str:
    """
    Format a duration for compact progress output.

    Args:
        seconds: Duration in seconds.

    Returns:
        Compact duration string.
    """
    value = max(0, int(seconds))
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
