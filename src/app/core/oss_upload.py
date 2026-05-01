"""OSS upload ETA baseline helpers."""

from __future__ import annotations

import logging

from app.config import Settings
from app.core.oss_metrics import (
    OSS_UPLOAD_PROVIDER,
    OssUploadEstimate,
    OssUploadObservation,
    estimate_oss_upload_seconds,
    record_oss_upload_observation,
)
from app.core.progress import CliProgressReporter, emit_progress

LOGGER = logging.getLogger(__name__)


def estimate_oss_upload(settings: Settings, *, size_bytes: int) -> OssUploadEstimate | None:
    """
    Estimate OSS upload duration from the persisted throughput baseline.

    Args:
        settings: Runtime settings.
        size_bytes: File size to upload.

    Returns:
        Upload estimate when a baseline exists.
    """
    try:
        return estimate_oss_upload_seconds(
            provider=OSS_UPLOAD_PROVIDER,
            endpoint=settings.oss_endpoint or "unknown",
            bucket_name=settings.oss_bucket_name or "unknown",
            size_bytes=size_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to estimate OSS upload baseline: %s", exc)
        return None


def record_oss_upload(
    settings: Settings,
    *,
    project_id: str | None,
    object_key: str,
    size_bytes: int,
    upload_seconds: float,
    status: str,
) -> None:
    """
    Record one OSS upload observation without affecting the main workflow.

    Args:
        settings: Runtime settings.
        project_id: Optional project id.
        object_key: OSS object key.
        size_bytes: Uploaded file size.
        upload_seconds: Observed upload duration.
        status: Observation status.

    Returns:
        None.
    """
    try:
        record_oss_upload_observation(
            OssUploadObservation(
                provider=OSS_UPLOAD_PROVIDER,
                endpoint=settings.oss_endpoint or "unknown",
                bucket_name=settings.oss_bucket_name or "unknown",
                project_id=project_id,
                object_key=object_key,
                size_bytes=size_bytes,
                upload_seconds=max(0.001, upload_seconds),
                status=status,
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to record OSS upload observation: %s", exc)


def emit_oss_upload_start(
    progress: CliProgressReporter | None,
    *,
    estimate: OssUploadEstimate | None,
    size_bytes: int,
) -> None:
    """
    Emit initial OSS upload progress.

    Args:
        progress: Optional progress reporter.
        estimate: Optional upload estimate.
        size_bytes: Upload size in bytes.

    Returns:
        None.
    """
    emit_progress(progress, oss_upload_description(estimate), total=size_bytes, completed=0)


def emit_oss_upload_progress(
    progress: CliProgressReporter | None,
    *,
    estimate: OssUploadEstimate | None,
    consumed_bytes: int,
    total_bytes: int,
) -> None:
    """
    Emit OSS upload byte progress.

    Args:
        progress: Optional progress reporter.
        estimate: Optional upload estimate.
        consumed_bytes: Uploaded byte count.
        total_bytes: Total byte count.

    Returns:
        None.
    """
    emit_progress(
        progress,
        oss_upload_description(estimate),
        total=max(1, total_bytes),
        completed=max(0, min(consumed_bytes, max(1, total_bytes))),
    )


def oss_upload_description(estimate: OssUploadEstimate | None) -> str:
    """
    Build the OSS upload progress description.

    Args:
        estimate: Optional upload estimate.

    Returns:
        Human-readable progress description.
    """
    if estimate is None:
        return "Uploading audio to OSS | baseline: collecting"
    return (
        f"Uploading audio to OSS | ETA ~{_format_duration_short(estimate.estimated_seconds)} "
        f"| {estimate.confidence} n={estimate.sample_count}"
    )


def _format_duration_short(seconds: float) -> str:
    """Format a duration for compact progress output."""
    value = max(0, int(seconds))
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
