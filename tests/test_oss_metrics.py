"""Tests for OSS upload throughput baselines."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.core.oss_metrics import (
    OSS_UPLOAD_PROVIDER,
    OssUploadObservation,
    estimate_oss_upload_seconds,
    record_oss_upload_observation,
)


def test_record_success_refreshes_upload_baseline(tmp_path: Path) -> None:
    """A successful upload should refresh the throughput baseline."""
    db_path = tmp_path / "runtime.sqlite"

    record_oss_upload_observation(_observation(size=1_000, seconds=2.0), db_path)
    estimate = estimate_oss_upload_seconds(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://oss.example.com",
        bucket_name="meeting-bucket",
        size_bytes=2_000,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.estimated_seconds == pytest.approx(4.0)
    assert estimate.sample_count == 1
    assert estimate.confidence == "low"
    assert _baseline_count(db_path) == 1


def test_failed_upload_does_not_create_baseline(tmp_path: Path) -> None:
    """Failed uploads should be stored but should not affect ETA estimates."""
    db_path = tmp_path / "runtime.sqlite"

    record_oss_upload_observation(_observation(size=1_000, seconds=2.0, status="failed"), db_path)
    estimate = estimate_oss_upload_seconds(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://oss.example.com",
        bucket_name="meeting-bucket",
        size_bytes=2_000,
        db_path=db_path,
    )

    assert estimate is None
    assert _baseline_count(db_path) == 0


def test_upload_baseline_is_scoped_by_endpoint_and_bucket(tmp_path: Path) -> None:
    """Different OSS endpoints or buckets should not share upload baselines."""
    db_path = tmp_path / "runtime.sqlite"
    record_oss_upload_observation(_observation(size=1_000, seconds=2.0), db_path)

    other_endpoint = estimate_oss_upload_seconds(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://other-oss.example.com",
        bucket_name="meeting-bucket",
        size_bytes=2_000,
        db_path=db_path,
    )
    other_bucket = estimate_oss_upload_seconds(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://oss.example.com",
        bucket_name="other-bucket",
        size_bytes=2_000,
        db_path=db_path,
    )

    assert other_endpoint is None
    assert other_bucket is None


def test_three_upload_samples_raise_confidence(tmp_path: Path) -> None:
    """Three successful samples should produce a medium-confidence baseline."""
    db_path = tmp_path / "runtime.sqlite"
    record_oss_upload_observation(_observation(size=1_000, seconds=2.0), db_path)
    record_oss_upload_observation(_observation(size=2_000, seconds=4.0), db_path)
    record_oss_upload_observation(_observation(size=3_000, seconds=6.0), db_path)

    estimate = estimate_oss_upload_seconds(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://oss.example.com",
        bucket_name="meeting-bucket",
        size_bytes=4_000,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.sample_count == 3
    assert estimate.confidence == "medium"
    assert estimate.bytes_per_second == pytest.approx(500.0)
    assert estimate.estimated_seconds == pytest.approx(8.0)


def _observation(size: int, seconds: float, status: str = "succeeded") -> OssUploadObservation:
    """Build one OSS upload observation fixture."""
    return OssUploadObservation(
        provider=OSS_UPLOAD_PROVIDER,
        endpoint="https://oss.example.com",
        bucket_name="meeting-bucket",
        project_id="p-demo",
        object_key="meeting-asr/projects/p-demo/audio.wav",
        size_bytes=size,
        upload_seconds=seconds,
        status=status,
    )


def _baseline_count(db_path: Path) -> int:
    """Return baseline row count from the test database."""
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM oss_upload_baselines").fetchone()[0])
