"""Tests for ASR runtime duration baselines."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.core.asr_metrics import (
    ASR_SERVICE,
    AsrWaitObservation,
    estimate_asr_wait_seconds,
    record_asr_wait_observation,
)


def test_record_success_refreshes_baseline(tmp_path: Path) -> None:
    """A successful wait observation should refresh the backend baseline."""
    db_path = tmp_path / "runtime.sqlite"

    record_asr_wait_observation(_observation(audio=60.0, wait=30.0), db_path)
    estimate = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=120.0,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.estimated_seconds == pytest.approx(60.0)
    assert estimate.sample_count == 1
    assert estimate.method == "weighted-ratio"
    assert _baseline_count(db_path) == 1


def test_failed_observation_does_not_create_baseline(tmp_path: Path) -> None:
    """Failed waits are stored but should not influence ETA baselines."""
    db_path = tmp_path / "runtime.sqlite"

    record_asr_wait_observation(
        _observation(audio=60.0, wait=30.0, status="failed"), db_path
    )
    estimate = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=120.0,
        db_path=db_path,
    )

    assert estimate is None
    assert _baseline_count(db_path) == 0


def test_baseline_is_scoped_by_model_and_endpoint(tmp_path: Path) -> None:
    """Different models and endpoints should not share ETA baselines."""
    db_path = tmp_path / "runtime.sqlite"
    record_asr_wait_observation(_observation(audio=60.0, wait=30.0), db_path)

    other_model = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="other-model",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=120.0,
        db_path=db_path,
    )
    other_endpoint = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://example.invalid/api/v1",
        audio_duration_seconds=120.0,
        db_path=db_path,
    )

    assert other_model is None
    assert other_endpoint is None


def test_three_samples_create_weighted_linear_baseline(tmp_path: Path) -> None:
    """Three usable samples should switch the baseline from ratio to linear."""
    db_path = tmp_path / "runtime.sqlite"
    record_asr_wait_observation(_observation(audio=60.0, wait=20.0), db_path)
    record_asr_wait_observation(_observation(audio=120.0, wait=35.0), db_path)
    record_asr_wait_observation(_observation(audio=180.0, wait=50.0), db_path)

    estimate = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=240.0,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.method == "weighted-linear"
    assert estimate.confidence == "medium"
    assert estimate.sample_count == 3
    assert estimate.estimated_seconds == pytest.approx(65.0, rel=0.02)


def test_recent_samples_dominate_old_outliers(tmp_path: Path) -> None:
    """Recent fast runs should pull ETA down even when old slow outliers exist."""
    db_path = tmp_path / "runtime.sqlite"
    old_time = datetime.now(UTC) - timedelta(days=30)

    for index, wait in enumerate((180.0, 220.0, 260.0, 300.0), start=1):
        project_id = f"p-old-{index}"
        record_asr_wait_observation(
            _observation(audio=300.0, wait=wait, project_id=project_id), db_path
        )
        _set_observation_created_at(db_path, project_id, old_time)
    record_asr_wait_observation(
        _observation(audio=4148.0, wait=53.0, project_id="p-recent-1"), db_path
    )
    record_asr_wait_observation(
        _observation(audio=4148.0, wait=54.0, project_id="p-recent-2"), db_path
    )
    record_asr_wait_observation(
        _observation(audio=1555.0, wait=27.0, project_id="p-recent-3"), db_path
    )

    estimate = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=4148.0,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.estimated_seconds < 120.0
    assert estimate.sample_count == 3
    assert estimate.confidence == "medium"


def test_estimate_refreshes_stale_precomputed_baseline(tmp_path: Path) -> None:
    """Estimating should not keep using a stale baseline from an older algorithm."""
    db_path = tmp_path / "runtime.sqlite"
    record_asr_wait_observation(
        _observation(audio=4148.0, wait=53.0, project_id="p-recent-1"), db_path
    )
    record_asr_wait_observation(
        _observation(audio=4148.0, wait=54.0, project_id="p-recent-2"), db_path
    )
    _overwrite_baseline_slope(db_path, slope=0.5)

    estimate = estimate_asr_wait_seconds(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=4148.0,
        db_path=db_path,
    )

    assert estimate is not None
    assert estimate.estimated_seconds < 120.0
    assert _baseline_slope(db_path) < 0.1


def _observation(
    audio: float,
    wait: float,
    status: str = "succeeded",
    project_id: str = "p-demo",
) -> AsrWaitObservation:
    """Build one ASR wait observation fixture."""
    return AsrWaitObservation(
        provider="dashscope",
        service=ASR_SERVICE,
        model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        project_id=project_id,
        task_id="task-demo",
        audio_duration_seconds=audio,
        wait_seconds=wait,
        status=status,
    )


def _baseline_count(db_path: Path) -> int:
    """Return baseline row count from the test database."""
    with sqlite3.connect(db_path) as connection:
        return int(
            connection.execute("SELECT COUNT(*) FROM asr_wait_baselines").fetchone()[0]
        )


def _set_observation_created_at(
    db_path: Path, project_id: str, created_at: datetime
) -> None:
    """Set fixture creation time for a recorded observation."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE asr_wait_observations SET created_at = ? WHERE project_id = ?",
            (created_at.isoformat(timespec="seconds"), project_id),
        )


def _overwrite_baseline_slope(db_path: Path, slope: float) -> None:
    """Overwrite the stored baseline slope to simulate a stale cache."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE asr_wait_baselines
            SET method = 'weighted-linear',
                confidence = 'high',
                intercept_seconds = 0,
                slope_seconds_per_audio_second = ?
            """,
            (slope,),
        )


def _baseline_slope(db_path: Path) -> float:
    """Return the current stored baseline slope."""
    with sqlite3.connect(db_path) as connection:
        return float(
            connection.execute(
                "SELECT slope_seconds_per_audio_second FROM asr_wait_baselines"
            ).fetchone()[0]
        )
