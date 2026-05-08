"""Runtime ASR duration baselines stored in SQLite."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_data_dir

SCHEMA_VERSION = 1
METRICS_DIR = "metrics"
METRICS_DB_FILENAME = "runtime.sqlite"
ASR_SERVICE = "transcription"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS asr_wait_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  service TEXT NOT NULL,
  model TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  project_id TEXT,
  task_id TEXT,
  audio_duration_seconds REAL NOT NULL,
  wait_seconds REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_asr_wait_backend_created
  ON asr_wait_observations(provider, service, model, endpoint, created_at);
CREATE INDEX IF NOT EXISTS idx_asr_wait_backend_duration
  ON asr_wait_observations(provider, service, model, endpoint, audio_duration_seconds);

CREATE TABLE IF NOT EXISTS asr_wait_baselines (
  provider TEXT NOT NULL,
  service TEXT NOT NULL,
  model TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  sample_count INTEGER NOT NULL,
  method TEXT NOT NULL,
  confidence TEXT NOT NULL,
  intercept_seconds REAL NOT NULL,
  slope_seconds_per_audio_second REAL NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (provider, service, model, endpoint)
);
"""

MIN_AUDIO_DURATION_SECONDS = 1.0
MIN_WAIT_SECONDS = 1.0
ESTIMATE_LOOKBACK_LIMIT = 300
ESTIMATE_HALF_LIFE_SECONDS = 2 * 24 * 60 * 60
OUTLIER_RATIO_MULTIPLIER = 3.0
ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True, slots=True)
class AsrWaitObservation:
    """One completed ASR wait observation."""

    provider: str
    service: str
    model: str
    endpoint: str
    audio_duration_seconds: float
    wait_seconds: float
    status: str
    project_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class AsrWaitEstimate:
    """Estimated ASR wait duration for one backend and audio length."""

    estimated_seconds: float
    sample_count: int
    method: str
    confidence: str


@dataclass(frozen=True, slots=True)
class AsrWaitBaseline:
    """Precomputed ASR wait baseline for one backend."""

    sample_count: int
    method: str
    confidence: str
    intercept_seconds: float
    slope_seconds_per_audio_second: float


def get_asr_metrics_db_path() -> Path:
    """
    Return the default runtime metrics database path.

    Returns:
        XDG-compliant SQLite path.
    """
    return get_data_dir() / METRICS_DIR / METRICS_DB_FILENAME


def record_asr_wait_observation(observation: AsrWaitObservation, db_path: Path | None = None) -> Path:
    """
    Store one ASR wait observation and refresh the backend baseline.

    Args:
        observation: Runtime observation to store.
        db_path: Optional SQLite database path.

    Returns:
        SQLite database path.
    """
    if observation.audio_duration_seconds < MIN_AUDIO_DURATION_SECONDS:
        raise ValueError("audio_duration_seconds must be >= 1.")
    if observation.wait_seconds < MIN_WAIT_SECONDS:
        raise ValueError("wait_seconds must be >= 1.")
    database_path = _resolve_db_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    provider = _normalize_dimension(observation.provider)
    service = _normalize_dimension(observation.service)
    model = _normalize_dimension(observation.model)
    endpoint = _normalize_dimension(observation.endpoint)
    status = _normalize_dimension(observation.status)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO asr_wait_observations (
              provider, service, model, endpoint, project_id, task_id,
              audio_duration_seconds, wait_seconds, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                service,
                model,
                endpoint,
                observation.project_id,
                observation.task_id,
                float(observation.audio_duration_seconds),
                float(observation.wait_seconds),
                status,
                _now_iso(),
            ),
        )
        if status == "succeeded":
            _refresh_baseline(connection, provider, service, model, endpoint)
    return database_path


def estimate_asr_wait_seconds(
    *,
    provider: str,
    service: str,
    model: str,
    endpoint: str,
    audio_duration_seconds: float,
    db_path: Path | None = None,
) -> AsrWaitEstimate | None:
    """
    Estimate remote ASR wait duration from a precomputed backend baseline.

    Args:
        provider: Remote ASR provider key.
        service: Remote service key.
        model: ASR model key.
        endpoint: Provider endpoint identity.
        audio_duration_seconds: Audio duration for the current request.
        db_path: Optional SQLite database path.

    Returns:
        Estimate when a baseline exists, otherwise None.
    """
    if audio_duration_seconds < MIN_AUDIO_DURATION_SECONDS:
        return None
    _refresh_baseline_for_estimate(provider, service, model, endpoint, db_path)
    baseline = _load_baseline(provider, service, model, endpoint, db_path)
    if baseline is None:
        return None
    estimate = baseline.intercept_seconds + baseline.slope_seconds_per_audio_second * audio_duration_seconds
    return AsrWaitEstimate(
        estimated_seconds=max(MIN_WAIT_SECONDS, estimate),
        sample_count=baseline.sample_count,
        method=baseline.method,
        confidence=baseline.confidence,
    )


def _refresh_baseline(
    connection: sqlite3.Connection,
    provider: str,
    service: str,
    model: str,
    endpoint: str,
) -> None:
    """
    Recompute and persist the baseline for one backend.

    Args:
        connection: Open SQLite connection.
        provider: Normalized remote ASR provider key.
        service: Normalized remote service key.
        model: Normalized ASR model key.
        endpoint: Normalized provider endpoint identity.

    Returns:
        None.
    """
    rows = _load_success_rows(connection, provider, service, model, endpoint)
    if not rows:
        _delete_baseline(connection, provider, service, model, endpoint)
        return
    now = datetime.now(UTC)
    weighted_rows = [(_row_weight(created_at, now), duration, wait) for duration, wait, created_at in rows]
    baseline = _baseline_from_weighted_rows(weighted_rows)
    connection.execute(
        """
        INSERT INTO asr_wait_baselines (
          provider, service, model, endpoint, sample_count, method, confidence,
          intercept_seconds, slope_seconds_per_audio_second, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, service, model, endpoint) DO UPDATE SET
          sample_count = excluded.sample_count,
          method = excluded.method,
          confidence = excluded.confidence,
          intercept_seconds = excluded.intercept_seconds,
          slope_seconds_per_audio_second = excluded.slope_seconds_per_audio_second,
          updated_at = excluded.updated_at
        """,
        (
            provider,
            service,
            model,
            endpoint,
            baseline.sample_count,
            baseline.method,
            baseline.confidence,
            baseline.intercept_seconds,
            baseline.slope_seconds_per_audio_second,
            _now_iso(),
        ),
    )


def _refresh_baseline_for_estimate(
    provider: str,
    service: str,
    model: str,
    endpoint: str,
    db_path: Path | None,
) -> None:
    """Refresh a persisted baseline before estimating with the current algorithm."""
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        _refresh_baseline(
            connection,
            _normalize_dimension(provider),
            _normalize_dimension(service),
            _normalize_dimension(model),
            _normalize_dimension(endpoint),
        )


def _load_baseline(
    provider: str,
    service: str,
    model: str,
    endpoint: str,
    db_path: Path | None,
) -> AsrWaitBaseline | None:
    """
    Load one precomputed backend baseline.

    Args:
        provider: Remote ASR provider key.
        service: Remote service key.
        model: ASR model key.
        endpoint: Provider endpoint identity.
        db_path: Optional SQLite database path.

    Returns:
        Baseline row or None.
    """
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return None
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT sample_count, method, confidence, intercept_seconds, slope_seconds_per_audio_second
            FROM asr_wait_baselines
            WHERE provider = ?
              AND service = ?
              AND model = ?
              AND endpoint = ?
            """,
            (
                _normalize_dimension(provider),
                _normalize_dimension(service),
                _normalize_dimension(model),
                _normalize_dimension(endpoint),
            ),
        ).fetchone()
    if row is None:
        return None
    return AsrWaitBaseline(
        sample_count=int(row[0]),
        method=str(row[1]),
        confidence=str(row[2]),
        intercept_seconds=float(row[3]),
        slope_seconds_per_audio_second=float(row[4]),
    )


def _delete_baseline(
    connection: sqlite3.Connection,
    provider: str,
    service: str,
    model: str,
    endpoint: str,
) -> None:
    """Remove a backend baseline when it has no successful observations."""
    connection.execute(
        """
        DELETE FROM asr_wait_baselines
        WHERE provider = ?
          AND service = ?
          AND model = ?
          AND endpoint = ?
        """,
        (provider, service, model, endpoint),
    )


def _load_success_rows(
    connection: sqlite3.Connection,
    provider: str,
    service: str,
    model: str,
    endpoint: str,
) -> list[tuple[float, float, str]]:
    """
    Load recent successful observations for one backend.

    Args:
        connection: Open SQLite connection.
        provider: Normalized remote ASR provider key.
        service: Normalized remote service key.
        model: Normalized ASR model key.
        endpoint: Normalized provider endpoint identity.

    Returns:
        Rows as ``(audio_duration_seconds, wait_seconds, created_at)``.
    """
    rows = connection.execute(
        """
        SELECT audio_duration_seconds, wait_seconds, created_at
        FROM asr_wait_observations
        WHERE provider = ?
          AND service = ?
          AND model = ?
          AND endpoint = ?
          AND status = 'succeeded'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (provider, service, model, endpoint, ESTIMATE_LOOKBACK_LIMIT),
    ).fetchall()
    return [(float(row[0]), float(row[1]), str(row[2])) for row in rows]


def _baseline_from_weighted_rows(rows: list[tuple[float, float, float]]) -> AsrWaitBaseline:
    """
    Build a persisted baseline from weighted rows.

    Args:
        rows: Rows as ``(weight, audio_duration_seconds, wait_seconds)``.

    Returns:
        Precomputed baseline.
    """
    robust_rows = _robust_ratio_rows(rows)
    if len(robust_rows) >= 3:
        linear = _weighted_linear_baseline(robust_rows)
        if linear is not None:
            return linear
    method = "weighted-ratio"
    return AsrWaitBaseline(
        sample_count=len(robust_rows),
        method=method,
        confidence=_estimate_confidence(len(robust_rows), method),
        intercept_seconds=0.0,
        slope_seconds_per_audio_second=_weighted_ratio_slope(robust_rows),
    )


def _robust_ratio_rows(rows: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """
    Drop extreme wait/audio ratio outliers before building a baseline.

    Args:
        rows: Rows as ``(weight, duration, wait)``.

    Returns:
        Rows that are close enough to the weighted median ratio.
    """
    if len(rows) < 4:
        return rows
    log_ratios = [(math.log(wait / max(MIN_AUDIO_DURATION_SECONDS, duration)), weight) for weight, duration, wait in rows]
    median = _weighted_quantile(log_ratios, 0.5)
    deviations = [(abs(value - median), weight) for value, weight in log_ratios]
    mad = _weighted_quantile(deviations, 0.5)
    threshold = max(math.log(OUTLIER_RATIO_MULTIPLIER), 3.0 * ROBUST_MAD_SCALE * mad)
    filtered = [
        row
        for row in rows
        if abs(math.log(row[2] / max(MIN_AUDIO_DURATION_SECONDS, row[1])) - median) <= threshold
    ]
    return filtered if len(filtered) >= 2 else rows


def _weighted_linear_baseline(rows: list[tuple[float, float, float]]) -> AsrWaitBaseline | None:
    """
    Estimate ``wait = intercept + slope * duration`` with recency weights.

    Args:
        rows: Rows as ``(weight, duration, wait)``.

    Returns:
        Linear baseline when the fit is usable.
    """
    sum_weight = sum(weight for weight, _, _ in rows)
    sum_x = sum(weight * duration for weight, duration, _ in rows)
    sum_y = sum(weight * wait for weight, _, wait in rows)
    sum_xx = sum(weight * duration * duration for weight, duration, _ in rows)
    sum_xy = sum(weight * duration * wait for weight, duration, wait in rows)
    denominator = sum_weight * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-9:
        return None
    slope = (sum_weight * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / sum_weight
    if slope <= 0:
        return None
    method = "weighted-linear"
    return AsrWaitBaseline(
        sample_count=len(rows),
        method=method,
        confidence=_estimate_confidence(len(rows), method),
        intercept_seconds=intercept,
        slope_seconds_per_audio_second=slope,
    )


def _weighted_ratio_slope(rows: list[tuple[float, float, float]]) -> float:
    """
    Estimate seconds of wait per audio second from weighted ratios.

    Args:
        rows: Rows as ``(weight, duration, wait)``.

    Returns:
        Seconds of wait per audio second.
    """
    ratios = [(wait / max(MIN_AUDIO_DURATION_SECONDS, duration), weight) for weight, duration, wait in rows]
    return _weighted_quantile(ratios, 0.5)


def _weighted_quantile(values: list[tuple[float, float]], quantile: float) -> float:
    """
    Return a weighted quantile from ``(value, weight)`` pairs.

    Args:
        values: Values with non-negative weights.
        quantile: Quantile in the inclusive ``[0, 1]`` range.

    Returns:
        Weighted quantile value.
    """
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    total_weight = sum(max(0.0, weight) for _, weight in ordered)
    if total_weight <= 0:
        return ordered[len(ordered) // 2][0]
    threshold = min(1.0, max(0.0, quantile)) * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _estimate_confidence(sample_count: int, method: str) -> str:
    """
    Classify estimate confidence.

    Args:
        sample_count: Historical sample count.
        method: Estimation method name.

    Returns:
        Confidence label.
    """
    if sample_count >= 10 and method == "weighted-linear":
        return "high"
    if sample_count >= 3:
        return "medium"
    return "low"


def _row_weight(created_at: str, now: datetime) -> float:
    """
    Return exponential recency weight for a row.

    Args:
        created_at: ISO timestamp.
        now: Current timestamp.

    Returns:
        Positive weight.
    """
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 1.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_seconds = max(0.0, (now - created.astimezone(UTC)).total_seconds())
    return math.exp(-math.log(2.0) * age_seconds / ESTIMATE_HALF_LIFE_SECONDS)


def _resolve_db_path(db_path: Path | None) -> Path:
    """Resolve a metrics database path."""
    return (db_path or get_asr_metrics_db_path()).expanduser().resolve()


def _configure_connection(connection: sqlite3.Connection) -> None:
    """Apply SQLite pragmas."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create metrics tables and indexes."""
    connection.executescript(SCHEMA_SQL)


def _normalize_dimension(value: str) -> str:
    """Normalize one backend dimension."""
    return value.strip().lower() or "unknown"


def _now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")
