"""Runtime OSS upload throughput baselines stored in SQLite."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_data_dir

METRICS_DIR = "metrics"
METRICS_DB_FILENAME = "runtime.sqlite"
OSS_UPLOAD_PROVIDER = "aliyun-oss"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS oss_upload_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  bucket_name TEXT NOT NULL,
  project_id TEXT,
  object_key TEXT,
  size_bytes INTEGER NOT NULL,
  upload_seconds REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_oss_upload_backend_created
  ON oss_upload_observations(provider, endpoint, bucket_name, created_at);

CREATE TABLE IF NOT EXISTS oss_upload_baselines (
  provider TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  bucket_name TEXT NOT NULL,
  sample_count INTEGER NOT NULL,
  confidence TEXT NOT NULL,
  bytes_per_second REAL NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (provider, endpoint, bucket_name)
);
"""

MIN_UPLOAD_SIZE_BYTES = 1
MIN_UPLOAD_SECONDS = 0.001
LOOKBACK_LIMIT = 300
HALF_LIFE_SECONDS = 14 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class OssUploadObservation:
    """One completed OSS upload observation."""

    provider: str
    endpoint: str
    bucket_name: str
    size_bytes: int
    upload_seconds: float
    status: str
    project_id: str | None = None
    object_key: str | None = None


@dataclass(frozen=True, slots=True)
class OssUploadBaseline:
    """Precomputed OSS upload throughput baseline."""

    sample_count: int
    confidence: str
    bytes_per_second: float


@dataclass(frozen=True, slots=True)
class OssUploadEstimate:
    """Estimated OSS upload duration for one file."""

    estimated_seconds: float
    sample_count: int
    confidence: str
    bytes_per_second: float


def get_oss_metrics_db_path() -> Path:
    """
    Return the default runtime metrics database path.

    Returns:
        XDG-compliant SQLite path.
    """
    return get_data_dir() / METRICS_DIR / METRICS_DB_FILENAME


def record_oss_upload_observation(
    observation: OssUploadObservation, db_path: Path | None = None
) -> Path:
    """
    Store one OSS upload observation and refresh the backend baseline.

    Args:
        observation: Runtime observation to store.
        db_path: Optional SQLite database path.

    Returns:
        SQLite database path.
    """
    if observation.size_bytes < MIN_UPLOAD_SIZE_BYTES:
        raise ValueError("size_bytes must be >= 1.")
    if observation.upload_seconds < MIN_UPLOAD_SECONDS:
        raise ValueError("upload_seconds must be >= 0.001.")
    database_path = _resolve_db_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    provider = _normalize_dimension(observation.provider)
    endpoint = _normalize_dimension(observation.endpoint)
    bucket_name = _normalize_dimension(observation.bucket_name)
    status = _normalize_dimension(observation.status)
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT INTO oss_upload_observations (
              provider, endpoint, bucket_name, project_id, object_key,
              size_bytes, upload_seconds, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                endpoint,
                bucket_name,
                observation.project_id,
                observation.object_key,
                int(observation.size_bytes),
                float(observation.upload_seconds),
                status,
                _now_iso(),
            ),
        )
        if status == "succeeded":
            _refresh_baseline(connection, provider, endpoint, bucket_name)
    return database_path


def estimate_oss_upload_seconds(
    *,
    provider: str,
    endpoint: str,
    bucket_name: str,
    size_bytes: int,
    db_path: Path | None = None,
) -> OssUploadEstimate | None:
    """
    Estimate OSS upload duration from a precomputed throughput baseline.

    Args:
        provider: OSS provider key.
        endpoint: OSS endpoint identity.
        bucket_name: OSS bucket name.
        size_bytes: File size to upload.
        db_path: Optional SQLite database path.

    Returns:
        Estimate when a baseline exists, otherwise None.
    """
    if size_bytes < MIN_UPLOAD_SIZE_BYTES:
        return None
    baseline = _load_baseline(provider, endpoint, bucket_name, db_path)
    if baseline is None:
        return None
    return OssUploadEstimate(
        estimated_seconds=max(
            MIN_UPLOAD_SECONDS, size_bytes / baseline.bytes_per_second
        ),
        sample_count=baseline.sample_count,
        confidence=baseline.confidence,
        bytes_per_second=baseline.bytes_per_second,
    )


def _refresh_baseline(
    connection: sqlite3.Connection, provider: str, endpoint: str, bucket_name: str
) -> None:
    """Refresh one OSS upload throughput baseline."""
    rows = _load_success_rows(connection, provider, endpoint, bucket_name)
    now = datetime.now(UTC)
    weighted_speed_sum = 0.0
    weight_sum = 0.0
    for size_bytes, upload_seconds, created_at in rows:
        weight = _row_weight(created_at, now)
        weighted_speed_sum += (
            weight * size_bytes / max(MIN_UPLOAD_SECONDS, upload_seconds)
        )
        weight_sum += weight
    bytes_per_second = max(1.0, weighted_speed_sum / weight_sum)
    sample_count = len(rows)
    connection.execute(
        """
        INSERT INTO oss_upload_baselines (
          provider, endpoint, bucket_name, sample_count, confidence, bytes_per_second, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, endpoint, bucket_name) DO UPDATE SET
          sample_count = excluded.sample_count,
          confidence = excluded.confidence,
          bytes_per_second = excluded.bytes_per_second,
          updated_at = excluded.updated_at
        """,
        (
            provider,
            endpoint,
            bucket_name,
            sample_count,
            _confidence(sample_count),
            bytes_per_second,
            _now_iso(),
        ),
    )


def _load_baseline(
    provider: str,
    endpoint: str,
    bucket_name: str,
    db_path: Path | None,
) -> OssUploadBaseline | None:
    """Load one precomputed OSS upload baseline."""
    database_path = _resolve_db_path(db_path)
    if not database_path.exists():
        return None
    with sqlite3.connect(database_path) as connection:
        _configure_connection(connection)
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT sample_count, confidence, bytes_per_second
            FROM oss_upload_baselines
            WHERE provider = ? AND endpoint = ? AND bucket_name = ?
            """,
            (
                _normalize_dimension(provider),
                _normalize_dimension(endpoint),
                _normalize_dimension(bucket_name),
            ),
        ).fetchone()
    if row is None:
        return None
    return OssUploadBaseline(
        sample_count=int(row[0]), confidence=str(row[1]), bytes_per_second=float(row[2])
    )


def _load_success_rows(
    connection: sqlite3.Connection,
    provider: str,
    endpoint: str,
    bucket_name: str,
) -> list[tuple[int, float, str]]:
    """Load recent successful upload observations for one backend."""
    rows = connection.execute(
        """
        SELECT size_bytes, upload_seconds, created_at
        FROM oss_upload_observations
        WHERE provider = ?
          AND endpoint = ?
          AND bucket_name = ?
          AND status = 'succeeded'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (provider, endpoint, bucket_name, LOOKBACK_LIMIT),
    ).fetchall()
    return [(int(row[0]), float(row[1]), str(row[2])) for row in rows]


def _confidence(sample_count: int) -> str:
    """Return a simple confidence label from sample count."""
    if sample_count >= 10:
        return "high"
    if sample_count >= 3:
        return "medium"
    return "low"


def _row_weight(created_at: str, now: datetime) -> float:
    """Return exponential recency weight for a row."""
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 1.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age_seconds = max(0.0, (now - created.astimezone(UTC)).total_seconds())
    return math.exp(-age_seconds / HALF_LIFE_SECONDS)


def _resolve_db_path(db_path: Path | None) -> Path:
    """Resolve a metrics database path."""
    return (db_path or get_oss_metrics_db_path()).expanduser().resolve()


def _configure_connection(connection: sqlite3.Connection) -> None:
    """Apply SQLite pragmas."""
    connection.execute("PRAGMA journal_mode = WAL")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create metrics tables and indexes."""
    connection.executescript(SCHEMA_SQL)


def _normalize_dimension(value: str | None) -> str:
    """Normalize one backend dimension."""
    return (value or "unknown").strip().lower() or "unknown"


def _now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")
