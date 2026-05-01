"""Aliyun OSS upload and signed URL helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import warnings
from uuid import uuid4

from app.config import Settings, load_settings

SIGNED_URL_EXPIRES_SECONDS = 24 * 60 * 60
DEFAULT_OSS_PREFIX = "meeting-asr/uploads"


def import_oss2():
    """
    Import oss2 while hiding a dependency SyntaxWarning on Python 3.14.

    Returns:
        Imported ``oss2`` module.
    """
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        message=r".*invalid escape sequence.*",
    )
    import oss2

    return oss2


def build_oss_bucket(settings: Settings):
    """
    Build an OSS bucket client.

    Args:
        settings: Runtime settings with OSS values.

    Returns:
        ``oss2.Bucket`` instance.
    """
    oss2 = import_oss2()

    missing = [
        name
        for name, value in (
            ("oss.access_key_id", settings.oss_access_key_id),
            ("oss.access_key_secret", settings.oss_access_key_secret),
            ("oss.endpoint", settings.oss_endpoint),
            ("oss.bucket_name", settings.oss_bucket_name),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required OSS config: {', '.join(missing)}")
    auth = oss2.Auth(settings.oss_access_key_id, settings.oss_access_key_secret)
    return oss2.Bucket(auth, settings.oss_endpoint, settings.oss_bucket_name)


def upload_file_to_oss(
    local_path: str | Path,
    object_name: str | None = None,
    *,
    settings: Settings | None = None,
    expires_seconds: int = SIGNED_URL_EXPIRES_SECONDS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    """
    Upload a file to private OSS and return a signed GET URL.

    Args:
        local_path: Local file path.
        object_name: Optional OSS object key.
        settings: Optional loaded settings.
        expires_seconds: Signed URL lifetime.
        progress_callback: Optional callback receiving uploaded and total bytes.

    Returns:
        Signed HTTPS URL.
    """
    source = Path(local_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"OSS upload source does not exist: {source}")
    resolved_settings = settings or load_settings(require_oss=True)
    bucket = build_oss_bucket(resolved_settings)
    key = object_name or f"{DEFAULT_OSS_PREFIX}/{uuid4().hex}{source.suffix}"
    bucket.put_object_from_file(key, str(source), progress_callback=progress_callback)
    return bucket.sign_url("GET", key, expires_seconds, slash_safe=True)
