"""Tests for OSS upload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.uploader import upload_file_to_oss


def test_upload_file_to_oss_forwards_progress_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The uploader should expose oss2 byte progress to callers."""
    source = tmp_path / "audio.wav"
    source.write_bytes(b"12345678")
    bucket = FakeBucket()
    monkeypatch.setattr("app.uploader.build_oss_bucket", lambda settings: bucket)
    events: list[tuple[int, int]] = []

    url = upload_file_to_oss(
        source,
        object_name="meeting-asr/projects/p-demo/audio.wav",
        settings=_settings(),
        expires_seconds=600,
        progress_callback=lambda consumed, total: events.append((consumed, total)),
    )

    assert (
        url
        == "https://signed.example.com/meeting-asr/projects/p-demo/audio.wav?expires=600"
    )
    assert bucket.uploaded == [
        ("meeting-asr/projects/p-demo/audio.wav", str(source.resolve()))
    ]
    assert events == [(4, 8), (8, 8)]
    # 兜底：上传时自动补上默认的 auto-delete lifecycle 规则。
    assert len(bucket.lifecycle_puts) == 1
    (rule,) = bucket.lifecycle_puts[0].rules
    assert rule.id == "meeting-asr-auto-delete"


def test_upload_survives_lifecycle_permission_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A credential without lifecycle permissions must not block the upload."""
    source = tmp_path / "audio.wav"
    source.write_bytes(b"12345678")
    bucket = _ForbiddenLifecycleBucket()
    monkeypatch.setattr("app.uploader.build_oss_bucket", lambda settings: bucket)

    url = upload_file_to_oss(
        source,
        object_name="meeting-asr/projects/p-demo/audio.wav",
        settings=_settings(),
        expires_seconds=600,
    )

    assert url.startswith("https://signed.example.com/")
    assert bucket.uploaded == [
        ("meeting-asr/projects/p-demo/audio.wav", str(source.resolve()))
    ]


class FakeBucket:
    """Small fake for the oss2 Bucket methods used by the uploader."""

    def __init__(self) -> None:
        """Create an empty fake bucket."""
        self.uploaded: list[tuple[str, str]] = []
        self.lifecycle_puts: list[object] = []

    def get_bucket_lifecycle(self):
        """Simulate a bucket that has no lifecycle configured yet."""
        from app.uploader import import_oss2

        raise import_oss2().exceptions.NoSuchLifecycle(404, {}, b"", {})

    def put_bucket_lifecycle(self, lifecycle) -> None:
        """Record the lifecycle configuration written by the fallback."""
        self.lifecycle_puts.append(lifecycle)

    def put_object_from_file(
        self, key: str, filename: str, progress_callback=None
    ) -> None:
        """
        Record the upload and simulate oss2 byte progress.

        Args:
            key: OSS object key.
            filename: Local file path.
            progress_callback: Optional oss2 callback.

        Returns:
            None.
        """
        self.uploaded.append((key, filename))
        if progress_callback is not None:
            progress_callback(4, 8)
            progress_callback(8, 8)

    def sign_url(self, method: str, key: str, expires: int, slash_safe: bool) -> str:
        """
        Return a deterministic signed URL.

        Args:
            method: HTTP method.
            key: OSS object key.
            expires: URL lifetime.
            slash_safe: Whether slashes stay unescaped.

        Returns:
            Signed URL fixture.
        """
        assert method == "GET"
        assert slash_safe is True
        return f"https://signed.example.com/{key}?expires={expires}"


class _ForbiddenLifecycleBucket(FakeBucket):
    """Fake bucket whose credential lacks lifecycle permissions."""

    def get_bucket_lifecycle(self):
        """Simulate an AccessDenied server error."""
        from app.uploader import import_oss2

        raise import_oss2().exceptions.ServerError(403, {}, b"", {})


def _settings() -> Settings:
    """Build runtime settings for uploader tests."""
    return Settings(
        dashscope_api_key="dashscope-key",
        dashscope_base_url="https://dashscope.example.com",
        oss_access_key_id="oss-id",
        oss_access_key_secret="oss-secret",
        oss_bucket_name="meeting-bucket",
        oss_region="cn-test",
        oss_endpoint="https://oss.example.com",
    )
