"""Tests for project OSS upload workflow progress."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import project_manager
from app.config import Settings
from app.core.oss_metrics import OssUploadEstimate
from app.project_manager import (
    _resolve_project_file_url,
    create_project,
    load_manifest,
    ProjectTranscribeOptions,
    project_paths,
    save_manifest,
)


def test_project_oss_upload_reports_eta_and_records_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project upload should render byte progress and record one runtime sample."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "meeting.wav"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    estimate = OssUploadEstimate(
        estimated_seconds=8.0,
        sample_count=3,
        confidence="medium",
        bytes_per_second=1.25,
    )
    records: list[dict[str, object]] = []

    def fake_upload_file_to_oss(*args, **kwargs) -> str:
        """
        Simulate a successful OSS upload.

        Args:
            *args: Positional upload arguments.
            **kwargs: Keyword upload arguments, including progress_callback.

        Returns:
            Signed URL fixture.
        """
        progress_callback = kwargs["progress_callback"]
        progress_callback(5, 10)
        progress_callback(10, 10)
        return "https://signed.example.com/audio.wav"

    monkeypatch.setattr(
        project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: estimate
    )
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(
        project_manager,
        "record_oss_upload",
        lambda settings, **kwargs: records.append(kwargs),
    )
    events = []

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        events.append,
    )

    assert file_url == "https://signed.example.com/audio.wav"
    assert source == "oss_signed_url"
    assert events[0].description == "Uploading audio to OSS | ETA ~8s | medium n=3"
    assert [event.completed for event in events if event.total == 10] == [0, 5, 10, 10]
    assert records[0]["project_id"] == manifest.project_id
    assert (
        records[0]["object_key"]
        == f"meeting-asr/projects/{manifest.project_id}/meeting.wav"
    )
    assert records[0]["size_bytes"] == 10
    assert records[0]["status"] == "succeeded"
    assert load_manifest(project_dir).oss["object_key"] == records[0]["object_key"]


def test_project_oss_upload_reuses_existing_project_object_without_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing matching project OSS object should only be signed again."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    object_key = f"meeting-asr/projects/{manifest.project_id}/audio.flac"
    manifest.oss = {
        "mode": "oss_signed_url",
        "bucket": "meeting-bucket",
        "object_key": object_key,
        "signed_url_expires_at": "2026-01-01T00:00:00+00:00",
    }
    save_manifest(project_dir, manifest)
    presigned: list[dict[str, object]] = []

    def fake_presign_oss_object(*args, **kwargs) -> str:
        presigned.append({"args": args, "kwargs": kwargs})
        return "https://signed.example.com/reused.flac"

    def fail_upload_file_to_oss(*args, **kwargs) -> str:
        raise AssertionError(
            "upload_file_to_oss should not be called for a reusable object"
        )

    monkeypatch.setattr(
        project_manager, "oss_object_exists", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(project_manager, "presign_oss_object", fake_presign_oss_object)
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fail_upload_file_to_oss)
    events = []

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        events.append,
    )

    saved_oss = load_manifest(project_dir).oss
    assert file_url == "https://signed.example.com/reused.flac"
    assert source == "oss_signed_url"
    assert presigned[0]["args"] == (object_key,)
    assert saved_oss["object_key"] == object_key
    assert saved_oss["signed_url_expires_at"] != "2026-01-01T00:00:00+00:00"
    assert events[-1].description == "Reused existing OSS object"


def test_project_oss_upload_uploads_when_existing_project_object_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing remote object should fall back to the normal upload path."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    object_key = f"meeting-asr/projects/{manifest.project_id}/audio.flac"
    manifest.oss = {
        "mode": "oss_signed_url",
        "bucket": "meeting-bucket",
        "object_key": object_key,
    }
    save_manifest(project_dir, manifest)
    uploaded: list[str] = []

    def fake_upload_file_to_oss(*args, **kwargs) -> str:
        uploaded.append(kwargs["object_name"])
        return "https://signed.example.com/reuploaded.flac"

    def fail_presign_oss_object(*args, **kwargs) -> str:
        raise AssertionError(
            "presign_oss_object should not be called for a missing object"
        )

    monkeypatch.setattr(
        project_manager, "oss_object_exists", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(project_manager, "presign_oss_object", fail_presign_oss_object)
    monkeypatch.setattr(
        project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: None
    )
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(
        project_manager, "record_oss_upload", lambda settings, **kwargs: None
    )
    events = []

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        events.append,
    )

    assert file_url == "https://signed.example.com/reuploaded.flac"
    assert source == "oss_signed_url"
    assert uploaded == [object_key]
    assert events[0].description.startswith(
        "Existing OSS object presign failed; uploading audio instead"
    )


def test_project_oss_upload_uploads_when_manifest_has_no_object_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing manifest object key should keep the first-upload behavior."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    manifest.oss = {"mode": "oss_signed_url", "bucket": "meeting-bucket"}
    save_manifest(project_dir, manifest)
    uploaded: list[dict[str, object]] = []

    def fake_upload_file_to_oss(*args, **kwargs) -> str:
        uploaded.append({"args": args, "kwargs": kwargs})
        return "https://signed.example.com/uploaded.flac"

    def fail_presign_oss_object(*args, **kwargs) -> str:
        raise AssertionError(
            "presign_oss_object should not be called without object_key"
        )

    monkeypatch.setattr(project_manager, "presign_oss_object", fail_presign_oss_object)
    monkeypatch.setattr(
        project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: None
    )
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(
        project_manager, "record_oss_upload", lambda settings, **kwargs: None
    )

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        None,
    )

    assert file_url == "https://signed.example.com/uploaded.flac"
    assert source == "oss_signed_url"
    assert (
        uploaded[0]["kwargs"]["object_name"]
        == f"meeting-asr/projects/{manifest.project_id}/audio.flac"
    )


def test_project_oss_upload_uploads_when_manifest_object_key_mismatches_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mismatched object key should not be reused for current project audio."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    manifest.oss = {
        "mode": "oss_signed_url",
        "bucket": "meeting-bucket",
        "object_key": f"meeting-asr/projects/{manifest.project_id}/other.flac",
    }
    save_manifest(project_dir, manifest)
    uploaded: list[str] = []

    def fake_upload_file_to_oss(*args, **kwargs) -> str:
        uploaded.append(kwargs["object_name"])
        return "https://signed.example.com/refreshed.flac"

    def fail_presign_oss_object(*args, **kwargs) -> str:
        raise AssertionError(
            "presign_oss_object should not be called for mismatched object_key"
        )

    monkeypatch.setattr(project_manager, "presign_oss_object", fail_presign_oss_object)
    monkeypatch.setattr(
        project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: None
    )
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(
        project_manager, "record_oss_upload", lambda settings, **kwargs: None
    )

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        None,
    )

    expected_key = f"meeting-asr/projects/{manifest.project_id}/audio.flac"
    assert file_url == "https://signed.example.com/refreshed.flac"
    assert source == "oss_signed_url"
    assert uploaded == [expected_key]
    assert load_manifest(project_dir).oss["object_key"] == expected_key


def test_project_oss_upload_falls_back_to_upload_when_presign_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Presign failure should be visible and then use the original upload path."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"1234567890")
    manifest = load_manifest(project_dir)
    object_key = f"meeting-asr/projects/{manifest.project_id}/audio.flac"
    manifest.oss = {
        "mode": "oss_signed_url",
        "bucket": "meeting-bucket",
        "object_key": object_key,
    }
    save_manifest(project_dir, manifest)
    uploaded: list[str] = []

    def fail_presign_oss_object(*args, **kwargs) -> str:
        raise RuntimeError("signature service rejected request")

    def fake_upload_file_to_oss(*args, **kwargs) -> str:
        uploaded.append(kwargs["object_name"])
        return "https://signed.example.com/fallback.flac"

    monkeypatch.setattr(
        project_manager, "oss_object_exists", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(project_manager, "presign_oss_object", fail_presign_oss_object)
    monkeypatch.setattr(
        project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: None
    )
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(
        project_manager, "record_oss_upload", lambda settings, **kwargs: None
    )
    events = []

    file_url, source = _resolve_project_file_url(
        project_paths(project_dir),
        manifest,
        audio_path,
        _transcribe_options(),
        True,
        _settings(),
        events.append,
    )

    assert file_url == "https://signed.example.com/fallback.flac"
    assert source == "oss_signed_url"
    assert uploaded == [object_key]
    assert events[0].description.startswith(
        "Existing OSS object presign failed; uploading audio instead"
    )


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _transcribe_options() -> ProjectTranscribeOptions:
    """Build minimal transcription options for private helper tests."""
    return ProjectTranscribeOptions(
        speaker_count=None,
        language=None,
        model="fun-asr",
        oss_upload=True,
        file_url=None,
        generate_srt=True,
        timestamp_alignment=True,
        disfluency_removal=False,
        audio_format="wav",
    )


def _settings() -> Settings:
    """Build runtime settings for private helper tests."""
    return Settings(
        dashscope_api_key="dashscope-key",
        dashscope_base_url="https://dashscope.example.com",
        oss_access_key_id="oss-id",
        oss_access_key_secret="oss-secret",
        oss_bucket_name="meeting-bucket",
        oss_region="cn-test",
        oss_endpoint="https://oss.example.com",
    )
