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

    monkeypatch.setattr(project_manager, "estimate_oss_upload", lambda settings, *, size_bytes: estimate)
    monkeypatch.setattr(project_manager, "upload_file_to_oss", fake_upload_file_to_oss)
    monkeypatch.setattr(project_manager, "record_oss_upload", lambda settings, **kwargs: records.append(kwargs))
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
    assert records[0]["object_key"] == f"meeting-asr/projects/{manifest.project_id}/meeting.wav"
    assert records[0]["size_bytes"] == 10
    assert records[0]["status"] == "succeeded"
    assert load_manifest(project_dir).oss["object_key"] == records[0]["object_key"]


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
