"""Tests for local speaker preview player commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import speaker_review
from app.speaker_review import build_preview_command


def test_iina_preview_uses_raw_mpv_options_after_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """IINA should get an auto-loadable sidecar subtitle."""
    video = tmp_path / "meeting.mp4"
    subtitle = tmp_path / "subtitle.srt"
    video.write_bytes(b"video")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    monkeypatch.setattr(speaker_review.shutil, "which", lambda name: None)
    monkeypatch.setattr(speaker_review, "_find_iina_cli", lambda: "/usr/local/bin/iina")
    monkeypatch.setattr(speaker_review, "_ffplay_supports_subtitles_filter", lambda: False)

    command = build_preview_command(video=video, subtitle=subtitle, start_seconds=12.3456)

    assert video.with_suffix(".srt").read_text(encoding="utf-8") == subtitle.read_text(encoding="utf-8")
    assert command == [
        "/usr/local/bin/iina",
        "--no-stdin",
        str(video.resolve()),
        "--",
        "--resume-playback=no",
        "--sub-auto=fuzzy",
        "--sid=1",
        "--start=12.346",
    ]


def test_mpv_preview_disables_resume_and_selects_subtitle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """mpv should not let watch-later state override preview inputs."""
    video = tmp_path / "meeting.mp4"
    subtitle = tmp_path / "subtitle.srt"
    video.write_bytes(b"video")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    monkeypatch.setattr(speaker_review.shutil, "which", lambda name: "/usr/local/bin/mpv" if name == "mpv" else None)

    command = build_preview_command(video=video, subtitle=subtitle, start_seconds=0.0)

    assert "--resume-playback=no" in command
    assert f"--sub-file={subtitle.resolve()}" in command
    assert "--sid=1" in command


def test_preview_rejects_missing_subtitle(tmp_path: Path) -> None:
    """Preview should fail before opening a player when subtitles are missing."""
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="Subtitle file does not exist"):
        build_preview_command(video=video, subtitle=tmp_path / "missing.srt", start_seconds=0.0)
