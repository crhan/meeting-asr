"""Tests for local speaker preview player commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import speaker_review
from app.speaker_review import build_audio_preview_command, build_preview_command


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
    monkeypatch.setattr(
        speaker_review, "_ffplay_supports_subtitles_filter", lambda: False
    )

    command = build_preview_command(
        video=video, subtitle=subtitle, start_seconds=12.3456
    )

    assert video.with_suffix(".srt").read_text(encoding="utf-8") == subtitle.read_text(
        encoding="utf-8"
    )
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
    monkeypatch.setattr(
        speaker_review.shutil,
        "which",
        lambda name: "/usr/local/bin/mpv" if name == "mpv" else None,
    )

    command = build_preview_command(video=video, subtitle=subtitle, start_seconds=0.0)

    assert "--resume-playback=no" in command
    assert f"--sub-file={subtitle.resolve()}" in command
    assert "--sid=1" in command


def test_mpv_audio_preview_disables_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audio preview should use the source media without opening video."""
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"video")
    monkeypatch.setattr(
        speaker_review.shutil,
        "which",
        lambda name: "/usr/local/bin/mpv" if name == "mpv" else None,
    )

    command = build_audio_preview_command(
        media=media, start_seconds=9.8765, duration_seconds=4.321
    )

    assert command == [
        "/usr/local/bin/mpv",
        "--really-quiet",
        "--resume-playback=no",
        "--vid=no",
        "--force-window=no",
        "--start=9.877",
        "--length=4.321",
        str(media.resolve()),
    ]


def test_ffplay_audio_preview_is_quiet_and_limited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ffplay fallback should avoid banner/stats output and stop after the clip."""
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"video")
    monkeypatch.setattr(
        speaker_review.shutil,
        "which",
        lambda name: "ffplay" if name == "ffplay" else None,
    )

    command = build_audio_preview_command(
        media=media, start_seconds=10.0, duration_seconds=5.0
    )

    assert command == [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-nodisp",
        "-autoexit",
        "-ss",
        "10.000",
        "-t",
        "5.000",
        "-i",
        str(media.resolve()),
    ]


def test_preview_rejects_missing_subtitle(tmp_path: Path) -> None:
    """Preview should fail before opening a player when subtitles are missing."""
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="Subtitle file does not exist"):
        build_preview_command(
            video=video, subtitle=tmp_path / "missing.srt", start_seconds=0.0
        )
