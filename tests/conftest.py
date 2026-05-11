"""Shared pytest configuration for meeting-asr tests."""

from __future__ import annotations

import shutil

import pytest


_AUDIO_PLAYER_BINARIES = ("mpv", "ffplay")
_FFMPEG_BINARY = "ffmpeg"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used to gate tests on missing system tools."""
    config.addinivalue_line(
        "markers",
        "requires_audio_player: skip when neither mpv nor ffplay is available on PATH",
    )
    config.addinivalue_line(
        "markers",
        "requires_ffmpeg: skip when ffmpeg is unavailable on PATH",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Apply skip markers to tests that depend on absent local binaries."""
    no_audio_player = not any(shutil.which(binary) for binary in _AUDIO_PLAYER_BINARIES)
    no_ffmpeg = shutil.which(_FFMPEG_BINARY) is None
    skip_audio_player = pytest.mark.skip(reason="requires mpv or ffplay on PATH")
    skip_ffmpeg = pytest.mark.skip(reason="requires ffmpeg on PATH")
    for item in items:
        if no_audio_player and "requires_audio_player" in item.keywords:
            item.add_marker(skip_audio_player)
        if no_ffmpeg and "requires_ffmpeg" in item.keywords:
            item.add_marker(skip_ffmpeg)
