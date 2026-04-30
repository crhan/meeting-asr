"""Tests for SRT utilities."""

from __future__ import annotations

from app.models import SentenceSegment
from app.srt_utils import build_srt, ms_to_srt_timestamp


def test_ms_to_srt_timestamp() -> None:
    """Milliseconds should format as SRT timestamp."""
    assert ms_to_srt_timestamp(1234) == "00:00:01,234"


def test_build_srt_includes_speaker_label() -> None:
    """SRT output should include speaker labels by default."""
    text = build_srt([SentenceSegment(0, 1000, "你好", 0, 1)])

    assert "Speaker A: 你好" in text
