"""Tests for transcript post-processing."""

from __future__ import annotations

from app.models import SentenceSegment, TranscriptResult
from app.postprocess import (
    merge_adjacent_sentences,
    parse_transcription_result,
    render_speaker_text,
    speaker_id_to_label,
)


def test_speaker_id_to_label() -> None:
    """Speaker labels should be deterministic."""
    assert speaker_id_to_label(0) == "Speaker A"
    assert speaker_id_to_label(26) == "Speaker 27"
    assert speaker_id_to_label(None) == "Speaker Unknown"


def test_parse_transcription_result_nested_sentences() -> None:
    """Parser should handle common nested transcript JSON."""
    raw = {"transcripts": [{"sentences": [{"begin_time": 1, "end_time": 2, "text": "hi", "speaker_id": 0}]}]}

    result = parse_transcription_result(raw)

    assert result.detected_speakers == [0]
    assert result.sentences[0].text == "hi"


def test_merge_adjacent_sentences_same_speaker() -> None:
    """Adjacent same-speaker sentences should merge for readable output."""
    sentences = [
        SentenceSegment(0, 1000, "大家好。", 0, 1),
        SentenceSegment(1500, 2000, "继续。", 0, 2),
        SentenceSegment(2500, 3000, "收到。", 1, 3),
    ]

    merged = merge_adjacent_sentences(sentences)

    assert len(merged) == 2
    assert merged[0].text == "大家好。继续。"


def test_render_speaker_text() -> None:
    """Speaker transcript should include timestamps and labels."""
    result = TranscriptResult("", [SentenceSegment(0, 1200, "你好", 0, 1)], [0])

    assert "Speaker A: 你好" in render_speaker_text(result)
