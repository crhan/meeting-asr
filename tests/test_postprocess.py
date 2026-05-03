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


def test_parse_transcription_result_filters_filler_only_speaker() -> None:
    """Pure filler speaker tracks should be removed before users review speakers."""
    raw = {
        "sentences": [
            {"begin_time": 1, "end_time": 2, "text": "嗯", "speaker_id": 0, "id": 1},
            {"begin_time": 3, "end_time": 4, "text": "呃", "speaker_id": 0, "id": 2},
            {"begin_time": 5, "end_time": 6, "text": "我们开始看 iSee 系统。", "speaker_id": 1, "id": 3},
        ]
    }

    result = parse_transcription_result(raw)

    assert result.detected_speakers == [1]
    assert [sentence.text for sentence in result.sentences] == ["我们开始看 iSee 系统。"]
    assert "嗯" not in result.full_text


def test_parse_transcription_result_filters_low_information_speaker_track() -> None:
    """Backchannel-only speaker tracks should not become review speakers."""
    raw = {
        "sentences": [
            {"begin_time": 1, "end_time": 2, "text": "嗯。", "speaker_id": 0, "id": 1},
            {"begin_time": 3, "end_time": 4, "text": "对对对。", "speaker_id": 0, "id": 2},
            {"begin_time": 5, "end_time": 6, "text": "啊，对。", "speaker_id": 0, "id": 3},
            {"begin_time": 7, "end_time": 8, "text": "这样吧。", "speaker_id": 0, "id": 4},
            {"begin_time": 9, "end_time": 10, "text": "就是听听听听他。", "speaker_id": 0, "id": 5},
            {"begin_time": 11, "end_time": 12, "text": "嗯嗯，OK。", "speaker_id": 0, "id": 6},
            {"begin_time": 13, "end_time": 14, "text": "就是可以再理一下了。", "speaker_id": 0, "id": 7},
            {"begin_time": 15, "end_time": 16, "text": "我们开始看供应商维修闭环。", "speaker_id": 1, "id": 8},
        ]
    }

    result = parse_transcription_result(raw)

    assert result.detected_speakers == [1]
    assert [sentence.text for sentence in result.sentences] == ["我们开始看供应商维修闭环。"]


def test_parse_transcription_result_keeps_short_meaningful_speaker_track() -> None:
    """Short answers with actual content should remain review speakers."""
    raw = {
        "sentences": [
            {"begin_time": 1, "end_time": 2, "text": "接口已经下线了。", "speaker_id": 0, "id": 1},
            {"begin_time": 3, "end_time": 4, "text": "明天回滚配置。", "speaker_id": 0, "id": 2},
            {"begin_time": 5, "end_time": 6, "text": "我负责跟进。", "speaker_id": 0, "id": 3},
            {"begin_time": 7, "end_time": 8, "text": "风险还在。", "speaker_id": 0, "id": 4},
            {"begin_time": 9, "end_time": 10, "text": "先暂停发布。", "speaker_id": 0, "id": 5},
            {"begin_time": 11, "end_time": 12, "text": "我们继续看下一个问题。", "speaker_id": 1, "id": 6},
        ]
    }

    result = parse_transcription_result(raw)

    assert result.detected_speakers == [0, 1]


def test_parse_transcription_result_keeps_empty_text_after_filtering_all_speakers() -> None:
    """Filtering all parsed sentences must not fall back to raw filler text."""
    raw = {
        "text": "嗯呃",
        "sentences": [
            {"begin_time": 1, "end_time": 2, "text": "嗯", "speaker_id": 0, "id": 1},
            {"begin_time": 3, "end_time": 4, "text": "呃", "speaker_id": 0, "id": 2},
        ],
    }

    result = parse_transcription_result(raw)

    assert result.full_text == ""
    assert result.sentences == []
    assert result.detected_speakers == []


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
