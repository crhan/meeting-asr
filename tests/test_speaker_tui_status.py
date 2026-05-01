"""Tests for speaker review status rendering."""

from __future__ import annotations

from app.models import SentenceSegment
from app.speaker_tui import ReviewSpeaker, SpeakerMatchCandidate
from app.speaker_tui_status import (
    SpeakerReviewOverview,
    VoiceprintReviewProgress,
    match_badge,
    render_overview_pane,
    speaker_status,
)


def test_speaker_status_distinguishes_conflict_mismatch_and_match() -> None:
    """Status helpers should not collapse accepted conflicts and review mismatches."""
    conflict = ReviewSpeaker(
        0,
        "Speaker A",
        _segments(),
        "人工姓名",
        SpeakerMatchCandidate("自动姓名", 0.91, True),
    )
    mismatch = ReviewSpeaker(
        1,
        "Speaker B",
        _segments(),
        "人工姓名",
        SpeakerMatchCandidate("自动姓名", 0.72, False),
    )
    matched = ReviewSpeaker(
        2,
        "Speaker C",
        _segments(),
        "自动姓名",
        SpeakerMatchCandidate("自动姓名", 0.95, True),
    )

    assert speaker_status(conflict) == "conflict"
    assert "CONFLICT" in match_badge(conflict)
    assert speaker_status(mismatch) == "mismatch"
    assert "mismatch" in match_badge(mismatch)
    assert speaker_status(matched) == "matched"
    assert "accepted" in match_badge(matched)


def test_overview_next_action_requires_match_before_review() -> None:
    """Missing match output should be the first next action."""
    speaker = ReviewSpeaker(0, "Speaker A", _segments(), "Speaker A", None)
    overview = _overview(match_file_exists=False, saved_names={})

    rendered = render_overview_pane([speaker], overview, speaker)

    assert "Match=[yellow]pending" in rendered
    assert "run `meeting-asr project speakers match`" in rendered


def test_overview_next_action_reports_embedding_config_problem() -> None:
    """Captured samples with unknown embedding state should produce an embed fix action."""
    speaker = ReviewSpeaker(
        0,
        "Speaker A",
        _segments(),
        "欧丁",
        SpeakerMatchCandidate("欧丁", 0.92, True),
    )
    overview = _overview(
        match_file_exists=True,
        saved_names={0: "欧丁"},
        voiceprint=VoiceprintReviewProgress(
            captured_names_by_speaker={0: frozenset({"欧丁"})},
            captured_sample_ids=frozenset({1}),
            embed_model=None,
            embedded_sample_ids=None,
            embed_error="bad config",
        ),
    )

    rendered = render_overview_pane([speaker], overview, speaker)

    assert "Embed=[yellow]unknown" in rendered
    assert "fix voiceprint embedding config" in rendered


def _overview(
    *,
    match_file_exists: bool,
    saved_names: dict[int, str],
    voiceprint: VoiceprintReviewProgress | None = None,
) -> SpeakerReviewOverview:
    """Build an overview fixture."""
    return SpeakerReviewOverview(
        project_id="project-1",
        title="Demo",
        project_status="named",
        source_name="meeting.mp4",
        duration_ms=1000,
        match_file_exists=match_file_exists,
        saved_names_by_speaker=saved_names,
        voiceprint=voiceprint
        or VoiceprintReviewProgress(
            captured_names_by_speaker={},
            captured_sample_ids=frozenset(),
            embed_model="test-model",
            embedded_sample_ids=frozenset(),
        ),
    )


def _segments() -> list[SentenceSegment]:
    """Build a minimal segment list for a speaker fixture."""
    return [
        SentenceSegment(
            begin_time_ms=0,
            end_time_ms=1000,
            text="测试",
            speaker_id=0,
            sentence_id=1,
        )
    ]
