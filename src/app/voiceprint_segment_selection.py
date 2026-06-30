"""Shared transcript segment scoring for voiceprint sampling."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import SentenceSegment

MIN_SELECTION_SCORE = 0.30
MIN_RECOMMENDED_SCORE = 0.55
DEFAULT_CANDIDATE_FLOOR = 12


@dataclass(frozen=True, slots=True)
class ScoredVoiceprintSegment:
    """One voiceprint candidate segment with selection diagnostics."""

    segment: SentenceSegment
    score: float
    reason: str
    recommended: bool = False


def select_voiceprint_segments(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    sample_count: int,
    *,
    candidate_count: int | None = None,
) -> list[ScoredVoiceprintSegment]:
    """
    Select high-quality, time-spread candidate segments for voiceprint use.

    The returned list may contain more than ``sample_count`` candidates so callers
    can review or apply their own embedding-level filtering. Items marked
    ``recommended`` are the default sample_count picks.
    """
    target_count = candidate_count or max(sample_count, DEFAULT_CANDIDATE_FLOOR)
    scored = _scored_unique_segments(segments, all_segments)
    selected = _select_diverse_segments(scored, target_count, min_gap_ms=10_000)
    if len(selected) < target_count:
        selected = _select_diverse_segments(scored, target_count, min_gap_ms=2_000)
    if len(selected) < target_count:
        selected = _select_diverse_segments(scored, target_count, min_gap_ms=0)
    recommended_ids = _recommended_segment_ids(selected, sample_count)
    marked = [
        ScoredVoiceprintSegment(
            item.segment, item.score, item.reason, id(item.segment) in recommended_ids
        )
        for item in selected
    ]
    return sorted(marked, key=lambda item: item.segment.begin_time_ms)


def _recommended_segment_ids(
    segments: list[ScoredVoiceprintSegment], sample_count: int
) -> set[int]:
    """
    Pick default checked samples from a good-enough, time-spread pool.

    The score is a usability gate, not a target to maximize. Always taking the
    highest scores overfits toward one speaking style and often clusters around
    long dense monologues, so defaults should cover the speaker's timeline.
    """
    if sample_count <= 0 or not segments:
        return set()
    eligible = [item for item in segments if item.score >= MIN_RECOMMENDED_SCORE]
    if len(eligible) < sample_count:
        eligible = segments
    return {
        id(item.segment) for item in _spread_segments_by_time(eligible, sample_count)
    }


def _spread_segments_by_time(
    segments: list[ScoredVoiceprintSegment], sample_count: int
) -> list[ScoredVoiceprintSegment]:
    """Return up to sample_count segments spread across the speaker timeline."""
    ordered = sorted(segments, key=lambda item: item.segment.begin_time_ms)
    if len(ordered) <= sample_count:
        return ordered
    if sample_count == 1:
        return [ordered[len(ordered) // 2]]
    last_index = len(ordered) - 1
    indices = [
        round(index * last_index / (sample_count - 1)) for index in range(sample_count)
    ]
    return [ordered[index] for index in indices]


def _scored_unique_segments(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
) -> list[ScoredVoiceprintSegment]:
    """Return unique candidate segments sorted by descending quality score."""
    seen: set[tuple[int, int, str]] = set()
    scored: list[ScoredVoiceprintSegment] = []
    for segment in segments:
        key = (
            segment.begin_time_ms,
            segment.end_time_ms,
            _normalized_text(segment.text),
        )
        if key in seen:
            continue
        seen.add(key)
        candidate = score_voiceprint_segment(segment, all_segments)
        if candidate.score >= MIN_SELECTION_SCORE:
            scored.append(candidate)
    return sorted(scored, key=lambda item: (-item.score, item.segment.begin_time_ms))


def _select_diverse_segments(
    candidates: list[ScoredVoiceprintSegment],
    sample_count: int,
    *,
    min_gap_ms: int,
) -> list[ScoredVoiceprintSegment]:
    """Select high-scoring segments while avoiding nearby duplicates."""
    selected: list[ScoredVoiceprintSegment] = []
    for candidate in candidates:
        if len(selected) >= sample_count:
            break
        if any(
            _segments_too_close(candidate.segment, item.segment, min_gap_ms)
            for item in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _segments_too_close(
    left: SentenceSegment, right: SentenceSegment, min_gap_ms: int
) -> bool:
    """Return whether two candidate segments are too close for diverse sampling."""
    if (
        left.begin_time_ms <= right.end_time_ms
        and right.begin_time_ms <= left.end_time_ms
    ):
        return True
    return abs(left.begin_time_ms - right.begin_time_ms) < min_gap_ms


def score_voiceprint_segment(
    segment: SentenceSegment, all_segments: list[SentenceSegment]
) -> ScoredVoiceprintSegment:
    """Score one transcript segment for voiceprint capture or matching."""
    duration_ms = _segment_duration_ms(segment)
    text = segment.text.strip()
    if _is_low_information_text(_normalized_text(text)):
        return ScoredVoiceprintSegment(segment, 0.0, "low-information")
    duration_score = _duration_score(duration_ms)
    text_score = _text_score(text)
    boundary_score = _boundary_score(segment, all_segments)
    score = round(0.45 * duration_score + 0.35 * text_score + 0.20 * boundary_score, 3)
    reason = f"duration={duration_score:.2f}, text={text_score:.2f}, boundary={boundary_score:.2f}"
    return ScoredVoiceprintSegment(segment, score, reason)


def _duration_score(duration_ms: int) -> float:
    """Score duration, preferring 6-18 second speech samples."""
    seconds = duration_ms / 1000
    if seconds < 3:
        return max(0.0, seconds / 3 * 0.45)
    if seconds <= 18:
        return 1.0
    if seconds <= 30:
        return 0.8
    return 0.55


def _text_score(text: str) -> float:
    """Score text content, penalizing filler-only fragments."""
    normalized = _normalized_text(text)
    if not normalized:
        return 0.0
    if _is_low_information_text(normalized):
        return 0.1
    length_score = min(1.0, len(normalized) / 24)
    unique_score = min(1.0, len(set(normalized)) / 10)
    return round(0.65 * length_score + 0.35 * unique_score, 3)


def _boundary_score(
    segment: SentenceSegment, all_segments: list[SentenceSegment]
) -> float:
    """Score speaker-boundary safety using neighboring transcript segments."""
    sorted_segments = sorted(
        all_segments, key=lambda item: (item.begin_time_ms, item.end_time_ms)
    )
    index = next((i for i, item in enumerate(sorted_segments) if item is segment), -1)
    if index < 0:
        return 0.7
    previous_score = _neighbor_score(
        segment, sorted_segments[index - 1] if index else None, before=True
    )
    next_score = _neighbor_score(
        segment,
        sorted_segments[index + 1] if index + 1 < len(sorted_segments) else None,
        before=False,
    )
    return min(previous_score, next_score)


def _neighbor_score(
    segment: SentenceSegment, neighbor: SentenceSegment | None, *, before: bool
) -> float:
    """Score one neighboring segment for possible speaker overlap."""
    if neighbor is None or neighbor.speaker_id == segment.speaker_id:
        return 1.0
    gap = (
        segment.begin_time_ms - neighbor.end_time_ms
        if before
        else neighbor.begin_time_ms - segment.end_time_ms
    )
    if gap < 0:
        return 0.2
    if gap < 500:
        return 0.45
    if gap < 1200:
        return 0.75
    return 1.0


def _normalized_text(text: str) -> str:
    """Return compact text for quality heuristics."""
    return re.sub(r"\s+", "", text.strip().casefold())


def _is_low_information_text(text: str) -> bool:
    """Return whether text is mostly filler/backchannel content."""
    filler_pattern = (
        r"(嗯+|啊+|呃+|哦+|对+|是+|好+|可以|就是|然后|那个|这个|ok|嗯哼|哈哈)+"
    )
    return re.fullmatch(filler_pattern, text) is not None


def _segment_duration_ms(segment: SentenceSegment) -> int:
    """Return a non-negative segment duration in milliseconds."""
    return max(0, segment.end_time_ms - segment.begin_time_ms)
