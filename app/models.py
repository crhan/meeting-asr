"""Dataclasses shared by transcription post-processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SentenceSegment:
    """One normalized ASR sentence segment."""

    begin_time_ms: int
    end_time_ms: int
    text: str
    speaker_id: int | None
    sentence_id: int | None = None

    def to_dict(self) -> dict:
        """Return a JSON-ready dictionary."""
        return asdict(self)


@dataclass(slots=True)
class TranscriptResult:
    """Normalized transcript data."""

    full_text: str
    sentences: list[SentenceSegment]
    detected_speakers: list[int]
