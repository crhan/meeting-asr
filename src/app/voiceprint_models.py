"""Shared voiceprint registry data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceprintSampleRow:
    """Stored voiceprint sample row."""

    sample_id: int
    public_id: str
    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    project_id: str
    project_speaker_id: int
    clip_path: Path
    clip_rel_path: str
    clip_sha256: str
    source_begin_time_ms: int
    source_end_time_ms: int
    transcript_text: str
    sample_status: str = "active"
    # The stored clip's own span. Capture truncates a long sentence to the
    # capture limit, so this is shorter than the source interval whenever the
    # sentence ran long. None when the query did not select these columns.
    clip_begin_time_ms: int | None = None
    clip_end_time_ms: int | None = None

    @property
    def embedded_duration_ms(self) -> int:
        """Return how much audio this sample actually contributes.

        The source interval measures the sentence; the embedding only ever saw
        the clip. Counting the sentence would credit a 30s utterance with 30s
        of reference audio after 12s was stored, which is enough to silence a
        "too little audio" warning that is still true. Falls back to the source
        interval when the clip span is unknown, since under-reporting to zero
        would make every affected person look starved.
        """
        if self.clip_begin_time_ms is not None and self.clip_end_time_ms is not None:
            span = self.clip_end_time_ms - self.clip_begin_time_ms
            if span > 0:
                return span
        return max(0, self.source_end_time_ms - self.source_begin_time_ms)


@dataclass(frozen=True, slots=True)
class VoiceprintSpeakerRow:
    """Stored speaker summary row."""

    speaker_id: int
    public_id: str
    name: str
    sample_count: int
    project_count: int
    embedded_sample_count: int
    embedding_model_count: int
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class VoiceprintEmbeddingRow:
    """Stored voiceprint embedding row."""

    sample_id: int
    sample_public_id: str
    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    clip_path: Path
    project_id: str
    source_begin_time_ms: int
    source_end_time_ms: int
    transcript_text: str
    model: str
    vector: list[float]
    sample_status: str = "active"


@dataclass(frozen=True, slots=True)
class DeletedVoiceprintSample:
    """Deleted voiceprint sample result."""

    sample_id: int
    public_id: str
    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    clip_path: Path
    clip_deleted: bool


@dataclass(frozen=True, slots=True)
class StoredVoiceprintSample:
    """Voiceprint sample passed to SQLite storage."""

    speaker_name: str
    project_id: str
    project_path: Path
    project_speaker_id: int
    source_path: Path
    clip_path: Path
    clip_rel_path: str
    source_begin_time_ms: int
    source_end_time_ms: int
    clip_begin_time_ms: int
    clip_end_time_ms: int
    transcript_text: str
    person_id: int | None = None
    sample_status: str = "active"
