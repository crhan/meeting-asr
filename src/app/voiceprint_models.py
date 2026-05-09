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
