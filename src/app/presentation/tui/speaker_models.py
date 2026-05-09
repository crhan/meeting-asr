"""Shared data models for the speaker review TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.models import SentenceSegment
from app.presentation.tui.speaker_correction import SentenceCorrectionEdit
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_people import KnownPerson
from app.presentation.tui.speaker_status import SpeakerReviewOverview


@dataclass(slots=True)
class ReviewSpeaker:
    """Mutable review state for one project speaker."""

    speaker_id: int
    label: str
    segments: list[SentenceSegment]
    current_name: str
    match: SpeakerMatchCandidate | None
    selected_sample_index: int = 0
    ignored: bool = False
    person_id: int | None = None
    person_public_id: str | None = None

    @property
    def segment_count(self) -> int:
        """Return the total number of transcript segments for this speaker."""
        return len(self.segments)


@dataclass(frozen=True, slots=True)
class SentenceReassignment:
    """One sentence whose speaker_id changed during review.

    The triple ``(sentence_id, begin_time_ms, end_time_ms)`` identifies the
    sentence inside the persisted ``sentences.json`` payload. The reassignment
    is applied on save to update the speaker_id in both the raw and corrected
    ASR sentence files.
    """

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    original_speaker_id: int | None
    new_speaker_id: int


@dataclass(frozen=True, slots=True)
class SpeakerReviewSession:
    """Inputs needed by the speaker review TUI."""

    project_dir: Path
    source_media: Path
    overview: SpeakerReviewOverview
    speakers: list[ReviewSpeaker]
    people_names: list[str]
    page_size: int | None = None
    allow_correction: bool = False
    people: tuple[KnownPerson, ...] = ()
    store_dir: Path | None = None
    projects_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class SpeakerReviewDecision:
    """Result returned by the TUI when it exits."""

    saved: bool
    mapping: dict[int, str]
    action: str = "save"
    person_mapping: dict[int, int] = field(default_factory=dict)
    person_public_mapping: dict[int, str] = field(default_factory=dict)
    ignored_speaker_ids: tuple[int, ...] = ()
    correction_edit: SentenceCorrectionEdit | None = None
    correction_edits: tuple[SentenceCorrectionEdit, ...] = ()
    sentence_reassignments: tuple[SentenceReassignment, ...] = ()
    project_dir: Path | None = field(default=None, compare=False)
