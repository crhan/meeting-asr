"""Shared data models for the speaker review TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.models import SentenceSegment
from app.presentation.tui.speaker_correction import SentenceCorrectionEdit
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_people import KnownPerson
from app.presentation.tui.speaker_status import SpeakerReviewOverview


SegmentScoreKey = tuple[int | None, int, int]


@dataclass(frozen=True, slots=True)
class SpeakerClusterSampleScore:
    """One sample score from speaker cluster diagnostics."""

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    score: float | None
    status: str
    text: str = ""
    nearest_speaker_id: int | None = None
    nearest_score: float | None = None
    margin_score: float | None = None

    @property
    def key(self) -> SegmentScoreKey:
        """Return the transcript segment key this score belongs to."""
        return (self.sentence_id, self.begin_time_ms, self.end_time_ms)


@dataclass(frozen=True, slots=True)
class SpeakerSampleIdentityScore:
    """One sample-level voiceprint identity score."""

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    assigned_score: float | None
    best_name: str | None
    best_score: float | None
    best_other_name: str | None
    best_other_score: float | None
    margin_score: float | None
    status: str
    text: str = ""

    @property
    def key(self) -> SegmentScoreKey:
        """Return the transcript segment key this score belongs to."""
        return (self.sentence_id, self.begin_time_ms, self.end_time_ms)


@dataclass(frozen=True, slots=True)
class SpeakerClusterDiagnostic:
    """Cluster quality diagnostics for one detected speaker."""

    speaker_id: int
    status: str
    centroid_mean: float | None
    centroid_min: float | None
    clip_count: int
    segment_count: int
    warning_clip_count: int
    critical_clip_count: int
    component_count: int
    component_sizes: tuple[int, ...]
    warnings: tuple[str, ...]
    samples: dict[SegmentScoreKey, SpeakerClusterSampleScore] = field(
        default_factory=dict
    )


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
    cluster_diagnostics: dict[int, SpeakerClusterDiagnostic] = field(
        default_factory=dict
    )
    sample_identity_scores: dict[
        int, dict[SegmentScoreKey, SpeakerSampleIdentityScore]
    ] = field(default_factory=dict)


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
