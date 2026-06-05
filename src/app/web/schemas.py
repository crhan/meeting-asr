"""Pydantic response/request models for the web API.

These mirror the internal frozen dataclasses (``ProjectListItem``,
``ProjectWorkflowSummary``, ...) at the HTTP boundary. Keeping a dedicated schema layer
means internal model changes do not silently reshape the public API, and FastAPI gets
the OpenAPI/validation it expects.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.project_models import ProjectListItem
from app.core.project_workflow import ProjectWorkflowSummary


class WorkflowState(BaseModel):
    """User-facing workflow state derived from on-disk artifacts."""

    state: str
    state_key: str
    next_action: str
    next_command_short: str
    outputs: list[str]
    missing: list[str]

    @classmethod
    def from_summary(cls, summary: ProjectWorkflowSummary) -> "WorkflowState":
        """Build from a :class:`ProjectWorkflowSummary`."""
        return cls(
            state=summary.state,
            state_key=summary.state_key,
            next_action=summary.next_action,
            next_command_short=summary.next_command_short,
            outputs=list(summary.outputs),
            missing=list(summary.missing),
        )


class ProjectSummary(BaseModel):
    """One project row for the projects list."""

    project_id: str
    title: str
    status: str
    meeting_time: str | None
    created_at: str
    updated_at: str
    meeting_keywords: list[str]
    path: str
    workflow: WorkflowState | None = None

    @classmethod
    def from_item(
        cls,
        item: ProjectListItem,
        workflow: ProjectWorkflowSummary | None = None,
    ) -> "ProjectSummary":
        """Build from a :class:`ProjectListItem` and optional workflow summary."""
        return cls(
            project_id=item.project_id,
            title=item.title,
            status=item.status,
            meeting_time=item.meeting_time,
            created_at=item.created_at,
            updated_at=item.updated_at,
            meeting_keywords=list(item.meeting_keywords),
            path=str(item.project_dir),
            workflow=WorkflowState.from_summary(workflow) if workflow else None,
        )


class ProjectListResponse(BaseModel):
    """Projects discovered under the configured projects directory."""

    projects_dir: str
    projects: list[ProjectSummary]


class JobRef(BaseModel):
    """A reference to a submitted background job."""

    job_id: str
    kind: str
    status: str


# ---- Speaker review --------------------------------------------------------


class SpeakerSegmentOut(BaseModel):
    """One transcript sentence belonging to a speaker."""

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    text: str
    speaker_id: int | None
    # Optional per-sample voiceprint identity diagnostics (for the review/low filter).
    score: float | None = None
    score_status: str | None = None


class MatchPersonOut(BaseModel):
    """One scored voiceprint person candidate for a speaker."""

    person_id: int | None
    name: str
    score: float | None
    person_public_id: str | None = None


class SpeakerMatchOut(BaseModel):
    """Voiceprint match summary for one speaker."""

    best_name: str | None
    best_score: float | None
    accepted: bool
    threshold: float | None
    status: str
    candidates: list[MatchPersonOut]


class ReviewSpeakerOut(BaseModel):
    """Mutable review state for one detected speaker."""

    speaker_id: int
    label: str
    current_name: str
    ignored: bool
    person_id: int | None
    person_public_id: str | None
    status: str
    crosstalk: bool
    segment_count: int
    duration_ms: int
    match: SpeakerMatchOut | None
    segments: list[SpeakerSegmentOut]


class PersonOut(BaseModel):
    """One known voiceprint person for the identity picker."""

    person_id: int
    name: str
    public_id: str


class ReviewOverviewOut(BaseModel):
    """Stable project/workflow context shown above the review panes."""

    project_id: str
    title: str
    project_status: str
    source_name: str
    duration_ms: int
    match_file_exists: bool


class SpeakerReviewOut(BaseModel):
    """Full speaker-review session for the web UI."""

    project_id: str
    project_dir: str
    overview: ReviewOverviewOut
    speakers: list[ReviewSpeakerOut]
    people: list[PersonOut]
    allow_correction: bool


class ReassignmentIn(BaseModel):
    """One sentence reassignment requested by the client."""

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    original_speaker_id: int | None
    new_speaker_id: int


class SaveSpeakerReviewIn(BaseModel):
    """Speaker-review save request (mirrors the TUI decision)."""

    # JSON object keys are strings; speaker ids are parsed server-side.
    mapping: dict[str, str] = {}
    person_mapping: dict[str, int] = {}
    person_public_mapping: dict[str, str] = {}
    ignored_speaker_ids: list[int] = []
    reassignments: list[ReassignmentIn] = []


class SaveSpeakerReviewOut(BaseModel):
    """Result of a speaker-review save."""

    mapping_path: str
    transcript_path: str
    srt_path: str
    reassigned_count: int
    deleted_sample_count: int
    rematch_skipped_reason: str | None = None


# ---- Voiceprint library / people / quality ---------------------------------


class VoiceprintPersonOut(BaseModel):
    """One person in the global voiceprint registry."""

    person_id: int
    public_id: str
    name: str
    sample_count: int
    project_count: int
    embedded_sample_count: int
    embedding_model_count: int
    updated_at: str | None


class VoiceprintSampleOut(BaseModel):
    """One stored voiceprint sample."""

    index: int  # 1-based position within the person's sample list (delete key)
    sample_id: int
    public_id: str
    speaker_public_id: str
    speaker_name: str
    project_id: str
    begin_time_ms: int
    end_time_ms: int
    transcript_text: str
    status: str
    clip_rel_path: str


class VoiceprintLibraryOut(BaseModel):
    """The global voiceprint registry overview."""

    store_dir: str | None
    people: list[VoiceprintPersonOut]


class VoiceprintSamplesOut(BaseModel):
    """Samples for one person."""

    person: VoiceprintPersonOut
    samples: list[VoiceprintSampleOut]


class QualitySampleOut(BaseModel):
    """One quality-scored sample."""

    sample_public_id: str
    project_id: str
    begin_time_ms: int
    end_time_ms: int
    transcript_text: str
    status: str
    score: float | None
    label: str
    reason: str


class QualityPersonOut(BaseModel):
    """Quality diagnostics for one person."""

    speaker_id: int
    public_id: str
    name: str
    sample_count: int
    active_sample_count: int
    mean_score: float | None
    stdev_score: float | None
    suspicious_count: int
    critical_count: int
    samples: list[QualitySampleOut]


class QualityReportOut(BaseModel):
    """Voiceprint quality report."""

    model: str
    sample_count: int
    suspicious_count: int
    critical_count: int
    people: list[QualityPersonOut]


class CreatePersonIn(BaseModel):
    """Create a new voiceprint person."""

    name: str


class RenamePersonIn(BaseModel):
    """Rename a voiceprint person."""

    name: str


class MergePeopleIn(BaseModel):
    """Merge one person into another."""

    from_ref: str
    into_ref: str


class SampleStatusIn(BaseModel):
    """Update one sample's lifecycle status."""

    status: str
