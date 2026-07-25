"""Pydantic response/request models for the web API.

These mirror the internal frozen dataclasses (``ProjectListItem``,
``ProjectWorkflowSummary``, ...) at the HTTP boundary. Keeping a dedicated schema layer
means internal model changes do not silently reshape the public API, and FastAPI gets
the OpenAPI/validation it expects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
    # Any non-ignored speaker match still unresolved (excludes crosstalk) -- the
    # list's key decision signal, deliberately not folded into workflow.state_key
    # (the file-derived state machine is shared with the CLI).
    has_unresolved_matches: bool = False

    @classmethod
    def from_item(
        cls,
        item: ProjectListItem,
        workflow: ProjectWorkflowSummary | None = None,
        *,
        has_unresolved_matches: bool = False,
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
            has_unresolved_matches=has_unresolved_matches,
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
    # True when the submit deduplicated onto an already-running identical job
    # (same kind + project); the client re-attaches instead of starting anew.
    existing: bool = False


# ---- Ingestion pipeline ----------------------------------------------------


class RunPipelineIn(BaseModel):
    """Run the full project pipeline (create -> ASR -> summarize -> match)."""

    input_path: str
    extra_inputs: list[str] = []
    title: str | None = None
    meeting_time: str | None = None
    variant: str | None = None
    model: str = "fun-asr"
    language: str | None = "zh,en"
    speaker_count: int | None = None
    oss_upload: str = "auto"
    audio_format: str = "flac"
    asr_hotwords: str = "auto"
    summarize: bool = True
    polish: bool = True
    local_correction: bool = True
    # None means "resolve from config at run time"; a literal default here would
    # freeze the built-in threshold at import and ignore voiceprint.match_threshold.
    match_threshold: float | None = None


class SummarizeIn(BaseModel):
    """Generate meeting memory-index artifacts for a project."""

    model: str | None = None
    update_title: bool = True


class MergePreviewIn(BaseModel):
    """Merge several projects into one transcript (preview, no write)."""

    project_refs: list[str]
    use_corrected: bool = True
    name_to_vpp: bool = True
    include_low_information: bool = False
    keep_order: bool = False
    title: str | None = None


class MergeApplyIn(MergePreviewIn):
    """Merge several projects and write the output bundle."""

    out_dir: str
    # Mirror the CLI: refuse to write into a non-empty directory unless force is set, so a
    # merge never silently overwrites an existing bundle.
    force: bool = False


# ---- Transcript correction -------------------------------------------------


class PolishIn(BaseModel):
    """Generate a transcript polish proposal."""

    model: str | None = None
    legacy: bool = False


class CorrectionChangeOut(BaseModel):
    """One proposed transcript change."""

    index: int
    sentence_id: int | None
    sentence_ref: str | None = None
    begin_time_ms: int | None = None
    end_time_ms: int | None = None
    speaker_name: str
    original_text: str
    corrected_text: str
    change_type: str
    reason: str


class ProposalOut(BaseModel):
    """A pending transcript correction proposal."""

    model: str
    change_count: int
    changes: list[CorrectionChangeOut]
    proposal_id: (
        str  # content hash; accept must echo it so a regenerate is not mis-applied
    )


class AcceptCorrectionIn(BaseModel):
    """Accept a correction proposal, optionally only the selected change indices."""

    selected_indices: list[int] | None = None
    # The proposal_id the user reviewed; the accept is refused if the on-disk proposal changed
    # (regenerated in another tab/CLI) so reviewed indices are never applied to a different one.
    proposal_id: str


class AcceptCorrectionOut(BaseModel):
    """Result of accepting a correction proposal."""

    accepted: bool
    change_count: int
    learned_count: int
    corrected_transcript_path: str | None


class DiscardProposalOut(BaseModel):
    """Result of discarding a proposal (the file is archived, not deleted)."""

    discarded: bool
    archived_name: str


class ProjectUpdateIn(BaseModel):
    """Edit a project's user-facing metadata (None = leave unchanged)."""

    title: str | None = None
    meeting_time: str | None = None


# ---- Project artifacts -------------------------------------------------------


class ArtifactOut(BaseModel):
    """One downloadable project artifact (final deliverable)."""

    name: str
    kind: str  # transcript | subtitle | summary
    corrected: bool
    file_name: str
    size_bytes: int
    media_type: str


class ArtifactListOut(BaseModel):
    """The project's existing exportable artifacts, deliverables first."""

    project_id: str
    artifacts: list[ArtifactOut]


# ---- Lexicon ---------------------------------------------------------------


class LexiconTermOut(BaseModel):
    """One cross-project lexicon term."""

    term_id: int
    public_id: str
    canonical: str
    category: str
    description: str
    status: str
    alias_count: int
    context_count: int
    ambiguous_alias_count: int
    created_at: str
    updated_at: str


class LexiconTermsOut(BaseModel):
    """A page of lexicon terms."""

    terms: list[LexiconTermOut]


class UpsertTermIn(BaseModel):
    """Create or update a lexicon term.

    ``category`` / ``description`` / ``status`` default to None = "preserve the
    existing value": an alias-only upsert from the web must not clobber a curated
    category back to ``unknown``, blank a description, or resurrect a deactivated
    term (new terms fall back to unknown/empty/active).
    """

    canonical: str
    category: str | None = None
    description: str | None = None
    aliases: list[str] = []
    status: str | None = None


class LexiconAliasOut(BaseModel):
    """One alias attached to a term (with its optional disambiguation guidance)."""

    alias: str
    alias_type: str
    disambiguation: str | None = None


class LexiconContextOut(BaseModel):
    """One stored correction context for a term."""

    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str
    speaker_name: str | None
    project_id: str
    sentence_id: int | None
    source: str
    created_at: str


class LexiconTermDetailOut(BaseModel):
    """Full term detail: aliases + recent correction contexts."""

    term: LexiconTermOut
    aliases: list[LexiconAliasOut]
    contexts: list[LexiconContextOut]


class SetDisambiguationIn(BaseModel):
    """Mark an alias as context-ambiguous. An empty ``guidance`` clears it, returning the
    alias to deterministic blanket replacement (mirrors ``lexicon disambiguate``)."""

    term: str
    alias: str
    guidance: str = ""


class LexiconStatsOut(BaseModel):
    """Aggregate lexicon statistics."""

    active_terms: int
    inactive_terms: int
    aliases: int
    contexts: int
    hotwords: int
    cached_vocabularies: int


class DisambiguationOut(BaseModel):
    """One context-dependent alias with user guidance."""

    alias: str
    canonical: str
    category: str
    guidance: str


class HotwordOut(BaseModel):
    """One ASR hotword entry."""

    text: str
    weight: int
    category: str
    source: str


# ---- Config + diagnostics --------------------------------------------------


class ConfigKeyOut(BaseModel):
    """One global config key and its current value."""

    name: str
    env_name: str
    secret: bool
    is_set: bool
    value: str | None  # null for masked (unrevealed) secrets


class ConfigOut(BaseModel):
    """Global configuration."""

    config_file: str
    keys: list[ConfigKeyOut]


class SetConfigIn(BaseModel):
    """Set one config key."""

    key: str
    value: str


class DoctorCheckOut(BaseModel):
    """One diagnostic check result."""

    name: str
    status: str  # ok | warn | fail
    detail: str
    fix_prompt: str | None


class DoctorOut(BaseModel):
    """Environment diagnostics."""

    ok: bool
    checks: list[DoctorCheckOut]


# ---- Speaker review --------------------------------------------------------


class SpeakerSegmentOut(BaseModel):
    """One transcript sentence belonging to a speaker."""

    sentence_id: int | None
    sentence_ref: str | None = None
    begin_time_ms: int
    end_time_ms: int
    text: str
    speaker_id: int | None
    # Optional per-sample voiceprint identity diagnostics (for the review/low filter).
    score: float | None = None
    score_status: str | None = None
    score_best_name: str | None = None
    score_best_score: float | None = None
    score_best_other_name: str | None = None
    score_best_other_score: float | None = None
    score_margin: float | None = None


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
    review_revision: str
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


class InlineCorrectionIn(BaseModel):
    """One sentence text correction staged from speaker review."""

    sentence_id: int | None
    speaker_id: int | None
    begin_time_ms: int
    end_time_ms: int
    original_text: str
    corrected_text: str


class SaveSpeakerReviewIn(BaseModel):
    """Speaker-review save request (mirrors the TUI decision)."""

    review_revision: str
    # JSON object keys are strings; speaker ids are parsed server-side.
    mapping: dict[str, str] = {}
    person_mapping: dict[str, int] = {}
    person_public_mapping: dict[str, str] = {}
    new_person_names: dict[str, str] = {}
    ignored_speaker_ids: list[int] = []
    reassignments: list[ReassignmentIn] = []
    deleted_speaker_ids: list[int] = []
    correction_edits: list[InlineCorrectionIn] = []


class SaveSpeakerReviewOut(BaseModel):
    """Result of a speaker-review save."""

    mapping_path: str
    transcript_path: str
    srt_path: str
    reassigned_count: int
    created_person_count: int = 0
    deleted_speaker_count: int = 0
    deleted_sentence_count: int = 0
    deleted_sample_count: int
    corrected_count: int = 0
    corrected_transcript_path: str | None = None
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
    identity_confirmed: bool
    matching_enabled: bool
    clip_rel_path: str
    # Another speaker holds the floor within half a second of this sample's
    # source audio, so the clip is a mixture rather than one voice. None means
    # the source project was unavailable and the check did not run -- clients
    # must not render that as "checked and clean".
    overlap_risk: bool | None = None


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
    identity_confirmed: bool
    matching_enabled: bool
    score: float | None
    label: str
    reason: str


class QualityProjectOut(BaseModel):
    """Quality diagnostics for one source project."""

    project_id: str
    sample_count: int
    matching_sample_count: int
    suspicious_count: int
    critical_count: int
    mean_score: float | None
    min_score: float | None


class QualityNeighborOut(BaseModel):
    """Closest other voiceprint person."""

    speaker_id: int
    public_id: str
    name: str
    score: float


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
    projects: list[QualityProjectOut]
    closest_people: list[QualityNeighborOut]
    samples: list[QualitySampleOut]


class QualityReportOut(BaseModel):
    """Voiceprint quality report."""

    model: str
    sample_count: int
    suspicious_count: int
    critical_count: int
    people: list[QualityPersonOut]


# ---- Voiceprint library health ---------------------------------------------


class PersonHealthOut(BaseModel):
    """Availability facts for one library person under the active model."""

    public_id: str
    name: str
    total_sample_count: int
    enabled_sample_count: int
    matching_sample_count: int
    missing_embedding_count: int
    # Missing embeddings whose clip audio is gone; a backfill cannot fix these.
    missing_clip_count: int
    embeddable_count: int
    matching_seconds: float
    project_count: int
    availability: str


class LibraryIssueOut(BaseModel):
    """One actionable library problem."""

    kind: str
    severity: str
    title: str
    detail: str
    action: str
    person_public_id: str | None = None
    person_name: str | None = None
    # Numbers behind title/detail so the bilingual UI can restate them.
    context: dict[str, float | int | str] = Field(default_factory=dict)


class ScoreDistributionOut(BaseModel):
    """Summary statistics for one calibration score population."""

    count: int
    min: float
    p5: float
    median: float
    p95: float
    max: float


class ThresholdCostOut(BaseModel):
    """What one candidate threshold costs on the current library."""

    threshold: float
    false_reject_count: int
    false_reject_rate: float
    false_accept_count: int
    false_accept_rate: float


class CalibrationOut(BaseModel):
    """Threshold calibration evidence for the active model."""

    model: str
    person_count: int
    scored_person_count: int
    sample_count: int
    genuine: ScoreDistributionOut | None
    impostor: ScoreDistributionOut | None
    eer_threshold: float | None
    eer_rate: float | None
    low_impostor_threshold: float | None
    current_threshold: float
    warnings: list[str]
    # Raw populations so the UI can price a dragged threshold locally.
    genuine_scores: list[float]
    impostor_scores: list[float]
    suggested_threshold: float | None
    suggested_reason: str
    suggested_kind: str
    low_confidence: bool
    current_cost: ThresholdCostOut | None
    suggested_cost: ThresholdCostOut | None


class LibraryHealthOut(BaseModel):
    """Voiceprint library availability and threshold health."""

    db_path: str
    provider: str
    model: str
    person_count: int
    usable_person_count: int
    matching_sample_count: int
    matching_seconds: float
    critical_count: int
    warning_count: int
    people: list[PersonHealthOut]
    issues: list[LibraryIssueOut]
    calibration: CalibrationOut | None


class MatchThresholdIn(BaseModel):
    """Set or clear the configured voiceprint match threshold."""

    # None clears the config key and restores the built-in default.
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class MatchThresholdOut(BaseModel):
    """The effective voiceprint match threshold and its coupling warnings."""

    effective: float
    configured: float | None
    default: float
    warnings: list[str]
    warning_kinds: list[str]


class EmbedBacklogOut(BaseModel):
    """How many stored samples still need an embedding for the active model."""

    model: str
    # Only samples a backfill could actually embed (clip audio still readable).
    missing_sample_count: int
    person_count: int
    missing_clip_count: int


class CandidateClipOut(BaseModel):
    """One clip a capture run would take, in capture's own identity terms.

    ``rel_path`` plus the times are exactly what ``POST /capture/{ref}/run``
    validates a selection against, so a client can capture these directly
    without re-planning.
    """

    rel_path: str
    begin_time_ms: int
    end_time_ms: int
    duration_seconds: float
    text: str
    score: float
    recommended: bool
    # Another speaker within half a second: usable, but never pre-selected.
    overlap_risk: bool = False


class SampleSourceOut(BaseModel):
    """One project holding harvestable speech for a person."""

    project_id: str
    title: str
    meeting_time: str | None
    speaker_id: int
    speaker_name: str
    # The person the capture plan resolved; echo it back when capturing.
    person_public_id: str | None
    # "person-map" (linked) or "name" (display name only).
    evidence: str
    candidate_count: int
    candidate_seconds: float
    best_score: float
    matching_sample_count: int
    quarantined_sample_count: int
    priority: float
    # Reason kinds, re-rendered per locale by the browser.
    reasons: list[str]
    clips: list[CandidateClipOut]


class SkippedProjectOut(BaseModel):
    """A project the sourcing scan could not read."""

    project_id: str
    title: str
    reason: str


class SampleSourcesOut(BaseModel):
    """Ranked places to harvest more samples for one person."""

    person_public_id: str
    person_name: str
    # Which health bars this person currently fails; these ordered the sources.
    deficits: list[str]
    matching_sample_count: int
    matching_seconds: float
    project_count: int
    scanned_project_count: int
    total_candidate_seconds: float
    new_project_count: int
    sources: list[SampleSourceOut]
    skipped: list[SkippedProjectOut]


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


class ExcludeQualitySamplesIn(BaseModel):
    """Bulk-exclude low-quality samples for one person."""

    sample_public_ids: list[str] | None = None
    include_warnings: bool = True


class ExcludeQualitySamplesOut(BaseModel):
    """Result of a bulk low-quality exclusion."""

    updated_count: int
    sample_public_ids: list[str]


class SpeakerRematchOut(BaseModel):
    """Result of refreshing project speaker voiceprint matches."""

    matched_count: int
    below_threshold_count: int
    total_count: int


# ---- Voiceprint capture workflow -------------------------------------------


class CaptureClipOut(BaseModel):
    """One candidate clip in a capture plan."""

    rel_path: str
    begin_time_ms: int
    end_time_ms: int
    duration_seconds: float
    text: str
    selection_score: float
    selection_reason: str
    audio_score: float | None
    audio_reason: str
    recommended: bool
    # Another speaker within half a second: usable, but never pre-selected.
    overlap_risk: bool = False


class CaptureSpeakerOut(BaseModel):
    """One named speaker's candidate clips."""

    speaker_id: int
    name: str
    person_public_id: str | None
    clips: list[CaptureClipOut]


class CapturePlanOut(BaseModel):
    """Dry-run capture plan for one project."""

    project_ref: str
    target_sample_count: int
    sample_count: int
    speakers: list[CaptureSpeakerOut]


class SelectedCaptureClipIn(BaseModel):
    """One clip the user picked, carrying the stable identifiers it was shown under.

    Capture rel_paths are index-based (``speaker_N/clip_NNN.wav``); the plan is recomputed at
    capture time, so the server validates these markers against the fresh plan to detect a plan
    that drifted (project edited between planning and capture) before embedding. begin/end guard
    the audio; name + person_public_id guard the IDENTITY (a rename / vpp rebind can keep the
    same rel_path+times yet store the clip under a different person than the user reviewed).
    """

    rel_path: str
    begin_time_ms: int
    end_time_ms: int
    name: str
    person_public_id: str | None = None


class CaptureRunIn(BaseModel):
    """Request to run capture for the selected clips."""

    selected_clips: list[SelectedCaptureClipIn] = Field(min_length=1, max_length=100)
    sample_count: int = Field(default=3, ge=1, le=10)
    max_seconds: float = Field(default=12.0, gt=0, le=60.0)
    padding_seconds: float = Field(default=0.5, ge=0, le=5.0)


class ScoreChangeOut(BaseModel):
    """One speaker's before/after voiceprint match score change."""

    speaker_id: int
    label: str
    before_name: str | None
    before_score: float | None
    after_name: str | None
    after_score: float | None
    delta: float | None
    status: str  # improved | declined | changed-best | lost-candidate | unchanged | ...
    is_critical: bool
    is_warning: bool
    threshold: float | None


class HistoricalProjectOut(BaseModel):
    """One historical project's regression risk after re-embedding."""

    project_id: str
    title: str
    improved: int
    declined: int
    changed_best: int
    warning_count: int
    critical_count: int
    risky_changes: list[ScoreChangeOut]


class CaptureResultOut(BaseModel):
    """Result of a capture+embed+evaluate run (pending accept/rollback).

    Carries the full per-speaker and per-project detail the TUI result screen shows, so
    the accept/rollback decision is informed -- not just aggregate counts.
    """

    transaction_id: str
    captured_count: int
    embedded_count: int
    skipped_count: int
    quality_gate_reviewed_count: int
    quality_gate_excluded_count: int
    quality_gate_warning_count: int
    quality_gate_critical_count: int
    current_project_id: str
    current_changes: list[ScoreChangeOut]
    current_improved: int
    current_declined: int
    current_changed_best: int
    current_warning: int
    current_critical: int
    historical_project_count: int
    historical_warning_count: int
    historical_critical_count: int
    historical_projects: list[HistoricalProjectOut]


class PendingCaptureOut(BaseModel):
    """A capture transaction awaiting accept/rollback, for the app-wide recovery banner.

    ``project_id`` may be null if the project the capture belongs to is no longer loadable;
    the transaction is still resolvable by its id.
    """

    transaction_id: str
    project_id: str | None
