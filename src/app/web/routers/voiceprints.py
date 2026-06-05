"""Global voiceprint registry: library browse, people CRUD, quality, clip serving.

All reads/writes target the configured store (``settings.store_dir`` -> db path), never a
request-supplied path -- that is the guard against mutating the wrong voiceprint library.
Writes to the shared global store run in the executor under the per-store lock.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.project_refs import resolve_project_ref
from app.core.voiceprint_review_service import (
    REGISTRY,
    CaptureConflictError,
    plan_capture,
)
from app.voiceprint_models import VoiceprintSampleRow, VoiceprintSpeakerRow
from app.voiceprint_people import (
    create_voiceprint_person,
    get_voiceprint_person,
    merge_voiceprint_people,
    rename_voiceprint_person,
)
from app.voiceprint_quality import analyze_voiceprint_quality
from app.voiceprint_store import (
    delete_voiceprint_sample,
    delete_voiceprint_speaker,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    list_voiceprint_speakers,
    update_voiceprint_sample_status,
)
from app.web.deps import get_jobs, get_locks, get_settings, require_auth
from app.web.jobs import JobManager
from app.web.locks import LockRegistry, store_lock_key
from app.web.schemas import (
    CaptureClipOut,
    CapturePlanOut,
    CaptureResultOut,
    CaptureRunIn,
    CaptureSpeakerOut,
    CreatePersonIn,
    HistoricalProjectOut,
    JobRef,
    ScoreChangeOut,
    MergePeopleIn,
    QualityPersonOut,
    QualityReportOut,
    QualitySampleOut,
    RenamePersonIn,
    SampleStatusIn,
    VoiceprintLibraryOut,
    VoiceprintPersonOut,
    VoiceprintSampleOut,
    VoiceprintSamplesOut,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/voiceprints",
    tags=["voiceprints"],
    dependencies=[Depends(require_auth)],
)

_STORE_LOCK = store_lock_key("voiceprints")


def _person_out(row: VoiceprintSpeakerRow) -> VoiceprintPersonOut:
    return VoiceprintPersonOut(
        person_id=row.speaker_id,
        public_id=row.public_id,
        name=row.name,
        sample_count=row.sample_count,
        project_count=row.project_count,
        embedded_sample_count=row.embedded_sample_count,
        embedding_model_count=row.embedding_model_count,
        updated_at=row.updated_at,
    )


def _sample_out(index: int, row: VoiceprintSampleRow) -> VoiceprintSampleOut:
    return VoiceprintSampleOut(
        index=index,
        sample_id=row.sample_id,
        public_id=row.public_id,
        speaker_public_id=row.speaker_public_id,
        speaker_name=row.speaker_name,
        project_id=row.project_id,
        begin_time_ms=row.source_begin_time_ms,
        end_time_ms=row.source_end_time_ms,
        transcript_text=row.transcript_text,
        status=row.sample_status,
        clip_rel_path=row.clip_rel_path,
    )


async def _run(locks: LockRegistry, fn):
    """Run a blocking store write in the executor under the global store lock.

    Refuses the write while a capture transaction is pending: that transaction holds a
    rollback snapshot of this same store, so any edit made now would be silently reverted
    if the user later rolls back. The caller must accept/rollback the capture first.
    """
    if REGISTRY.has_pending():
        raise CaptureConflictError(
            "A voiceprint capture is awaiting accept/rollback; resolve it before editing "
            "the store."
        )
    loop = asyncio.get_running_loop()
    async with locks.acquire(_STORE_LOCK):
        return await loop.run_in_executor(None, fn)


# ---- library + people reads ------------------------------------------------


@router.get("/library", response_model=VoiceprintLibraryOut)
def get_library(settings: WebSettings = Depends(get_settings)) -> VoiceprintLibraryOut:
    """List all people in the global voiceprint registry."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    rows = list_voiceprint_speakers(db_path)
    return VoiceprintLibraryOut(
        store_dir=str(settings.store_dir) if settings.store_dir else None,
        people=[_person_out(r) for r in rows],
    )


@router.get("/people/{ref}/samples", response_model=VoiceprintSamplesOut)
def get_person_samples(
    ref: str, settings: WebSettings = Depends(get_settings)
) -> VoiceprintSamplesOut:
    """List stored samples for one person."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    person = get_voiceprint_person(ref, db_path)
    if person is None:
        raise FileNotFoundError(f"Voiceprint person not found: {ref}")
    rows = list_voiceprint_samples(ref, db_path)
    return VoiceprintSamplesOut(
        person=_person_out(person),
        samples=[_sample_out(i + 1, row) for i, row in enumerate(rows)],
    )


@router.get("/people/{ref}/clips/{sample_public_id}")
def get_sample_clip(
    ref: str,
    sample_public_id: str,
    settings: WebSettings = Depends(get_settings),
) -> FileResponse:
    """Serve one stored sample's WAV clip (with HTTP Range)."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    rows = list_voiceprint_samples(ref, db_path)
    match = next((r for r in rows if r.public_id == sample_public_id), None)
    if match is None or not match.clip_path.is_file():
        raise FileNotFoundError(f"Sample clip not found: {sample_public_id}")
    return FileResponse(
        match.clip_path,
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=3600"},
    )


# ---- quality ---------------------------------------------------------------


@router.get("/quality", response_model=QualityReportOut)
def get_quality(settings: WebSettings = Depends(get_settings)) -> QualityReportOut:
    """Analyze outlier samples across the voiceprint library."""
    report = analyze_voiceprint_quality(store_dir=settings.store_dir)
    people = [
        QualityPersonOut(
            speaker_id=p.speaker_id,
            public_id=p.speaker_public_id,
            name=p.speaker_name,
            sample_count=p.sample_count,
            active_sample_count=p.active_sample_count,
            mean_score=p.mean_score,
            stdev_score=p.stdev_score,
            suspicious_count=p.suspicious_count,
            critical_count=p.critical_count,
            samples=[
                QualitySampleOut(
                    sample_public_id=s.sample_public_id,
                    project_id=s.project_id,
                    begin_time_ms=s.source_begin_time_ms,
                    end_time_ms=s.source_end_time_ms,
                    transcript_text=s.transcript_text,
                    status=s.status,
                    score=s.score,
                    label=s.label,
                    reason=s.reason,
                )
                for s in p.samples
            ],
        )
        for p in report.people
    ]
    return QualityReportOut(
        model=report.model,
        sample_count=report.sample_count,
        suspicious_count=report.suspicious_count,
        critical_count=report.critical_count,
        people=people,
    )


# ---- writes ----------------------------------------------------------------


@router.patch("/samples/{sample_public_id}/status", response_model=VoiceprintSampleOut)
async def set_sample_status(
    sample_public_id: str,
    payload: SampleStatusIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> VoiceprintSampleOut:
    """Update one sample's lifecycle status (active/quarantined/verified-active)."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    row = await _run(
        locks,
        lambda: update_voiceprint_sample_status(
            sample_public_id, payload.status, db_path
        ),
    )
    return _sample_out(0, row)


@router.delete("/people/{ref}/samples/{index}")
async def delete_sample(
    ref: str,
    index: int,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> dict[str, object]:
    """Delete one sample by its 1-based position within the person's sample list."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    deleted = await _run(
        locks, lambda: delete_voiceprint_sample(ref, index, db_path=db_path)
    )
    return {"deleted_sample_public_id": deleted.public_id}


@router.delete("/people/{ref}")
async def delete_person(
    ref: str,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> dict[str, object]:
    """Delete a person and all their stored samples."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    deleted = await _run(locks, lambda: delete_voiceprint_speaker(ref, db_path=db_path))
    return {"deleted_sample_count": len(deleted)}


@router.post("/people", response_model=VoiceprintPersonOut)
async def create_person(
    payload: CreatePersonIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> VoiceprintPersonOut:
    """Create a new voiceprint person."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    row = await _run(locks, lambda: create_voiceprint_person(payload.name, db_path))
    return _person_out(row)


@router.patch("/people/{ref}", response_model=VoiceprintPersonOut)
async def rename_person(
    ref: str,
    payload: RenamePersonIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> VoiceprintPersonOut:
    """Rename a voiceprint person."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    row = await _run(
        locks, lambda: rename_voiceprint_person(ref, payload.name, db_path)
    )
    return _person_out(row)


@router.post("/people/merge", response_model=VoiceprintPersonOut)
async def merge_people(
    payload: MergePeopleIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> VoiceprintPersonOut:
    """Merge one person into another; returns the surviving person."""
    db_path = get_voiceprint_db_path(settings.store_dir)
    await _run(
        locks,
        lambda: merge_voiceprint_people(payload.from_ref, payload.into_ref, db_path),
    )
    survivor = get_voiceprint_person(payload.into_ref, db_path)
    if survivor is None:
        raise FileNotFoundError(f"Merged person not found: {payload.into_ref}")
    return _person_out(survivor)


# ---- capture workflow (plan -> run job -> accept/rollback) ------------------


def _change_out(change) -> ScoreChangeOut:
    return ScoreChangeOut(
        speaker_id=change.speaker_id,
        label=change.label,
        before_name=change.before_name,
        before_score=change.before_score,
        after_name=change.after_name,
        after_score=change.after_score,
        delta=change.delta,
        status=change.status,
        is_critical=change.is_critical,
        is_warning=change.is_warning,
        threshold=change.threshold,
    )


def _historical_out(project) -> HistoricalProjectOut:
    return HistoricalProjectOut(
        project_id=project.project_id,
        title=project.title,
        improved=project.improved_count,
        declined=project.declined_count,
        changed_best=project.changed_best_count,
        warning_count=project.warning_count,
        critical_count=project.critical_count,
        risky_changes=[
            _change_out(c) for c in project.changes if c.is_warning or c.is_critical
        ],
    )


def _plan_out(project_ref: str, summary) -> CapturePlanOut:
    speakers = [
        CaptureSpeakerOut(
            speaker_id=sp.speaker_id,
            name=sp.name,
            person_public_id=sp.person_public_id,
            clips=[
                CaptureClipOut(
                    rel_path=c.rel_path,
                    begin_time_ms=c.source_begin_time_ms,
                    end_time_ms=c.source_end_time_ms,
                    duration_seconds=c.duration_seconds,
                    text=c.text,
                    selection_score=c.selection_score,
                    selection_reason=c.selection_reason,
                    audio_score=c.audio_score,
                    audio_reason=c.audio_reason,
                    recommended=c.recommended,
                )
                for c in sp.clips
            ],
        )
        for sp in summary.speakers
    ]
    return CapturePlanOut(
        project_ref=project_ref,
        target_sample_count=summary.target_sample_count,
        sample_count=summary.sample_count,
        speakers=speakers,
    )


@router.post("/capture/{project_ref}/plan", response_model=CapturePlanOut)
async def capture_plan(
    project_ref: str,
    settings: WebSettings = Depends(get_settings),
) -> CapturePlanOut:
    """Plan voiceprint capture candidates for a project (read-only)."""
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(
        None, lambda: plan_capture(project_dir, store_dir=settings.store_dir)
    )
    return _plan_out(project_ref, summary)


@router.post("/capture/{project_ref}/run", response_model=JobRef)
def capture_run(
    project_ref: str,
    payload: CaptureRunIn,
    settings: WebSettings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> JobRef:
    """Run capture+embed+evaluate for the selected clips as a background job.

    The job result carries a ``transaction_id`` plus the evaluation summary; the client
    then accepts or rolls the transaction back.
    """
    # Fail fast before queueing if a prior capture is unresolved; ``REGISTRY.run`` also
    # enforces this so a race between the check and the executor cannot slip through.
    if REGISTRY.has_pending():
        raise CaptureConflictError(
            "A previous voiceprint capture is still awaiting accept/rollback; "
            "resolve it before starting another."
        )
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)
    store_dir = settings.store_dir

    def work(_reporter) -> dict[str, object]:
        planned = plan_capture(
            project_dir,
            store_dir=store_dir,
            sample_count=payload.sample_count,
            max_seconds=payload.max_seconds,
            padding_seconds=payload.padding_seconds,
        )
        txn_id, summary = REGISTRY.run(
            project_dir=project_dir,
            planned=planned,
            selected_clip_rel_paths=frozenset(payload.selected_clip_rel_paths),
            store_dir=store_dir,
        )
        evaluation = summary.evaluation
        current = evaluation.current
        return CaptureResultOut(
            transaction_id=txn_id,
            captured_count=summary.capture.sample_count,
            embedded_count=summary.embedding.embedded_count,
            skipped_count=summary.embedding.skipped_count,
            current_project_id=current.project_id,
            current_changes=[_change_out(c) for c in current.changes],
            current_improved=current.improved_count,
            current_declined=current.declined_count,
            current_changed_best=current.changed_best_count,
            current_warning=current.warning_count,
            current_critical=current.critical_count,
            historical_project_count=evaluation.historical_project_count,
            historical_warning_count=evaluation.historical_warning_count,
            historical_critical_count=evaluation.historical_critical_count,
            historical_projects=[
                _historical_out(project)
                for project in evaluation.historical
                if project.risk_count > 0
            ],
        ).model_dump()

    job = jobs.submit("voiceprint-capture", work, project_id=str(project_dir))
    return JobRef(job_id=job.id, kind=job.kind, status=job.status)


@router.post("/capture/transactions/{transaction_id}/accept")
async def capture_accept(transaction_id: str) -> dict[str, str]:
    """Accept a pending capture transaction (keep the changes)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: REGISTRY.accept(transaction_id))
    return {"status": "accepted"}


@router.post("/capture/transactions/{transaction_id}/rollback")
async def capture_rollback(transaction_id: str) -> dict[str, str]:
    """Roll back a pending capture transaction (undo the changes)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: REGISTRY.rollback(transaction_id))
    return {"status": "rolled_back"}
