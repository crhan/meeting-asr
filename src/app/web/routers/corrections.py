"""Transcript correction: polish (LLM) -> review proposal -> accept selected changes.

Reuses the same correction functions the CLI and the speaker-review save use
(``prepare_transcript_polish_for_review`` / ``load_correction_proposal`` /
``accept_correction_for_review``), so the web review accepts the exact same proposal the
CLI would. Polish calls an LLM, so it runs as a background job; accept runs under the
project lock.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.commands.project_correct import (
    accept_correction_for_review,
    load_speaker_mapping_for_correction,
    prepare_transcript_polish_for_review,
)
from app.core.project_refs import resolve_project_ref
from app.correction_proposals import load_correction_proposal
from app.project_manager import load_manifest, project_paths
from app.transcript_corrections import CorrectionEditOptions
from app.web.deps import get_jobs, get_locks, get_settings, require_auth
from app.web.jobs import JobManager
from app.web.locks import LockRegistry, project_lock_key
from app.web.schemas import (
    AcceptCorrectionIn,
    AcceptCorrectionOut,
    CorrectionChangeOut,
    JobRef,
    PolishIn,
    ProposalOut,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/corrections",
    tags=["corrections"],
    dependencies=[Depends(require_auth)],
)


@router.post("/{project_ref}/polish", response_model=JobRef)
def polish(
    project_ref: str,
    payload: PolishIn,
    settings: WebSettings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> JobRef:
    """Generate a transcript polish proposal via LLM (background job)."""
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)

    def work(reporter) -> dict[str, object]:
        paths = project_paths(project_dir)
        manifest = load_manifest(paths.root)
        speaker_mapping = load_speaker_mapping_for_correction(paths.root)
        options = CorrectionEditOptions(
            open_editor=False,
            open_proposal=False,
            use_ai=True,
            model=payload.model,
            polish_legacy=payload.legacy,
        )
        summary = prepare_transcript_polish_for_review(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            options=options,
            progress=reporter,
        )
        return {
            "proposed_change_count": summary.proposed_change_count,
            "model": summary.model,
            "model_error": summary.model_error,
        }

    job = jobs.submit("correction-polish", work, project_id=str(project_dir))
    return JobRef(job_id=job.id, kind=job.kind, status=job.status)


@router.get("/{project_ref}/proposal", response_model=ProposalOut)
def get_proposal(
    project_ref: str, settings: WebSettings = Depends(get_settings)
) -> ProposalOut:
    """Load the latest pending transcript correction proposal."""
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)
    paths = project_paths(project_dir)
    proposal = load_correction_proposal(paths, None)
    changes = [
        CorrectionChangeOut(
            index=index,
            sentence_id=change.sentence_id,
            speaker_name=change.speaker_name,
            original_text=change.original_text,
            corrected_text=change.corrected_text,
            change_type=change.change_type,
            reason=change.reason,
        )
        for index, change in enumerate(proposal.proposed_changes)
    ]
    return ProposalOut(model=proposal.model, change_count=len(changes), changes=changes)


@router.post("/{project_ref}/accept", response_model=AcceptCorrectionOut)
async def accept(
    project_ref: str,
    payload: AcceptCorrectionIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> AcceptCorrectionOut:
    """Accept the pending proposal (optionally only the selected change indices)."""
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)
    selected = (
        tuple(payload.selected_indices)
        if payload.selected_indices is not None
        else None
    )

    def work() -> AcceptCorrectionOut:
        paths = project_paths(project_dir)
        manifest = load_manifest(paths.root)
        speaker_mapping = load_speaker_mapping_for_correction(paths.root)
        summary = accept_correction_for_review(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            proposal_path=None,
            lexicon_db=None,
            selected_change_indices=selected,
        )
        return AcceptCorrectionOut(
            accepted=summary.accepted,
            change_count=summary.change_count,
            learned_count=summary.learned_count,
            corrected_transcript_path=(
                str(summary.corrected_named_transcript_path)
                if summary.corrected_named_transcript_path
                else None
            ),
        )

    loop = asyncio.get_running_loop()
    async with locks.acquire(project_lock_key(str(project_dir))):
        return await loop.run_in_executor(None, work)
