"""Transcript correction: polish (LLM) -> review proposal -> accept selected changes.

Reuses the same correction functions the CLI and the speaker-review save use
(``prepare_transcript_polish_for_review`` / ``load_correction_proposal`` /
``accept_correction_for_review``), so the web review accepts the exact same proposal the
CLI would. Polish calls an LLM, so it runs as a background job; accept runs under the
project lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException

from app.commands.project_correct import (
    accept_correction_for_review,
    load_speaker_mapping_for_correction,
    prepare_transcript_polish_for_review,
)
from app.core.voiceprint_review_service import REGISTRY, CaptureConflictError
from app.correction_proposals import (
    archive_correction_proposal,
    load_correction_proposal,
)
from app.lexicon_store import get_lexicon_db_path
from app.project_manager import load_manifest, project_paths
from app.sentence_locator import format_sentence_ref
from app.transcript_corrections import CorrectionEditOptions
from app.web.deps import (
    get_jobs,
    get_locks,
    get_settings,
    require_auth,
    resolve_web_project_ref,
)
from app.web.jobs import JobManager
from app.web.locks import LockRegistry, project_lock_key, store_lock_key
from app.web.schemas import (
    AcceptCorrectionIn,
    AcceptCorrectionOut,
    CorrectionChangeOut,
    DiscardProposalIn,
    DiscardProposalOut,
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
    project_dir = resolve_web_project_ref(project_ref, settings)

    lexicon_db = get_lexicon_db_path(settings.store_dir)

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
            # Disambiguation guidance must come from the same (possibly isolated) store the
            # accept step writes to, not the real XDG lexicon.
            lexicon_db=lexicon_db,
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

    job, existing = jobs.submit(
        "correction-polish", work, project_id=str(project_dir)
    )
    return JobRef(job_id=job.id, kind=job.kind, status=job.status, existing=existing)


def _proposal_id(proposal) -> str:
    """Content hash identifying exactly this proposal.

    Lets an accept bind to the proposal the user reviewed: a regenerate (another tab / the CLI)
    changes the changes, hence the hash, so the accept is refused instead of applying the
    reviewed selection indices to a different proposal.
    """
    payload = [
        [c.sentence_id, c.original_text, c.corrected_text, c.change_type, c.reason]
        for c in proposal.proposed_changes
    ]
    blob = json.dumps([proposal.model, payload], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _audio_window(change) -> tuple[int | None, int | None]:
    """Return a valid sentence audio window, or nulls for legacy/malformed proposals."""
    try:
        begin = int(getattr(change, "begin_time_ms", None))
        end = int(getattr(change, "end_time_ms", None))
    except TypeError, ValueError:
        return None, None
    if end <= begin:
        return None, None
    return begin, end


@router.get("/{project_ref}/proposal", response_model=ProposalOut)
def get_proposal(
    project_ref: str, settings: WebSettings = Depends(get_settings)
) -> ProposalOut:
    """Load the latest pending transcript correction proposal."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    paths = project_paths(project_dir)
    try:
        proposal = load_correction_proposal(paths, None)
    except RuntimeError as exc:
        # Only "no proposal file" is a not-found condition -> 404, where the correction page
        # renders its "no pending proposal" empty state. load_correction_proposal also raises
        # RuntimeError for a MALFORMED proposal (non-object JSON); that is corruption, not
        # absence, so let it surface as a 500 instead of hiding a repairable file behind 404.
        if "No correction proposal found" not in str(exc):
            raise
        raise FileNotFoundError(str(exc)) from exc
    manifest = load_manifest(paths.root)
    changes = []
    for index, change in enumerate(proposal.proposed_changes):
        begin_time_ms, end_time_ms = _audio_window(change)
        changes.append(
            CorrectionChangeOut(
                index=index,
                sentence_id=change.sentence_id,
                sentence_ref=format_sentence_ref(
                    manifest.project_id, change.sentence_id
                ),
                begin_time_ms=begin_time_ms,
                end_time_ms=end_time_ms,
                speaker_name=change.speaker_name,
                original_text=change.original_text,
                corrected_text=change.corrected_text,
                change_type=change.change_type,
                reason=change.reason,
            )
        )
    return ProposalOut(
        model=proposal.model,
        change_count=len(changes),
        changes=changes,
        proposal_id=_proposal_id(proposal),
    )


@router.delete("/{project_ref}/proposal", response_model=DiscardProposalOut)
async def discard_proposal(
    project_ref: str,
    payload: DiscardProposalIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> DiscardProposalOut:
    """Discard the pending proposal without applying anything (archives the file)."""
    project_dir = resolve_web_project_ref(project_ref, settings)

    def work() -> DiscardProposalOut:
        paths = project_paths(project_dir)
        try:
            proposal = load_correction_proposal(paths, None)
        except RuntimeError as exc:
            # Mirror get_proposal: only absence maps to 404; malformed stays a 500.
            if "No correction proposal found" not in str(exc):
                raise
            raise FileNotFoundError(str(exc)) from exc
        if _proposal_id(proposal) != payload.proposal_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The correction proposal changed since you reviewed it. "
                    "Reload the proposal before discarding."
                ),
            )
        archived = archive_correction_proposal(proposal, suffix="discarded")
        return DiscardProposalOut(discarded=True, archived_name=archived.name)

    loop = asyncio.get_running_loop()
    async with locks.acquire(project_lock_key(str(project_dir))):
        return await loop.run_in_executor(None, work)


@router.post("/{project_ref}/accept", response_model=AcceptCorrectionOut)
async def accept(
    project_ref: str,
    payload: AcceptCorrectionIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> AcceptCorrectionOut:
    """Accept the pending proposal (optionally only the selected change indices)."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    selected = (
        tuple(payload.selected_indices)
        if payload.selected_indices is not None
        else None
    )
    # Learned correction contexts are recorded into the lexicon; honor --store-dir so an
    # isolated experiment never leaks into the real XDG correction dictionary.
    lexicon_db = get_lexicon_db_path(settings.store_dir)

    def work() -> AcceptCorrectionOut:
        paths = project_paths(project_dir)
        manifest = load_manifest(paths.root)
        speaker_mapping = load_speaker_mapping_for_correction(paths.root)
        # Bind the accept to the reviewed proposal: if it was regenerated (another tab/CLI)
        # since the user reviewed it, the selected indices belong to a different proposal, so
        # applying them would write the wrong subset. Refuse (409) and let the user re-review.
        current = load_correction_proposal(paths, None)
        if _proposal_id(current) != payload.proposal_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The correction proposal changed since you reviewed it. "
                    "Reload the proposal and re-select before accepting."
                ),
            )
        summary = accept_correction_for_review(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            proposal_path=None,
            lexicon_db=lexicon_db,
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

    # accept_correction_for_review records learned contexts into the shared lexicon
    # SQLite, so the per-project lock alone is not enough -- hold the lexicon store lock
    # too, the same one the lexicon router takes, or concurrent writes race the DB.
    loop = asyncio.get_running_loop()
    async with locks.acquire(
        project_lock_key(str(project_dir)), store_lock_key("lexicon")
    ):
        # This writes the project's manifest/transcript, which a pending capture's rollback
        # would restore to the pre-capture snapshot -- silently dropping the accepted
        # corrections. Refuse under the project lock (where the capture registers its txn).
        if REGISTRY.has_pending():
            raise CaptureConflictError(
                "A voiceprint capture is awaiting accept/rollback; resolve it before "
                "accepting corrections."
            )
        return await loop.run_in_executor(None, work)
