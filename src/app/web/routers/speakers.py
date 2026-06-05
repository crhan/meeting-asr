"""Speaker review: load the review session and persist a save decision.

Reads reuse the canonical TUI loader (``load_speaker_review_session``), which is pure
data loading. Saves go through the shared ``app.core.speaker_review_service`` so the web
and the CLI take the identical save path (including the voiceprint-sample invalidation +
rematch on reassignment). Writes run in the executor under per-project (and, when a
reassignment touches the global voiceprint store, per-store) locks.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.core.speaker_review_service import save_speaker_review
from app.core.voiceprint_review_service import REGISTRY, CaptureConflictError
from app.presentation.tui.speaker_matches import SpeakerMatchCandidate
from app.presentation.tui.speaker_models import (
    ReviewSpeaker,
    SegmentScoreKey,
    SpeakerReviewSession,
    SpeakerSampleIdentityScore,
)
from app.presentation.tui.speaker_session import load_speaker_review_session
from app.presentation.tui.speaker_status import speaker_status
from app.speaker_labeling import SentenceReassignmentSpec
from app.speaker_match_status import MATCH_STATUS_CROSSTALK
from app.web.deps import (
    get_locks,
    get_settings,
    require_auth,
    resolve_web_project_ref,
)
from app.web.locks import LockRegistry, project_lock_key, store_lock_key
from app.web.schemas import (
    MatchPersonOut,
    PersonOut,
    ReviewOverviewOut,
    ReviewSpeakerOut,
    SaveSpeakerReviewIn,
    SaveSpeakerReviewOut,
    SpeakerMatchOut,
    SpeakerReviewOut,
    SpeakerSegmentOut,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/speakers", tags=["speakers"], dependencies=[Depends(require_auth)]
)


def _serialize_match(match: SpeakerMatchCandidate | None) -> SpeakerMatchOut | None:
    if match is None:
        return None
    return SpeakerMatchOut(
        best_name=match.best_name or match.name or None,
        best_score=match.best_score if match.best_score is not None else match.score,
        accepted=match.accepted,
        threshold=match.threshold,
        status=match.status,
        candidates=[
            MatchPersonOut(
                person_id=person.person_id,
                name=person.name,
                score=person.score,
                person_public_id=person.person_public_id,
            )
            for person in match.candidates
        ],
    )


def _serialize_speaker(
    speaker: ReviewSpeaker,
    identity_scores: dict[SegmentScoreKey, SpeakerSampleIdentityScore],
) -> ReviewSpeakerOut:
    duration_ms = sum(seg.end_time_ms - seg.begin_time_ms for seg in speaker.segments)
    segments: list[SpeakerSegmentOut] = []
    for seg in speaker.segments:
        score = identity_scores.get(
            (seg.sentence_id, seg.begin_time_ms, seg.end_time_ms)
        )
        segments.append(
            SpeakerSegmentOut(
                sentence_id=seg.sentence_id,
                begin_time_ms=seg.begin_time_ms,
                end_time_ms=seg.end_time_ms,
                text=seg.text,
                speaker_id=seg.speaker_id,
                score=score.assigned_score if score else None,
                score_status=score.status if score else None,
            )
        )
    return ReviewSpeakerOut(
        speaker_id=speaker.speaker_id,
        label=speaker.label,
        current_name=speaker.current_name,
        ignored=speaker.ignored,
        person_id=speaker.person_id,
        person_public_id=speaker.person_public_id,
        status=speaker_status(speaker),
        crosstalk=bool(
            speaker.match and speaker.match.status == MATCH_STATUS_CROSSTALK
        ),
        segment_count=speaker.segment_count,
        duration_ms=duration_ms,
        match=_serialize_match(speaker.match),
        segments=segments,
    )


def _serialize_session(session: SpeakerReviewSession) -> SpeakerReviewOut:
    overview = session.overview
    speakers = [
        _serialize_speaker(
            speaker, session.sample_identity_scores.get(speaker.speaker_id, {})
        )
        for speaker in session.speakers
    ]
    return SpeakerReviewOut(
        project_id=overview.project_id,
        project_dir=str(session.project_dir),
        overview=ReviewOverviewOut(
            project_id=overview.project_id,
            title=overview.title,
            project_status=overview.project_status,
            source_name=overview.source_name,
            duration_ms=overview.duration_ms,
            match_file_exists=overview.match_file_exists,
        ),
        speakers=speakers,
        people=[
            PersonOut(person_id=p.person_id, name=p.name, public_id=p.public_id)
            for p in session.people
        ],
        allow_correction=session.allow_correction,
    )


@router.get("/{project_ref}", response_model=SpeakerReviewOut)
def get_review(
    project_ref: str, settings: WebSettings = Depends(get_settings)
) -> SpeakerReviewOut:
    """Load the full speaker-review session for one project."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    session = load_speaker_review_session(
        project_dir, store_dir=settings.voiceprint_store_dir
    )
    return _serialize_session(session)


@router.post("/{project_ref}/save", response_model=SaveSpeakerReviewOut)
async def save_review(
    project_ref: str,
    payload: SaveSpeakerReviewIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> SaveSpeakerReviewOut:
    """Persist speaker names, person bindings, ignore flags, and reassignments."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    mapping = {int(k): v for k, v in payload.mapping.items()}
    person_mapping = {int(k): v for k, v in payload.person_mapping.items()}
    person_public_mapping = {
        int(k): v for k, v in payload.person_public_mapping.items()
    }
    specs = [
        SentenceReassignmentSpec(
            sentence_id=item.sentence_id,
            begin_time_ms=item.begin_time_ms,
            end_time_ms=item.end_time_ms,
            new_speaker_id=item.new_speaker_id,
            original_speaker_id=item.original_speaker_id,
        )
        for item in payload.reassignments
    ]

    # Reassignments touch the global voiceprint store (sample invalidation + rematch);
    # naming-only saves stay project-local, so only take the store lock when needed.
    keys = [project_lock_key(str(project_dir))]
    if specs:
        keys.append(store_lock_key("voiceprints"))

    def do_save():
        return save_speaker_review(
            project_dir,
            mapping=mapping,
            person_mapping=person_mapping,
            person_public_mapping=person_public_mapping,
            ignored_speaker_ids=payload.ignored_speaker_ids,
            reassignments=specs,
            store_dir=settings.voiceprint_store_dir,
        )

    # A reassignment write must join the same store-wide critical section as voiceprint
    # CRUD and capture runs (and be refused while a capture is pending), otherwise a later
    # capture rollback could silently undo the sample invalidation. Naming-only saves never
    # touch the global store, so they skip it.
    runner = (lambda: REGISTRY.run_store_write(do_save)) if specs else do_save

    loop = asyncio.get_running_loop()
    async with locks.acquire(*keys):
        # A pending capture's rollback would restore the pre-capture project.json /
        # speaker_matches.json, silently clobbering this save. Refuse under the project lock
        # (the capture run registers its txn under the same lock, so the check can't race
        # the registration). The reassignment path is already refused inside
        # run_store_write; this covers the naming-only path too.
        if REGISTRY.has_pending():
            raise CaptureConflictError(
                "A voiceprint capture is awaiting accept/rollback; resolve it before "
                "saving the speaker review."
            )
        result = await loop.run_in_executor(None, runner)

    reassignment = result.reassignment
    return SaveSpeakerReviewOut(
        mapping_path=str(result.mapping_path),
        transcript_path=str(result.transcript_path),
        srt_path=str(result.srt_path),
        reassigned_count=len(specs),
        deleted_sample_count=(len(reassignment.deleted_samples) if reassignment else 0),
        rematch_skipped_reason=(
            reassignment.rematch_skipped_reason if reassignment else None
        ),
    )
