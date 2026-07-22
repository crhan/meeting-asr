"""Speaker review: load the review session and persist a save decision.

Reads reuse the canonical TUI loader (``load_speaker_review_session``), which is pure
data loading. Saves go through the shared ``app.core.speaker_review_service`` so the web
and the CLI take the identical save path (including the voiceprint-sample invalidation +
rematch on reassignment). Writes run in the executor under per-project (and, when a
reassignment touches the global voiceprint store, per-store) locks.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.commands.project_correct import (
    accept_correction_for_review,
    load_speaker_mapping_for_correction,
    prepare_inline_corrections_for_review,
)
from app.core.speaker_review_service import save_speaker_review
from app.core.voiceprint_review_service import REGISTRY, CaptureConflictError
from app.correction_proposals import load_correction_proposal
from app.correction_types import CorrectionEditSummary
from app.lexicon_store import get_lexicon_db_path
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
from app.sentence_locator import format_sentence_ref
from app.speaker_match_status import MATCH_STATUS_CROSSTALK
from app.speaker_matching import match_project_speakers
from app.speaker_pipeline_params import (
    DEFAULT_MATCH_MAX_SECONDS,
    DEFAULT_MATCH_PADDING_SECONDS,
    DEFAULT_MATCH_SAMPLE_COUNT,
    DEFAULT_MATCH_THRESHOLD,
)
from app.project_manager import load_manifest, project_paths
from app.transcript_corrections import CorrectionEditOptions
from app.voiceprint_ids import valid_person_public_id
from app.voiceprint_people import get_voiceprint_person
from app.voiceprint_store import get_voiceprint_db_path
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
    InlineCorrectionIn,
    ReviewSpeakerOut,
    SaveSpeakerReviewIn,
    SaveSpeakerReviewOut,
    SpeakerRematchOut,
    SpeakerMatchOut,
    SpeakerReviewOut,
    SpeakerSegmentOut,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/speakers", tags=["speakers"], dependencies=[Depends(require_auth)]
)

_REVIEW_REVISION_FILES = (
    "project.json",
    "asr/sentences.json",
    "asr/sentences_corrected.json",
    "speakers/speaker_map.json",
    "speakers/speaker_person_map.json",
    "speakers/speaker_ignore.json",
    "speakers/speaker_matches.json",
)


def _review_revision(project_dir: Path) -> str:
    """Hash the persisted project state that a speaker-review save depends on."""
    digest = hashlib.sha256()
    root = project_dir.resolve()
    for rel_path in _REVIEW_REVISION_FILES:
        path = root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        if not path.exists():
            digest.update(b"missing")
            digest.update(b"\0")
            continue
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _require_current_revision(project_dir: Path, reviewed_revision: str) -> None:
    """Reject saves based on a stale review session."""
    current = _review_revision(project_dir)
    if current != reviewed_revision:
        raise HTTPException(
            status_code=409,
            detail=(
                "The speaker review changed since you loaded it. Reload before saving."
            ),
        )


def _validate_person_public_mapping(
    person_public_mapping: dict[int, str],
    *,
    store_dir: Path | None,
) -> dict[int, str]:
    """Return stripped public ids after checking shape and current store existence."""
    if not person_public_mapping:
        return {}
    db_path = get_voiceprint_db_path(store_dir)
    validated: dict[int, str] = {}
    for speaker_id, raw_public_id in sorted(person_public_mapping.items()):
        public_id = raw_public_id.strip()
        if not valid_person_public_id(public_id):
            raise ValueError(
                f"Invalid voiceprint person id for speaker {speaker_id}: "
                f"{raw_public_id!r}. Expected vpp-<16 hex>."
            )
        if get_voiceprint_person(public_id, db_path) is None:
            raise ValueError(
                f"Voiceprint person does not exist for speaker {speaker_id}: {public_id}"
            )
        validated[speaker_id] = public_id
    return validated


def _inline_correction_key(item: object) -> tuple[int | None, int | None, int, int]:
    """Return the stable sentence identity used by TUI/Web inline correction."""
    return (
        getattr(item, "sentence_id"),
        getattr(item, "speaker_id"),
        int(getattr(item, "begin_time_ms")),
        int(getattr(item, "end_time_ms")),
    )


def _accept_inline_corrections(
    project_dir: Path,
    correction_edits: list[InlineCorrectionIn],
    *,
    lexicon_db: Path,
) -> CorrectionEditSummary | None:
    """Accept only the concrete Web-edited sentences from an inline proposal."""
    if not correction_edits:
        return None
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    speaker_mapping = load_speaker_mapping_for_correction(paths.root)
    options = CorrectionEditOptions(
        open_editor=False,
        open_proposal=False,
        use_ai=False,
        category="web-inline",
        lexicon_db=lexicon_db,
    )
    summary = prepare_inline_corrections_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=correction_edits,
        options=options,
    )
    if summary.proposal_json_path is None:
        return summary
    proposal = load_correction_proposal(paths, summary.proposal_json_path)
    edited_keys = {_inline_correction_key(edit) for edit in correction_edits}
    selected_indices = tuple(
        index
        for index, change in enumerate(proposal.proposed_changes)
        if _inline_correction_key(change) in edited_keys
    )
    if not selected_indices:
        raise RuntimeError("Inline correction proposal did not include edited sentences.")
    return accept_correction_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        proposal_path=summary.proposal_json_path,
        lexicon_db=lexicon_db,
        selected_change_indices=selected_indices,
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
    project_id: str,
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
                sentence_ref=format_sentence_ref(project_id, seg.sentence_id),
                begin_time_ms=seg.begin_time_ms,
                end_time_ms=seg.end_time_ms,
                text=seg.text,
                speaker_id=seg.speaker_id,
                score=score.assigned_score if score else None,
                score_status=score.status if score else None,
                score_best_name=score.best_name if score else None,
                score_best_score=score.best_score if score else None,
                score_best_other_name=score.best_other_name if score else None,
                score_best_other_score=score.best_other_score if score else None,
                score_margin=score.margin_score if score else None,
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


def _serialize_session(
    session: SpeakerReviewSession, *, review_revision: str
) -> SpeakerReviewOut:
    overview = session.overview
    speakers = [
        _serialize_speaker(
            speaker,
            session.sample_identity_scores.get(speaker.speaker_id, {}),
            overview.project_id,
        )
        for speaker in session.speakers
    ]
    return SpeakerReviewOut(
        project_id=overview.project_id,
        project_dir=str(session.project_dir),
        review_revision=review_revision,
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
async def get_review(
    project_ref: str,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> SpeakerReviewOut:
    """Load the full speaker-review session for one project."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    loop = asyncio.get_running_loop()
    async with locks.acquire(project_lock_key(str(project_dir))):
        session = await loop.run_in_executor(
            None,
            lambda: load_speaker_review_session(
                project_dir,
                store_dir=settings.voiceprint_store_dir,
                allow_correction=True,
            ),
        )
        return _serialize_session(
            session, review_revision=_review_revision(project_dir)
        )


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
    new_person_names = {int(k): v for k, v in payload.new_person_names.items()}
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
    deleted_speaker_ids = sorted(
        {int(speaker_id) for speaker_id in payload.deleted_speaker_ids}
    )
    correction_edits = list(payload.correction_edits)
    lexicon_db = get_lexicon_db_path(settings.store_dir)

    # Reassignments touch the global voiceprint store (sample invalidation + rematch);
    # naming-only saves stay project-local, so only take the store lock when needed.
    keys = [project_lock_key(str(project_dir))]
    if specs or person_public_mapping or new_person_names:
        keys.append(store_lock_key("voiceprints"))
    if correction_edits:
        keys.append(store_lock_key("lexicon"))

    def do_save():
        result = save_speaker_review(
            project_dir,
            mapping=mapping,
            person_mapping=person_mapping,
            person_public_mapping=person_public_mapping,
            new_person_names=new_person_names,
            ignored_speaker_ids=payload.ignored_speaker_ids,
            reassignments=specs,
            deleted_speaker_ids=deleted_speaker_ids,
            store_dir=settings.voiceprint_store_dir,
        )
        correction_summary = _accept_inline_corrections(
            project_dir,
            correction_edits,
            lexicon_db=lexicon_db,
        )
        return result, correction_summary

    # A reassignment write must join the same store-wide critical section as voiceprint
    # CRUD and capture runs (and be refused while a capture is pending), otherwise a later
    # capture rollback could silently undo the sample invalidation. Naming-only saves never
    # touch the global store, so they skip it.
    runner = (
        (lambda: REGISTRY.run_store_write(do_save))
        if specs or new_person_names
        else do_save
    )

    loop = asyncio.get_running_loop()
    async with locks.acquire(*keys):
        _require_current_revision(project_dir, payload.review_revision)
        person_public_mapping = _validate_person_public_mapping(
            person_public_mapping, store_dir=settings.voiceprint_store_dir
        )
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
        result, correction_summary = await loop.run_in_executor(None, runner)

    reassignment = result.reassignment
    deletion = result.deletion
    return SaveSpeakerReviewOut(
        mapping_path=str(result.mapping_path),
        transcript_path=str(result.transcript_path),
        srt_path=str(result.srt_path),
        reassigned_count=len(specs),
        created_person_count=result.created_person_count,
        deleted_speaker_count=len(deleted_speaker_ids),
        deleted_sentence_count=(deletion.deleted_sentence_count if deletion else 0),
        deleted_sample_count=(len(reassignment.deleted_samples) if reassignment else 0),
        corrected_count=(correction_summary.change_count if correction_summary else 0),
        corrected_transcript_path=(
            str(correction_summary.corrected_named_transcript_path)
            if correction_summary
            and correction_summary.corrected_named_transcript_path is not None
            else None
        ),
        rematch_skipped_reason=(
            reassignment.rematch_skipped_reason if reassignment else None
        ),
    )


@router.post("/{project_ref}/rematch", response_model=SpeakerRematchOut)
async def rematch_review(
    project_ref: str,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> SpeakerRematchOut:
    """Refresh project speaker matches against the current voiceprint library."""
    project_dir = resolve_web_project_ref(project_ref, settings)

    def do_rematch():
        return match_project_speakers(
            project_dir,
            store_dir=settings.voiceprint_store_dir,
            provider=None,
            model=None,
            threshold=DEFAULT_MATCH_THRESHOLD,
            sample_count=DEFAULT_MATCH_SAMPLE_COUNT,
            max_seconds=DEFAULT_MATCH_MAX_SECONDS,
            padding_seconds=DEFAULT_MATCH_PADDING_SECONDS,
            crosstalk_params=None,
            progress=None,
        )

    loop = asyncio.get_running_loop()
    async with locks.acquire(
        project_lock_key(str(project_dir)), store_lock_key("voiceprints")
    ):
        if REGISTRY.has_pending():
            raise CaptureConflictError(
                "A voiceprint capture is awaiting accept/rollback; resolve it before "
                "refreshing speaker matches."
            )
        summary = await loop.run_in_executor(
            None, lambda: REGISTRY.run_store_write(do_rematch)
        )
    below = sum(1 for item in summary.matches if not item.accepted)
    matched = len(summary.matches) - below
    return SpeakerRematchOut(
        matched_count=matched,
        below_threshold_count=below,
        total_count=len(summary.matches),
    )
