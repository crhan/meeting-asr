"""Focused voiceprint learning workflow for confirmed project speakers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.project_manager import apply_project_speakers
from app.speaker_match_status import best_candidate_score, voiceprint_match_status
from app.speaker_matching import (
    SpeakerMatch,
    SpeakerMatchSummary,
    match_project_speakers,
    preview_project_speaker_matches,
)
from app.voiceprint_embedding import VoiceprintEmbedSummary, embed_voiceprint_samples
from app.voiceprints import (
    VoiceprintCaptureDecision,
    VoiceprintCaptureSummary,
    capture_voiceprints,
)


@dataclass(frozen=True, slots=True)
class SpeakerLearningMatch:
    """Compact before/after match state for one project speaker."""

    status: str
    best_name: str | None
    best_score: float | None
    best_person_public_id: str | None
    accepted_person_public_id: str | None


@dataclass(frozen=True, slots=True)
class SpeakerLearningResult:
    """Learning result for one explicitly selected project speaker."""

    speaker_id: int
    canonical_name: str
    person_public_id: str | None
    existing_sample_count: int
    capture_decision: str
    capture_reason: str
    captured_sample_ids: tuple[str, ...]
    embedding_generated: bool
    before: SpeakerLearningMatch | None
    after: SpeakerLearningMatch | None
    score_delta: float | None
    threshold: float
    status: str
    reason: str
    applied: bool = False


@dataclass(frozen=True, slots=True)
class SpeakerLearningSummary:
    """End-to-end focused speaker learning summary."""

    project_id: str
    status: str
    dry_run: bool
    threshold: float
    capture: VoiceprintCaptureSummary
    embedding: VoiceprintEmbedSummary | None
    matches: SpeakerMatchSummary | None
    speakers: list[SpeakerLearningResult]

    @property
    def needs_review(self) -> bool:
        """Return whether automation should stop for human review."""
        return self.status in {"needs_review", "failed", "partial_failure"}


def learn_project_speakers(
    project_dir: Path,
    *,
    speaker_ids: set[int],
    store_dir: Path | None,
    model: str | None,
    threshold: float,
    capture_sample_count: int,
    match_sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    min_samples: int,
    only_needed: bool,
    apply_changes: bool,
    embed: bool,
    rematch: bool,
    progress: CliProgressReporter | None = None,
) -> SpeakerLearningSummary:
    """Plan or run capture -> focused embed -> rematch for selected speakers."""
    if not speaker_ids:
        raise ValueError("At least one --speaker-id is required.")
    if (embed or rematch) and not apply_changes:
        raise ValueError("--embed and --rematch require --apply.")

    emit_progress(progress, "Planning selected speaker voiceprints", stage="plan")
    planned = capture_voiceprints(
        project_dir,
        sample_count=capture_sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        store_dir=store_dir,
        dry_run=True,
        speaker_ids=speaker_ids,
        only_needed=only_needed,
        min_samples=min_samples,
    )
    if not apply_changes:
        results = _learning_results(
            planned.decisions,
            before=None,
            after=None,
            threshold=threshold,
            embedded=False,
            dry_run=True,
            rematched=False,
        )
        return SpeakerLearningSummary(
            planned.project_id,
            "planned",
            True,
            threshold,
            planned,
            None,
            None,
            results,
        )

    before = None
    if rematch:
        emit_progress(progress, "Reading pre-learning match scores", stage="before")
        before = preview_project_speaker_matches(
            project_dir,
            store_dir=store_dir,
            provider=None,
            model=model,
            threshold=threshold,
            sample_count=match_sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            progress=None,
        )

    emit_progress(progress, "Capturing selected speaker voiceprints", stage="capture")
    capture = capture_voiceprints(
        project_dir,
        sample_count=capture_sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        store_dir=store_dir,
        dry_run=False,
        speaker_ids=speaker_ids,
        only_needed=only_needed,
        min_samples=min_samples,
        progress=None,
    )
    captured_row_ids = {
        sample.sample_id
        for decision in capture.decisions
        for sample in decision.samples
        if decision.decision == "captured"
    }
    embedding = None
    if embed and captured_row_ids:
        emit_progress(progress, "Embedding newly captured samples", stage="embed")
        embedding = embed_voiceprint_samples(
            store_dir=store_dir or capture.store_dir,
            provider=None,
            model=model,
            rebuild=True,
            sample_ids=captured_row_ids,
            progress=None,
        )
        capture = _mark_capture_samples_embedded(capture, captured_row_ids)

    after = None
    if rematch:
        emit_progress(progress, "Re-matching project speakers", stage="rematch")
        after = match_project_speakers(
            project_dir,
            store_dir=store_dir,
            provider=None,
            model=embedding.model if embedding is not None else model,
            threshold=threshold,
            sample_count=match_sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            progress=None,
        )

    results = _learning_results(
        capture.decisions,
        before=before,
        after=after,
        threshold=threshold,
        embedded=embedding is not None,
        dry_run=False,
        rematched=rematch,
    )
    if rematch and after is not None:
        results = _apply_safe_learning_matches(project_dir, results, after)
    status = _learning_status(results, rematched=rematch)
    emit_progress(progress, "Speaker learning complete", stage="complete")
    return SpeakerLearningSummary(
        capture.project_id,
        status,
        False,
        threshold,
        capture,
        embedding,
        after,
        results,
    )


def speaker_learning_payload(summary: SpeakerLearningSummary) -> dict[str, object]:
    """Return stable JSON output for ``project speakers learn``."""
    embedding = summary.embedding
    matches = summary.matches
    return {
        "project_id": summary.project_id,
        "status": summary.status,
        "dry_run": summary.dry_run,
        "threshold": summary.threshold,
        "capture": {
            "stored_sample_count": sum(
                len(item.samples)
                for item in summary.capture.decisions
                if item.decision == "captured"
            ),
            "failed_count": summary.capture.failed_count,
            "database": summary.capture.db_path,
        },
        "embedding": (
            None
            if embedding is None
            else {
                "provider": embedding.provider,
                "model": embedding.model,
                "embedded_count": embedding.embedded_count,
                "skipped_count": embedding.skipped_count,
            }
        ),
        "rematch": (
            None
            if matches is None
            else {
                "path": matches.match_path,
                "provider": matches.provider,
                "model": matches.model,
            }
        ),
        "speakers": [
            _speaker_learning_result_payload(item) for item in summary.speakers
        ],
    }


def _speaker_learning_result_payload(
    item: SpeakerLearningResult,
) -> dict[str, object]:
    """Serialize one focused learning result."""
    return {
        "speaker_id": item.speaker_id,
        "canonical_name": item.canonical_name,
        "person_id": item.person_public_id,
        "person_public_id": item.person_public_id,
        "existing_sample_count": item.existing_sample_count,
        "capture": {
            "decision": item.capture_decision,
            "reason": item.capture_reason,
            "sample_ids": list(item.captured_sample_ids),
        },
        "embedding_generated": item.embedding_generated,
        "match": {
            "before": _learning_match_payload(item.before),
            "after": _learning_match_payload(item.after),
            "score_delta": item.score_delta,
            "threshold": item.threshold,
        },
        "status": item.status,
        "reason": item.reason,
        "applied": item.applied,
    }


def _learning_match_payload(
    match: SpeakerLearningMatch | None,
) -> dict[str, object] | None:
    """Serialize one optional match snapshot."""
    if match is None:
        return None
    return {
        "status": match.status,
        "best_name": match.best_name,
        "best_score": match.best_score,
        "best_person_public_id": match.best_person_public_id,
        "accepted_person_public_id": match.accepted_person_public_id,
    }


def _mark_capture_samples_embedded(
    capture: VoiceprintCaptureSummary, sample_ids: set[int]
) -> VoiceprintCaptureSummary:
    """Mark focused samples embedded in the workflow result payload."""
    decisions = [
        replace(
            decision,
            samples=[
                replace(sample, embedded=True)
                if sample.sample_id in sample_ids
                else sample
                for sample in decision.samples
            ],
        )
        for decision in capture.decisions
    ]
    return replace(capture, decisions=decisions)


def _learning_results(
    decisions: list[VoiceprintCaptureDecision],
    *,
    before: SpeakerMatchSummary | None,
    after: SpeakerMatchSummary | None,
    threshold: float,
    embedded: bool,
    dry_run: bool,
    rematched: bool,
) -> list[SpeakerLearningResult]:
    """Build per-speaker outcomes with strict threshold and identity checks."""
    before_by_id = _matches_by_speaker(before)
    after_by_id = _matches_by_speaker(after)
    results: list[SpeakerLearningResult] = []
    for decision in decisions:
        before_match = _learning_match(before_by_id.get(decision.speaker_id))
        after_match = _learning_match(after_by_id.get(decision.speaker_id))
        status, reason = _learning_decision_status(
            decision,
            after_match,
            threshold=threshold,
            dry_run=dry_run,
            rematched=rematched,
        )
        score_delta = _score_delta(before_match, after_match)
        results.append(
            SpeakerLearningResult(
                speaker_id=decision.speaker_id,
                canonical_name=decision.name,
                person_public_id=decision.person_public_id,
                existing_sample_count=decision.existing_sample_count,
                capture_decision=decision.decision,
                capture_reason=decision.reason,
                captured_sample_ids=tuple(
                    sample.public_id for sample in decision.samples
                ),
                embedding_generated=embedded and bool(decision.samples),
                before=before_match,
                after=after_match,
                score_delta=score_delta,
                threshold=threshold,
                status=status,
                reason=reason,
            )
        )
    return results


def _learning_decision_status(
    decision: VoiceprintCaptureDecision,
    after: SpeakerLearningMatch | None,
    *,
    threshold: float,
    dry_run: bool,
    rematched: bool,
) -> tuple[str, str]:
    """Enforce exact identity plus the explicit threshold for automation."""
    if decision.decision == "failed":
        return "failed", decision.error or decision.reason
    if dry_run:
        return "planned", decision.reason
    if not rematched:
        return "completed", decision.reason
    expected = decision.person_public_id
    if expected is None:
        return "needs_review", "missing_person_id"
    if after is None:
        return "needs_review", "missing_match_result"
    if after.best_person_public_id != expected:
        return "needs_review", "identity_mismatch"
    if after.best_score is None or after.best_score < threshold:
        return "needs_review", "below_threshold"
    if after.accepted_person_public_id != expected:
        return "needs_review", "not_accepted"
    return "matched", "matched_expected_person"


def _apply_safe_learning_matches(
    project_dir: Path,
    results: list[SpeakerLearningResult],
    matches: SpeakerMatchSummary,
) -> list[SpeakerLearningResult]:
    """Apply only selected results that passed strict identity and score checks."""
    match_by_id = {item.speaker_id: item for item in matches.matches}
    safe = [item for item in results if item.status == "matched"]
    mappings: dict[int, str] = {}
    person_mapping: dict[int, int] = {}
    person_public_mapping: dict[int, str] = {}
    for result in safe:
        match = match_by_id[result.speaker_id]
        # The capture decision resolved the registry's canonical person name.
        # Keep that authority instead of trusting an incidental display string
        # from the matcher payload, even though the stable public id matched.
        mappings[result.speaker_id] = result.canonical_name
        if match.accepted_person_id is not None:
            person_mapping[result.speaker_id] = match.accepted_person_id
        if match.accepted_person_public_id:
            person_public_mapping[result.speaker_id] = match.accepted_person_public_id
    if mappings:
        apply_project_speakers(
            project_dir,
            mappings,
            person_mapping=person_mapping,
            person_public_mapping=person_public_mapping,
        )
    applied_ids = set(mappings)
    return [replace(item, applied=item.speaker_id in applied_ids) for item in results]


def _matches_by_speaker(
    summary: SpeakerMatchSummary | None,
) -> dict[int, SpeakerMatch]:
    """Index an optional match summary by project speaker id."""
    if summary is None:
        return {}
    return {item.speaker_id: item for item in summary.matches}


def _learning_match(match: SpeakerMatch | None) -> SpeakerLearningMatch | None:
    """Build a compact match snapshot."""
    if match is None:
        return None
    return SpeakerLearningMatch(
        status=voiceprint_match_status(match),
        best_name=match.best_name or match.name,
        best_score=best_candidate_score(match),
        best_person_public_id=match.best_person_public_id,
        accepted_person_public_id=match.accepted_person_public_id,
    )


def _score_delta(
    before: SpeakerLearningMatch | None, after: SpeakerLearningMatch | None
) -> float | None:
    """Return after-minus-before when both scores exist."""
    if before is None or after is None:
        return None
    if before.best_score is None or after.best_score is None:
        return None
    return after.best_score - before.best_score


def _learning_status(results: list[SpeakerLearningResult], *, rematched: bool) -> str:
    """Aggregate per-speaker outcomes into an automation status."""
    failed = sum(1 for item in results if item.status == "failed")
    if failed:
        return "failed" if failed == len(results) else "partial_failure"
    if rematched and any(item.status == "needs_review" for item in results):
        return "needs_review"
    if rematched:
        return "matched"
    return "completed"
