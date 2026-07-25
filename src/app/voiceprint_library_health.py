"""Voiceprint library health: can this library actually identify people?

:mod:`app.voiceprint_quality` answers "are a person's samples consistent with
each other" — a within-cluster question. That check is blind to the failure
modes that silently remove a person from matching altogether:

- a person whose samples were never embedded for the *active* model, so they
  are in the library but not in the matching pool, and
- a person whose samples are all quarantined, which the per-sample view
  reports as "0 suspicious" because there is nothing left to find fault with.

Both render as healthy green in a sample-consistency view while the person is
simply never matched. This module answers the prior question — *availability* —
and joins it with the store-wide threshold calibration into one prioritized,
actionable issue list.

Read-only; nothing here mutates the store.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.voiceprint_audio import resolve_voiceprint_sample_source
from app.voiceprint_calibration import (
    VoiceprintCalibrationReport,
    calibrate_voiceprint_thresholds,
)
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_models import VoiceprintSampleRow, VoiceprintSpeakerRow
from app.voiceprint_sample_overlap import overlapped_sample_ids
from app.voiceprint_quality import (
    DEFAULT_MIN_CLUSTER_SIZE,
    VOICEPRINT_MATCHING_SAMPLE_STATUSES,
)
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
    list_voiceprint_speakers,
)

AVAILABILITY_OK = "ok"
AVAILABILITY_FRAGILE = "fragile"
AVAILABILITY_UNUSABLE = "unusable"

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# A person under this much matching audio is characterized by a handful of
# short clips; the probe centroid then swings with whatever was said.
MIN_HEALTHY_MATCHING_SECONDS = 20.0
# One recording session means one microphone, one room and one mood; scores
# hold up there and drop on the next meeting.
MIN_HEALTHY_PROJECT_COUNT = 2


@dataclass(frozen=True, slots=True)
class PersonHealth:
    """Availability facts for one library person under the active model."""

    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    total_sample_count: int
    enabled_sample_count: int
    matching_sample_count: int
    missing_embedding_count: int
    # Of the missing ones, how many cannot be embedded at all because their
    # clip audio is gone. Backfilling can never fix these.
    missing_clip_count: int
    matching_seconds: float
    project_count: int
    availability: str

    @property
    def embeddable_count(self) -> int:
        """Return missing embeddings a backfill could actually produce."""
        return self.missing_embedding_count - self.missing_clip_count

    @property
    def usable(self) -> bool:
        """Return whether this person can be matched at all."""
        return self.matching_sample_count > 0


@dataclass(frozen=True, slots=True)
class LibraryIssue:
    """One actionable library problem.

    ``title``/``detail`` are English prose for CLI and API consumers. The web
    UI is bilingual, so it re-renders both from ``kind`` plus the numbers in
    ``context`` rather than translating sentences — that keeps the numbers
    single-sourced here and the wording where the locale lives.
    """

    kind: str
    severity: str
    title: str
    detail: str
    action: str
    person_public_id: str | None = None
    person_name: str | None = None
    context: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LibraryHealthReport:
    """Availability and threshold health for the whole voiceprint library."""

    db_path: Path
    provider: str
    model: str
    people: tuple[PersonHealth, ...]
    issues: tuple[LibraryIssue, ...]
    calibration: VoiceprintCalibrationReport | None

    @property
    def person_count(self) -> int:
        """Return how many people the library holds."""
        return len(self.people)

    @property
    def usable_person_count(self) -> int:
        """Return how many people can actually be matched."""
        return sum(1 for person in self.people if person.usable)

    @property
    def matching_sample_count(self) -> int:
        """Return how many samples participate in matching."""
        return sum(person.matching_sample_count for person in self.people)

    @property
    def matching_seconds(self) -> float:
        """Return total matching audio in the library."""
        return sum(person.matching_seconds for person in self.people)

    @property
    def critical_count(self) -> int:
        """Return the number of critical issues."""
        return sum(1 for item in self.issues if item.severity == SEVERITY_CRITICAL)

    @property
    def warning_count(self) -> int:
        """Return the number of warning issues."""
        return sum(1 for item in self.issues if item.severity == SEVERITY_WARNING)


def analyze_library_health(
    *,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    projects_dir: Path | None = None,
) -> LibraryHealthReport:
    """
    Report which library people are matchable and what blocks the rest.

    Args:
        store_dir: Optional voiceprint store directory.
        provider: Optional embedding provider override.
        model: Optional embedding model key override.
        projects_dir: Optional projects parent directory. Supplied, each
            sample is checked against its source transcript for speaker
            overlap; omitted, that check is simply absent from the report
            rather than guessed at.

    Returns:
        Library health report with a prioritized issue list.
    """
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_all_voiceprint_samples(db_path)
    embedded_ids = list_embedded_sample_ids(resolved_model, db_path)
    people = _people_health(
        samples, embedded_ids, store_dir, list_voiceprint_speakers(db_path)
    )
    calibration = _calibration(store_dir, resolved_provider, resolved_model)
    overlapped = _overlapped_samples(samples, embedded_ids, projects_dir)
    issues = _issues(people, calibration, overlapped)
    return LibraryHealthReport(
        db_path=db_path,
        provider=resolved_provider,
        model=resolved_model,
        people=people,
        issues=issues,
        calibration=calibration,
    )


def person_availability(
    person_public_id: str,
    *,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> PersonHealth | None:
    """
    Return availability facts for one person, without library-wide calibration.

    Callers that only need "how short is this person" (sample sourcing, per-person
    views) should not pay for the pairwise genuine/impostor sweep that threshold
    calibration runs over the whole store.

    Args:
        person_public_id: Library person public id.
        store_dir: Optional voiceprint store directory.
        provider: Optional embedding provider override.
        model: Optional embedding model key override.

    Returns:
        Availability facts, or None when the person has no samples at all.
    """
    _, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    db_path = get_voiceprint_db_path(store_dir)
    rows = [
        row
        for row in list_all_voiceprint_samples(db_path)
        if row.speaker_public_id == person_public_id
    ]
    if not rows:
        return None
    embedded_ids = list_embedded_sample_ids(resolved_model, db_path)
    return _person_health(rows, embedded_ids, store_dir)


def _calibration(
    store_dir: Path | None, provider: str, model: str
) -> VoiceprintCalibrationReport | None:
    """Return threshold calibration, or None when the store cannot supply it."""
    try:
        return calibrate_voiceprint_thresholds(
            store_dir=store_dir, provider=provider, model=model
        )
    except Exception:  # noqa: BLE001 - health must render without calibration
        return None


def _people_health(
    samples: list[VoiceprintSampleRow],
    embedded_ids: set[int],
    store_dir: Path | None,
    speakers: list[VoiceprintSpeakerRow],
) -> tuple[PersonHealth, ...]:
    """
    Build per-person availability facts, including people with no samples.

    Grouping by sample would drop anyone created but never captured -- and a
    person with zero samples is the most unmatchable person in the library, so
    omitting them lets an "everyone is usable" report coexist with a name in
    the library that never matches anything.
    """
    grouped: dict[int, list[VoiceprintSampleRow]] = defaultdict(list)
    for sample in samples:
        grouped[sample.speaker_id].append(sample)
    people = [
        _person_health(rows, embedded_ids, store_dir) for rows in grouped.values()
    ]
    people.extend(
        _empty_person_health(row) for row in speakers if row.speaker_id not in grouped
    )
    return tuple(sorted(people, key=_person_sort_key))


def _empty_person_health(row: VoiceprintSpeakerRow) -> PersonHealth:
    """Build availability facts for a person who has no samples at all."""
    return PersonHealth(
        speaker_id=row.speaker_id,
        speaker_public_id=row.public_id,
        speaker_name=row.name,
        total_sample_count=0,
        enabled_sample_count=0,
        matching_sample_count=0,
        missing_embedding_count=0,
        missing_clip_count=0,
        matching_seconds=0.0,
        project_count=0,
        availability=AVAILABILITY_UNUSABLE,
    )


def _person_health(
    rows: list[VoiceprintSampleRow],
    embedded_ids: set[int],
    store_dir: Path | None,
) -> PersonHealth:
    """Build availability facts for one person."""
    first = rows[0]
    enabled = [
        row for row in rows if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
    ]
    matching = [row for row in enabled if row.sample_id in embedded_ids]
    missing = [row for row in enabled if row.sample_id not in embedded_ids]
    # Only stat the samples a backfill would touch: an issue that offers
    # "backfill embeddings" must not be raised for clips that no longer exist,
    # or the button runs, reports success, and changes nothing.
    missing_clips = [row for row in missing if not _clip_readable(row, store_dir)]
    seconds = sum(
        max(0, row.source_end_time_ms - row.source_begin_time_ms) / 1000.0
        for row in matching
    )
    return PersonHealth(
        speaker_id=first.speaker_id,
        speaker_public_id=first.speaker_public_id,
        speaker_name=first.speaker_name,
        total_sample_count=len(rows),
        enabled_sample_count=len(enabled),
        matching_sample_count=len(matching),
        missing_embedding_count=len(missing),
        missing_clip_count=len(missing_clips),
        matching_seconds=round(seconds, 1),
        project_count=len({row.project_id for row in matching}),
        availability=_availability(len(matching)),
    )


def _clip_readable(row: VoiceprintSampleRow, store_dir: Path | None) -> bool:
    """Return whether a sample's clip audio can still be read."""
    try:
        return resolve_voiceprint_sample_source(row, store_dir=store_dir).exists()
    except OSError:
        return False


def _availability(matching_sample_count: int) -> str:
    """Classify a person by how many samples reach the matching pool."""
    if matching_sample_count == 0:
        return AVAILABILITY_UNUSABLE
    if matching_sample_count < DEFAULT_MIN_CLUSTER_SIZE:
        return AVAILABILITY_FRAGILE
    return AVAILABILITY_OK


def _issues(
    people: tuple[PersonHealth, ...],
    calibration: VoiceprintCalibrationReport | None,
    overlapped: dict[str, int],
) -> tuple[LibraryIssue, ...]:
    """Build the prioritized issue list."""
    issues: list[LibraryIssue] = []
    issues.extend(_threshold_issues(calibration))
    for person in people:
        issues.extend(_person_issues(person))
        issues.extend(
            _overlap_issues(person, overlapped.get(person.speaker_public_id, 0))
        )
    return tuple(sorted(issues, key=_issue_sort_key))


def _overlapped_samples(
    samples: list[VoiceprintSampleRow],
    embedded_ids: set[int],
    projects_dir: Path | None,
) -> dict[str, int]:
    """
    Count each person's matching samples taken from a two-people-at-once stretch.

    Only samples that actually reach the matching pool are counted: a
    quarantined or unembedded sample cannot drag anyone's centroid, so
    reporting it here would be an issue with no consequence.

    Args:
        samples: Every stored sample row.
        embedded_ids: Sample ids with a vector for the active model.
        projects_dir: Projects parent directory, or None to skip the check.

    Returns:
        Person public id to overlapped matching-sample count. Empty when no
        projects directory was supplied, so "not checked" never renders as
        "checked and clean".
    """
    in_pool = [
        row
        for row in samples
        if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
        and row.sample_id in embedded_ids
    ]
    flagged = overlapped_sample_ids(in_pool, projects_dir)
    if not flagged:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for row in in_pool:
        if row.sample_id in flagged:
            counts[row.speaker_public_id] += 1
    return dict(counts)


def _overlap_issues(person: PersonHealth, overlapped: int) -> list[LibraryIssue]:
    """Report samples recorded while someone else was talking."""
    if overlapped <= 0:
        return []
    return [
        _person_issue(
            person,
            kind="overlapped-samples",
            severity=SEVERITY_WARNING,
            title=(
                f"{person.speaker_name} has {overlapped} sample(s) recorded "
                "over another speaker"
            ),
            detail=(
                "Another speaker holds the floor within half a second of these "
                "samples, so the reference audio is a mixture rather than one "
                "voice. That drags this person's centroid toward whoever they "
                "were talking to, which is exactly the pair hardest to tell "
                "apart. Replace them with samples from a stretch where they "
                "speak uninterrupted."
            ),
            action="review-samples",
            context={
                "name": person.speaker_name,
                "overlapped_count": overlapped,
                "matching_sample_count": person.matching_sample_count,
            },
        )
    ]


def _person_issues(person: PersonHealth) -> list[LibraryIssue]:
    """Build issues for one person, most blocking first."""
    issues: list[LibraryIssue] = []
    facts: dict[str, float | int | str] = {
        "name": person.speaker_name,
        "total_sample_count": person.total_sample_count,
        "enabled_sample_count": person.enabled_sample_count,
        "matching_sample_count": person.matching_sample_count,
        "missing_embedding_count": person.missing_embedding_count,
        "missing_clip_count": person.missing_clip_count,
        "embeddable_count": person.embeddable_count,
        "matching_seconds": person.matching_seconds,
        "project_count": person.project_count,
        "min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
        "min_healthy_seconds": MIN_HEALTHY_MATCHING_SECONDS,
    }
    if person.total_sample_count == 0:
        issues.append(
            _person_issue(
                person,
                kind="no-samples",
                severity=SEVERITY_CRITICAL,
                title=f"{person.speaker_name} has no voiceprint samples at all",
                detail=(
                    "This person exists in the library but nothing was ever "
                    "captured for them, so they can never be matched. Capture "
                    "samples from a project where they speak, or delete the "
                    "entry."
                ),
                action="capture",
                context=facts,
            )
        )
    elif person.enabled_sample_count == 0:
        issues.append(
            _person_issue(
                person,
                kind="no-enabled-samples",
                severity=SEVERITY_CRITICAL,
                title=f"{person.speaker_name} has no samples enabled for matching",
                detail=(
                    f"All {person.total_sample_count} sample(s) are quarantined, "
                    "rejected or invalidated, so this person is never matched. "
                    "Sample-consistency checks report nothing wrong because there "
                    "is nothing left to check."
                ),
                action="review-samples",
                context=facts,
            )
        )
    elif person.embeddable_count > 0 and person.matching_sample_count == 0:
        issues.append(
            _person_issue(
                person,
                kind="missing-embeddings",
                severity=SEVERITY_CRITICAL,
                title=f"{person.speaker_name} has no embeddings for the active model",
                detail=(
                    f"{person.embeddable_count} enabled sample(s) have readable "
                    "audio but no embedding for this model, so this person is "
                    "absent from the matching pool entirely. This usually means "
                    "the library was not re-embedded after switching provider."
                ),
                action="embed",
                context=facts,
            )
        )
    elif person.embeddable_count > 0:
        issues.append(
            _person_issue(
                person,
                kind="partial-embeddings",
                severity=SEVERITY_WARNING,
                title=f"{person.speaker_name} is missing some embeddings",
                detail=(
                    f"{person.embeddable_count} of "
                    f"{person.enabled_sample_count} enabled sample(s) are not "
                    "embedded for the active model and do not contribute to "
                    "matching."
                ),
                action="embed",
                context=facts,
            )
        )
    if person.missing_clip_count > 0:
        # Deliberately NOT action="embed": a backfill cannot read audio that is
        # gone, so offering it would run, report success and change nothing.
        issues.append(
            _person_issue(
                person,
                kind="missing-clips",
                severity=(
                    SEVERITY_CRITICAL
                    if person.matching_sample_count == 0
                    else SEVERITY_WARNING
                ),
                title=f"{person.speaker_name} has "
                f"{person.missing_clip_count} sample(s) whose audio is gone",
                detail=(
                    "The registry rows exist but their clip files cannot be read "
                    "from the active store, so these samples can never be "
                    "embedded. Delete them, or re-capture this person from a "
                    "project. Backfilling embeddings will not help."
                ),
                action="review-samples",
                context=facts,
            )
        )
    if person.availability == AVAILABILITY_FRAGILE:
        issues.append(
            _person_issue(
                person,
                kind="fragile-cluster",
                severity=SEVERITY_WARNING,
                title=f"{person.speaker_name} has only "
                f"{person.matching_sample_count} matching sample(s)",
                detail=(
                    f"Below {DEFAULT_MIN_CLUSTER_SIZE} samples the centroid is one "
                    "recording's quirks rather than a voice, and consistency "
                    "checks cannot flag an outlier because every sample defines "
                    "the centroid it is compared against."
                ),
                action="capture",
                context=facts,
            )
        )
    if person.usable and person.matching_seconds < MIN_HEALTHY_MATCHING_SECONDS:
        issues.append(
            _person_issue(
                person,
                kind="short-audio",
                severity=SEVERITY_WARNING,
                title=f"{person.speaker_name} has only "
                f"{person.matching_seconds:.0f}s of matching audio",
                detail=(
                    "Total voiced duration, not sample count, decides how stable "
                    f"an embedding is. Aim for at least "
                    f"{MIN_HEALTHY_MATCHING_SECONDS:.0f}s across several "
                    "utterances."
                ),
                action="capture",
                context=facts,
            )
        )
    if person.usable and person.project_count < MIN_HEALTHY_PROJECT_COUNT:
        issues.append(
            _person_issue(
                person,
                kind="single-source",
                severity=SEVERITY_INFO,
                title=f"{person.speaker_name} is characterized by one recording",
                detail=(
                    "All matching samples come from a single project, so the "
                    "voiceprint also encodes that room, microphone and mood. "
                    "Add samples from another meeting to generalize."
                ),
                action="capture",
                context=facts,
            )
        )
    return issues


def _threshold_issues(
    calibration: VoiceprintCalibrationReport | None,
) -> list[LibraryIssue]:
    """Build issues about the acceptance threshold's fit to this library."""
    if calibration is None:
        return []
    current = calibration.current_cost
    suggested = calibration.suggested_cost
    if current is None or suggested is None:
        return []
    if calibration.suggested_threshold is None:
        return []
    facts: dict[str, float | int | str] = {
        "current_threshold": current.threshold,
        "current_false_reject_count": current.false_reject_count,
        "current_false_reject_rate": current.false_reject_rate,
        "current_false_accept_count": current.false_accept_count,
        "current_false_accept_rate": current.false_accept_rate,
        "suggested_threshold": calibration.suggested_threshold,
        "suggested_false_reject_count": suggested.false_reject_count,
        "suggested_false_accept_count": suggested.false_accept_count,
        "genuine_count": len(calibration.genuine_scores),
        "impostor_count": len(calibration.impostor_scores),
    }
    issues: list[LibraryIssue] = []
    if current.false_accept_count > 0:
        issues.append(
            LibraryIssue(
                kind="threshold-too-low",
                severity=SEVERITY_CRITICAL,
                title=(
                    f"Threshold {current.threshold:.2f} accepts "
                    f"{current.false_accept_count} wrong-person score(s)"
                ),
                detail=(
                    f"{current.false_accept_rate:.0%} of best-other-person scores "
                    "reach the current threshold, so the pipeline can attach the "
                    "wrong name automatically. Suggested threshold "
                    f"{calibration.suggested_threshold:.3f}: "
                    f"{calibration.suggested_reason}."
                ),
                action="set-threshold",
                context=facts,
            )
        )
    elif current.false_reject_count > suggested.false_reject_count:
        gained = current.false_reject_count - suggested.false_reject_count
        issues.append(
            LibraryIssue(
                kind="threshold-too-high",
                severity=(
                    SEVERITY_WARNING
                    if current.false_reject_rate < 0.2
                    else SEVERITY_CRITICAL
                ),
                title=(
                    f"Threshold {current.threshold:.2f} rejects "
                    f"{current.false_reject_rate:.0%} of correct matches"
                ),
                detail=(
                    f"{current.false_reject_count} of "
                    f"{len(calibration.genuine_scores)} same-person scores fall "
                    "below the current threshold and would need manual naming. "
                    f"Moving to {calibration.suggested_threshold:.3f} recovers "
                    f"{gained} of them while still accepting "
                    f"{suggested.false_accept_count} wrong-person score(s) — "
                    f"{calibration.suggested_reason}."
                ),
                action="set-threshold",
                context=facts,
            )
        )
    return issues


def _person_issue(
    person: PersonHealth,
    *,
    kind: str,
    severity: str,
    title: str,
    detail: str,
    action: str,
    context: dict[str, float | int | str],
) -> LibraryIssue:
    """Build one person-scoped issue."""
    return LibraryIssue(
        kind=kind,
        severity=severity,
        title=title,
        detail=detail,
        action=action,
        person_public_id=person.speaker_public_id,
        person_name=person.speaker_name,
        context=dict(context),
    )


def _issue_sort_key(issue: LibraryIssue) -> tuple[int, str, str]:
    """Sort issues by severity, then stably by kind and person."""
    rank = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
    return (rank.get(issue.severity, 3), issue.kind, issue.person_name or "")


def _person_sort_key(person: PersonHealth) -> tuple[int, float, str]:
    """Sort the least usable people first."""
    rank = {AVAILABILITY_UNUSABLE: 0, AVAILABILITY_FRAGILE: 1, AVAILABILITY_OK: 2}
    return (
        rank.get(person.availability, 3),
        person.matching_seconds,
        person.speaker_name.casefold(),
    )


__all__ = [
    "AVAILABILITY_FRAGILE",
    "AVAILABILITY_OK",
    "AVAILABILITY_UNUSABLE",
    "LibraryHealthReport",
    "LibraryIssue",
    "PersonHealth",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "analyze_library_health",
]
