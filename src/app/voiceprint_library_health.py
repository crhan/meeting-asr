"""Voiceprint library health: can this library actually identify people?

:mod:`app.voiceprint_quality` answers "are a person's samples consistent with
each other" — a within-cluster question. That check is blind to the failure
modes that silently remove a person from matching altogether:

- a person whose samples were never embedded for the *active* model, so they
  are in the library but not in the matching pool, and
- a person whose samples are all quarantined, which the per-sample view
  reports as "0 suspicious" because there is nothing left to find fault with.

Both render as healthy green in a sample-consistency view while the person is
simply never matched. A third failure mode hides the same way: two people
whose samples sit close enough to be accepted as each other are each perfectly
self-consistent, so nothing inside either cluster looks wrong. This module
answers the prior question — *availability*, and who can be told apart from
whom — and joins it with the store-wide threshold calibration into one
prioritized, actionable issue list.

Read-only; nothing here mutates the store.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.voiceprint_audio import resolve_voiceprint_sample_source
from app.voiceprint_calibration import (
    ConfusablePair,
    VoiceprintCalibrationReport,
    calibrate_voiceprint_thresholds,
)
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_models import VoiceprintSampleRow, VoiceprintSpeakerRow
from app.voiceprint_sample_overlap import inspect_sample_sources
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
# How thin a person's win over their nearest other person may get before it
# is worth naming. Deliberately a *margin* and not an absolute score: what
# decides a name is which candidate ranks first, so two people at 0.82 who
# each score 0.95 as themselves are separable and must stay silent, while two
# at 0.40 separated by 0.01 are one noisy capture from swapping. Wider than
# the 0.02 the threshold suggester keeps clear of the worst impostor, because
# this advisory has to fire *before* the order reverses, not as it does.
CONFUSABLE_WARNING_MARGIN = 0.05


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
    overlapped, orphaned = _source_findings(samples, embedded_ids, projects_dir)
    issues = _issues(people, calibration, overlapped, orphaned)
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
    seconds = sum(row.embedded_duration_ms / 1000.0 for row in matching)
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
    orphaned: dict[str, int],
) -> tuple[LibraryIssue, ...]:
    """Build the prioritized issue list."""
    issues: list[LibraryIssue] = []
    issues.extend(_threshold_issues(calibration))
    issues.extend(_confusable_issues(calibration))
    for person in people:
        issues.extend(_person_issues(person))
        issues.extend(
            _overlap_issues(person, overlapped.get(person.speaker_public_id, 0))
        )
        issues.extend(_orphan_issues(person, orphaned.get(person.speaker_public_id, 0)))
    return tuple(sorted(issues, key=_issue_sort_key))


def _source_findings(
    samples: list[VoiceprintSampleRow],
    embedded_ids: set[int],
    projects_dir: Path | None,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Count what each person's matching samples' source projects revealed.

    Only samples that actually reach the matching pool are counted: a
    quarantined or unembedded sample cannot drag anyone's centroid, so
    reporting it here would be an issue with no consequence.

    Args:
        samples: Every stored sample row.
        embedded_ids: Sample ids with a vector for the active model.
        projects_dir: Projects parent directory, or None to skip the check.

    Returns:
        Two person-public-id maps: overlapped matching-sample count, and
        matching-sample count whose source project is gone. Both empty when no
        projects directory was supplied, so "not checked" never renders as
        "checked and clean".
    """
    in_pool = [
        row
        for row in samples
        if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
        and row.sample_id in embedded_ids
    ]
    report = inspect_sample_sources(in_pool, projects_dir)
    overlapped: dict[str, int] = defaultdict(int)
    orphaned: dict[str, int] = defaultdict(int)
    for row in in_pool:
        if report.overlap.get(row.sample_id):
            overlapped[row.speaker_public_id] += 1
        # `is False` on purpose: a missing key means the check never ran at
        # all, which must not be counted as a gone project.
        if report.available.get(row.project_id) is False:
            orphaned[row.speaker_public_id] += 1
    return dict(overlapped), dict(orphaned)


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


def _orphan_issues(person: PersonHealth, orphaned: int) -> list[LibraryIssue]:
    """Report matching samples whose source project no longer exists.

    Clips live in the library, so these samples keep matching normally -- this
    is not a broken person. What is gone is every way of *checking* them: they
    cannot be heard in context, and the overlap check silently skips them, so
    a clean bill of health for this person is partly an untested claim. When
    the whole matching pool is orphaned that claim is entirely untested, which
    is why the severity turns.
    """
    if orphaned <= 0:
        return []
    total = person.matching_sample_count
    entire = orphaned >= total
    return [
        _person_issue(
            person,
            kind="orphaned-samples",
            severity=SEVERITY_WARNING if entire else SEVERITY_INFO,
            title=(
                f"{person.speaker_name} has {orphaned} of {total} matching "
                "sample(s) from a deleted project"
            ),
            detail=(
                "The clips are still in the library and still match, but their "
                "source project is gone: they cannot be reviewed in context, "
                "and the overlap check cannot run on them, so they are counted "
                "as clean without having been examined."
                + (
                    " That covers every matching sample this person has, so "
                    "nothing about this voiceprint has actually been verified. "
                    "Capture from a project that still exists."
                    if entire
                    else ""
                )
            ),
            action="capture",
            context={
                "name": person.speaker_name,
                "orphaned_count": orphaned,
                "matching_sample_count": total,
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


def _confusable_issues(
    calibration: VoiceprintCalibrationReport | None,
) -> list[LibraryIssue]:
    """Report people whose samples come too close to another person.

    This is the per-pair reading of the same evidence ``threshold-too-low``
    summarizes library-wide, and the two are deliberately both raised: they
    offer different remedies for one fact. Moving the threshold trades these
    wrong accepts for wrong rejects everywhere; fixing the two people named
    here costs nothing elsewhere.

    Neither the availability facts nor ``voiceprint quality`` can see this.
    Both people can have plenty of embedded samples, and each cluster can be
    perfectly self-consistent, precisely while they are consistent with each
    other -- consistency is measured against a person's own centroid, so being
    close to someone else's is invisible from inside.

    What is *not* reported is a merely high score against another centroid.
    Matching ranks candidates and applies the threshold to the winner, so
    someone who scores 0.82 as their neighbour while scoring 0.95 as
    themselves is named correctly every time; calling that critical would send
    the operator off to re-capture perfectly good audio. Only a thin or lost
    lead counts -- see ``ConfusablePair``.
    """
    if calibration is None:
        return []
    threshold = calibration.current_threshold
    return [
        _confusable_issue(pair, threshold)
        for pair in calibration.neighbors
        if pair.crossing_count > 0
        or pair.outranked_count > 0
        or (pair.min_lead is not None and pair.min_lead < CONFUSABLE_WARNING_MARGIN)
    ]


def _acceptance_phrase(reason: str | None, threshold: float) -> str:
    """Say which rule attached the name, since they are not interchangeable.

    ``_acceptance_decision`` has two ways to say yes, and only one of them is
    about the bar. A winner that stays under the threshold but runs away from
    the runner-up is accepted all the same, and describing that as clearing
    the threshold tells the operator something that did not happen -- they
    would then raise the threshold and watch the wrong name survive it.
    """
    if reason == "strong-margin":
        return (
            f"it stays under {threshold:.2f}, but leads the runner-up by enough "
            "that the strong-margin rule accepts it anyway"
        )
    if reason == "mixed":
        return (
            f"some clear the {threshold:.2f} bar, the rest stay under it and are "
            "accepted by the strong-margin rule for leading the runner-up"
        )
    return f"it clears the {threshold:.2f} bar"


def _tie_phrase(pair: ConfusablePair) -> str:
    """Name the tie case, because its remedy is a different one.

    Winning first place at an identical score is not the other person sounding
    more like the probe -- it is library name order breaking a draw. Two
    entries that draw on every sample are almost always one person entered
    twice, and telling the operator to capture more audio for "both" of them
    would be advice for a problem they do not have.
    """
    if pair.tied_win_count <= 0:
        return ""
    if pair.tied_win_count >= pair.crossing_count:
        return (
            " Every one of those is an exact draw, decided only by which name "
            "sorts first -- which is what one person entered into the library "
            "twice looks like. Check that before capturing anything."
        )
    return (
        f" {pair.tied_win_count} of them are exact draws, decided only by "
        "which name sorts first rather than by sounding more alike."
    )


def _confusable_issue(pair: ConfusablePair, threshold: float) -> LibraryIssue:
    """Build one issue for a person who risks being taken for someone else."""
    crossing = pair.crossing_count
    entire = crossing >= pair.sample_count
    lead = pair.min_lead if pair.min_lead is not None else 0.0
    facts: dict[str, float | int | str] = {
        "name": pair.person_name,
        "other_name": pair.other_name,
        "other_public_id": pair.other_public_id,
        "best_score": round(pair.best_score, 3),
        "min_lead": round(lead, 3),
        # Recorded during the replay, not derivable from min_lead: a tie ranks
        # the rival first at a lead of exactly 0.000.
        "outranked_count": pair.outranked_count,
        "tied_win_count": pair.tied_win_count,
        "threshold": threshold,
        "crossing_count": crossing,
        "sample_count": pair.sample_count,
        # Named, not just counted, so a CLI or API consumer can act on the
        # rows. The web sample list does not surface sample ids yet, which is
        # why the offered action stays "capture" rather than pointing at rows
        # the operator would then have to find by hand.
        "crossing_sample_public_ids": ",".join(pair.crossing_sample_public_ids),
        # Which acceptance rule attached the name. A strong-margin acceptance
        # happens *below* the bar, so a report that says "clears 0.75" about
        # a 0.707 winner is simply false.
        "accept_reason": pair.accept_reason or "threshold",
    }
    if crossing > 0:
        return LibraryIssue(
            kind="confusable-people",
            severity=SEVERITY_CRITICAL,
            title=(
                f"{pair.person_name} has {crossing} of {pair.sample_count} "
                f"sample(s) the pipeline would name {pair.other_name}"
            ),
            detail=(
                f"For {crossing} of this person's samples, matching ranks "
                f"{pair.other_name} ahead of the person themselves -- measured "
                "against their own leave-one-out centroid, so the sample is "
                "judged as an unseen probe would be -- and attaches that "
                f"name: {_acceptance_phrase(pair.accept_reason, threshold)}. "
                "These are wrong names today, not a risk."
                + _tie_phrase(pair)
                + (
                    " That covers every sample they have, so this voiceprint "
                    "cannot be told apart from the other one at all; capture "
                    "audio from a meeting where only one of them speaks."
                    if entire
                    else " Capture more audio for this person so the centroid "
                    "moves onto what is distinctive about them, or confirm "
                    "that these two library entries are not in fact the same "
                    "person."
                )
            ),
            action="capture",
            person_public_id=pair.person_public_id,
            person_name=pair.person_name,
            context=facts,
        )
    # Who ranked first is read from the replay, never inferred from the lead:
    # on an exact tie the lead is 0.000 while the rival is still ahead, and
    # falling through to the sentence below would announce that the right name
    # is winning when it is not.
    if pair.outranked_count > 0:
        return LibraryIssue(
            kind="confusable-people",
            severity=SEVERITY_WARNING,
            title=(
                f"{pair.person_name} ranks behind {pair.other_name} on "
                f"{pair.outranked_count} of {pair.sample_count} sample(s)"
            ),
            detail=(
                f"On those samples {pair.other_name} is already the pipeline's "
                "first choice, so the right name is not winning -- it is only "
                f"that the score falls short of the {threshold:.2f} bar, so the "
                "sample lands in manual review instead of being given the wrong "
                "name outright. Lowering the threshold would turn this into a "
                "wrong name; capturing audio for this person is what fixes it."
            ),
            action="capture",
            person_public_id=pair.person_public_id,
            person_name=pair.person_name,
            context=facts,
        )
    return LibraryIssue(
        kind="confusable-people",
        severity=SEVERITY_WARNING,
        title=(
            f"{pair.person_name} beats {pair.other_name} on their own samples "
            f"by only {lead:.3f}"
        ),
        detail=(
            "The right name still wins, so nothing is mislabelled today, but "
            "the two centroids are close enough that one noisy capture can "
            "reverse the order. Adding audio for this person from another "
            "meeting is what widens the gap."
        ),
        action="capture",
        person_public_id=pair.person_public_id,
        person_name=pair.person_name,
        context=facts,
    )


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
        # The true populations, not len(exported scores): the counts above are
        # scaled back to the full population, so pairing them with the capped
        # array length yields sentences like "3000 of 2000 same-person scores".
        "genuine_count": calibration.genuine.count if calibration.genuine else 0,
        "impostor_count": calibration.impostor.count if calibration.impostor else 0,
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
                    # facts["genuine_count"], not len(genuine_scores): the count
                    # above is scaled to the whole library, so the capped export
                    # length would read "3000 of 2000".
                    f"{current.false_reject_count} of "
                    f"{facts['genuine_count']} same-person scores fall "
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
    "CONFUSABLE_WARNING_MARGIN",
    "LibraryHealthReport",
    "LibraryIssue",
    "PersonHealth",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "analyze_library_health",
]
