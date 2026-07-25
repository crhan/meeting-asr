"""Where to harvest more voiceprint samples for one person.

:mod:`app.voiceprint_library_health` says a person is short on audio, is
characterized by a single recording, or has too few samples to validate. What
it cannot say is *what to do about it*: its ``capture`` action only knows the
verb, not the object, so the operator was dropped on the project list to
rediscover by hand which meetings that person even appears in, open each one,
and read transcripts looking for their name.

This module answers the missing question. For one person it scans every
project and reports which ones hold harvestable speech for them, how much,
what it would sound like, and why that project is worth opening -- so the
existing per-project capture picker (plan -> select -> run) can be entered
already aimed at a speaker instead of from scratch.

Attribution runs on the same identity keys the merge pipeline uses, in the
same priority order:

``person-map``
    ``speakers/speaker_person_map.json`` links a project speaker to this
    person's public id. Confirmed by a human apply or by stabilization.
``name``
    ``speakers/speaker_map.json`` display name folds to the person's name and
    no person link contradicts it. Weaker, but the common real case: a speaker
    can be named in a project without ever being linked to the library.

Ranking is deficit-driven, not generic. "Best source" means different projects
depending on whether the person needs *more audio*, *more samples*, or
*another room* -- so the report carries the person's deficits and each source
carries the reason kinds that earned it its rank, for the caller to render.

Read-only: nothing here writes to a project or to the voiceprint store.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.core.project_refs import list_projects
from app.core.voiceprint_review_service import plan_capture
from app.models import SentenceSegment
from app.speaker_labeling import load_transcript_result
from app.transcript_merge import is_placeholder_name, name_fold
from app.voiceprint_library_health import (
    MIN_HEALTHY_MATCHING_SECONDS,
    MIN_HEALTHY_PROJECT_COUNT,
    PersonHealth,
    person_availability,
)
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_people import get_voiceprint_person
from app.voiceprint_quality import DEFAULT_MIN_CLUSTER_SIZE
from app.voiceprint_segment_selection import (
    MIN_RECOMMENDED_SCORE,
    ScoredVoiceprintSegment,
    select_voiceprint_segments,
)
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
)

LOGGER = logging.getLogger(__name__)

EVIDENCE_PERSON_MAP = "person-map"
EVIDENCE_NAME = "name"

# How confident attribution is, per evidence kind. A person-map entry was
# written by an apply or by stabilization and names the library person
# directly; a display-name match only says two strings agree.
_EVIDENCE_WEIGHT = {
    EVIDENCE_PERSON_MAP: 1.0,
    EVIDENCE_NAME: 0.85,
}

DEFICIT_UNUSABLE = "unusable"
DEFICIT_FRAGILE = "fragile-cluster"
DEFICIT_SHORT_AUDIO = "short-audio"
DEFICIT_SINGLE_SOURCE = "single-source"

REASON_NEW_PROJECT = "new-project"
REASON_LARGE_SUPPLY = "large-supply"
REASON_RETRY_QUARANTINED = "retry-quarantined"
REASON_NAME_ONLY = "name-only"
REASON_ALREADY_HARVESTED = "already-harvested"
REASON_THIN_SUPPLY = "thin-supply"

# Supply saturates here: past two minutes of good speech a project is not a
# better source, it just has more of the same voice.
_SUPPLY_SATURATION_SECONDS = 120.0
# Below this a project is worth listing but should not outrank a real source.
_THIN_SUPPLY_SECONDS = 15.0
# Fallback number of pre-selected clips when the planner's own default picks
# were all filtered out as already-captured.
_DEFAULT_PICKS = 3
# Clips whose time range is within this of an existing sample are the same
# utterance: re-capturing them adds a duplicate, not a new observation.
_OVERLAP_TOLERANCE_MS = 250


@dataclass(frozen=True, slots=True)
class CandidateClip:
    """One clip a capture run would take, identified the way capture identifies it.

    These come from ``plan_voiceprint_capture`` rather than a second selection
    of our own, for two reasons. The obvious one is that ``rel_path`` is the
    currency a capture run speaks, so a caller can hand these straight back and
    capture without re-planning anything. The subtler one is that the planner
    spreads its picks across the speaker's timeline on purpose -- ranking
    candidates by score alone clusters them inside a single dense monologue and
    ends up characterizing one speaking style rather than a voice.
    """

    rel_path: str
    begin_time_ms: int
    end_time_ms: int
    duration_seconds: float
    text: str
    score: float
    recommended: bool
    # Another speaker's turn begins or ends within
    # ``_OVERLAP_RISK_GAP_MS`` of this one. Clip extraction no longer pads into
    # them, but a gap that short means the two were genuinely talking over each
    # other, and diarizer boundaries inside the segment cannot be trusted to
    # be clean either -- measurably worse as a reference.
    overlap_risk: bool = False


@dataclass(frozen=True, slots=True)
class SampleSource:
    """One project that holds harvestable speech for a person."""

    project_id: str
    project_dir: Path
    title: str
    meeting_time: str | None
    created_at: str | None
    speaker_id: int
    speaker_name: str
    # The library person the capture plan resolved for this speaker. Capture
    # validates a selection against (rel_path, times, name, person), so a
    # caller reusing these clips must echo the *plan's* identity, not ours.
    person_public_id: str | None
    evidence: str
    # Harvestable supply: segments good enough to be a default capture pick,
    # excluding utterances already in the library.
    candidate_count: int
    candidate_seconds: float
    best_score: float
    # What this project already contributed, by sample status.
    matching_sample_count: int
    quarantined_sample_count: int
    other_sample_count: int
    priority: float
    reasons: tuple[str, ...]
    # Directly capturable clips, or empty when the plan could not be built.
    clips: tuple[CandidateClip, ...] = ()

    @property
    def existing_sample_count(self) -> int:
        """Return every library sample sourced from this project."""
        return (
            self.matching_sample_count
            + self.quarantined_sample_count
            + self.other_sample_count
        )


@dataclass(frozen=True, slots=True)
class SkippedProject:
    """A project that could not be searched, and why."""

    project_id: str
    title: str
    reason: str


@dataclass(frozen=True, slots=True)
class SampleSourceReport:
    """Harvest opportunities for one person across every known project."""

    person_public_id: str
    person_name: str
    health: PersonHealth | None
    deficits: tuple[str, ...]
    sources: tuple[SampleSource, ...]
    scanned_project_count: int
    skipped: tuple[SkippedProject, ...] = field(default_factory=tuple)

    @property
    def total_candidate_seconds(self) -> float:
        """Return harvestable speech across every source."""
        return round(sum(source.candidate_seconds for source in self.sources), 1)

    @property
    def new_project_count(self) -> int:
        """Return sources that contribute no matching sample today."""
        return sum(1 for source in self.sources if REASON_NEW_PROJECT in source.reasons)


def find_sample_sources(
    person_public_id: str,
    *,
    projects_dir: Path | None = None,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    limit: int = 20,
    with_clips: bool = True,
) -> SampleSourceReport:
    """
    Find where more samples for a person can be harvested.

    Args:
        person_public_id: Library person public id (``vpp-...``).
        projects_dir: Optional projects parent directory.
        store_dir: Optional voiceprint store directory.
        provider: Optional embedding provider override.
        model: Optional embedding model key override.
        limit: Maximum sources to return.
        with_clips: Also plan each source's capturable clips. Costs one capture
            plan per candidate project; turn it off for a ranking-only answer.

    Returns:
        Ranked harvest opportunities with the deficits that ordered them.
    """
    db_path = get_voiceprint_db_path(store_dir)
    samples = [
        row
        for row in list_all_voiceprint_samples(db_path)
        if row.speaker_public_id == person_public_id
    ]
    health = person_availability(
        person_public_id, store_dir=store_dir, provider=provider, model=model
    )
    _provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    embedded_ids = list_embedded_sample_ids(resolved_model, db_path)
    person_name = _resolve_person_name(person_public_id, health, samples, db_path)
    deficits = _deficits(health)
    matching_project_ids = _matching_project_ids(samples, health, embedded_ids)
    sampled_ranges = _sampled_ranges(samples)
    by_project_status = _sample_counts_by_project(samples, embedded_ids)

    sources: list[SampleSource] = []
    skipped: list[SkippedProject] = []
    scanned = 0
    listing = list_projects(projects_dir, restrict_to_projects_dir=True)
    for item in listing.projects:
        scanned += 1
        try:
            found = _project_sources(
                item,
                person_public_id=person_public_id,
                person_name=person_name,
                sampled_ranges=sampled_ranges.get(item.project_id, ()),
                counts=by_project_status.get(item.project_id, _StatusCounts()),
                is_matching_project=item.project_id in matching_project_ids,
                deficits=deficits,
                with_clips=with_clips,
                store_dir=store_dir,
            )
        except Exception as error:  # noqa: BLE001 - one bad project must not blind the rest
            LOGGER.warning(
                "Sample sourcing skipped project %s: %s", item.project_id, error
            )
            skipped.append(
                SkippedProject(item.project_id, item.title, _skip_reason(error))
            )
            continue
        sources.extend(found)

    sources.sort(key=lambda source: (-source.priority, source.project_id))
    return SampleSourceReport(
        person_public_id=person_public_id,
        person_name=person_name,
        health=health,
        deficits=deficits,
        sources=tuple(sources[:limit]),
        scanned_project_count=scanned,
        skipped=tuple(skipped),
    )


@dataclass(slots=True)
class _StatusCounts:
    """Library samples this project already contributed, by matching status."""

    matching: int = 0
    quarantined: int = 0
    other: int = 0


def _deficits(health: PersonHealth | None) -> tuple[str, ...]:
    """Return which health bars this person currently fails."""
    if health is None:
        return (DEFICIT_UNUSABLE,)
    found: list[str] = []
    if not health.usable:
        found.append(DEFICIT_UNUSABLE)
    if health.matching_sample_count < DEFAULT_MIN_CLUSTER_SIZE:
        found.append(DEFICIT_FRAGILE)
    if health.matching_seconds < MIN_HEALTHY_MATCHING_SECONDS:
        found.append(DEFICIT_SHORT_AUDIO)
    if health.project_count < MIN_HEALTHY_PROJECT_COUNT:
        found.append(DEFICIT_SINGLE_SOURCE)
    return tuple(found)


def _resolve_person_name(
    person_public_id: str,
    health: PersonHealth | None,
    samples: list[VoiceprintSampleRow],
    db_path: Path,
) -> str:
    """
    Return this person's display name, sample-free people included.

    Availability facts are derived from stored samples, so a person created in
    the library but never captured has neither health nor samples to read a
    name from. Leaving it blank would silently disable the display-name half of
    attribution -- and that is exactly the person the "no samples" issue sends
    here, whose speakers in past meetings are usually named but not yet linked.
    """
    if health is not None:
        return health.speaker_name
    if samples:
        return samples[0].speaker_name
    row = get_voiceprint_person(person_public_id, db_path)
    return row.name if row else ""


def _is_matchable(row: VoiceprintSampleRow, embedded_ids: set[int]) -> bool:
    """Return whether one stored sample actually reaches the matching pool.

    Both conditions are required, and the second is the one that quietly
    changes under everyone's feet: switching embedding provider or model
    leaves every row enabled but without a vector for the *active* model, so
    a status-only test would report a fully harvested library that matches
    nobody.
    """
    from app.voiceprint_quality import VOICEPRINT_MATCHING_SAMPLE_STATUSES

    return (
        row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
        and row.sample_id in embedded_ids
    )


def _matching_project_ids(
    samples: list[VoiceprintSampleRow],
    health: PersonHealth | None,
    embedded_ids: set[int],
) -> set[str]:
    """
    Return projects already contributing a *matchable* sample.

    Quarantined and un-embedded samples deliberately do not count: a project
    that only ever yielded quarantined clips contributes nothing to the
    matching pool, so for sourcing purposes it is still an unharvested room.
    """
    if health is None:
        return set()
    return {row.project_id for row in samples if _is_matchable(row, embedded_ids)}


def _sample_counts_by_project(
    samples: list[VoiceprintSampleRow], embedded_ids: set[int]
) -> dict[str, _StatusCounts]:
    """Group this person's existing samples by project and matching status."""
    counts: dict[str, _StatusCounts] = defaultdict(_StatusCounts)
    for row in samples:
        bucket = counts[row.project_id]
        if _is_matchable(row, embedded_ids):
            bucket.matching += 1
        elif row.sample_status == "quarantined":
            bucket.quarantined += 1
        else:
            bucket.other += 1
    return dict(counts)


def _sampled_ranges(
    samples: list[VoiceprintSampleRow],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return already-captured source time ranges per project."""
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in samples:
        ranges[row.project_id].append(
            (row.source_begin_time_ms, row.source_end_time_ms)
        )
    return {key: tuple(sorted(value)) for key, value in ranges.items()}


def _project_sources(
    item,
    *,
    person_public_id: str,
    person_name: str,
    sampled_ranges: tuple[tuple[int, int], ...],
    counts: _StatusCounts,
    is_matching_project: bool,
    deficits: tuple[str, ...],
    with_clips: bool,
    store_dir: Path | None,
) -> list[SampleSource]:
    """Find this person's harvestable speaker tracks inside one project."""
    speakers_dir = item.project_dir / "speakers"
    attributions = _attributed_speakers(
        speakers_dir, person_public_id=person_public_id, person_name=person_name
    )
    if not attributions:
        return []
    sentences_path = item.project_dir / "asr" / "sentences.json"
    if not sentences_path.is_file():
        raise FileNotFoundError("project has no normalized transcript")
    result = load_transcript_result(sentences_path)
    all_segments = sorted(
        result.sentences, key=lambda seg: (seg.begin_time_ms, seg.end_time_ms)
    )
    plan = _capture_plan(item.project_dir, store_dir) if with_clips else None
    sources: list[SampleSource] = []
    for speaker_id, (evidence, speaker_name) in sorted(attributions.items()):
        segments = [seg for seg in all_segments if seg.speaker_id == speaker_id]
        if not segments:
            continue
        candidates = _harvestable(segments, all_segments, sampled_ranges)
        if not candidates:
            continue
        planned = _planned_speaker(plan, speaker_id)
        sources.append(
            _build_source(
                item,
                speaker_id=speaker_id,
                speaker_name=planned.name if planned else speaker_name,
                person_public_id=planned.person_public_id if planned else None,
                evidence=evidence,
                candidates=candidates,
                clips=_planned_clips(planned, sampled_ranges),
                counts=counts,
                is_matching_project=is_matching_project,
                deficits=deficits,
            )
        )
    return sources


def _capture_plan(project_dir: Path, store_dir: Path | None):
    """Plan this project's capture clips, or None when it cannot be planned.

    ``store_dir`` must be threaded through: the planner resolves each speaker's
    library person against it, so defaulting here would make an isolated
    ``--store-dir`` run silently identify people from the real library.

    A project with no named speaker raises rather than returning an empty plan,
    and a plan failure must not lose the source: the row is still worth showing
    with its supply figures, it just cannot offer one-click capture.
    """
    try:
        return plan_capture(project_dir, store_dir=store_dir)
    except Exception as error:  # noqa: BLE001 - a source without clips still ranks
        LOGGER.debug("Capture plan unavailable for %s: %s", project_dir, error)
        return None


def _planned_speaker(plan, speaker_id: int):
    """Return the planned speaker matching this project speaker id."""
    if plan is None:
        return None
    for speaker in plan.speakers:
        if speaker.speaker_id == speaker_id:
            return speaker
    return None


def _planned_clips(
    planned, sampled_ranges: tuple[tuple[int, int], ...]
) -> tuple[CandidateClip, ...]:
    """
    Return this speaker's planned clips, minus utterances already in the library.

    Re-offering a stored clip wastes the operator's attention and, if taken,
    stores a duplicate observation that inflates the cluster without adding
    information. When the filter removes every default pick, the highest-scoring
    survivors are promoted so the caller still has something pre-selected.
    """
    if planned is None:
        return ()
    fresh = [
        clip
        for clip in planned.clips
        if not _range_overlaps(
            clip.source_begin_time_ms, clip.source_end_time_ms, sampled_ranges
        )
    ]
    if not fresh:
        return ()
    # The planner already judged overlap risk while scoring; recomputing it
    # here would let the two disagree about the same clip.
    risky = {clip.rel_path for clip in fresh if clip.overlap_risk}
    picks = _default_picks(fresh, risky)
    return tuple(
        CandidateClip(
            rel_path=clip.rel_path,
            begin_time_ms=clip.source_begin_time_ms,
            end_time_ms=clip.source_end_time_ms,
            duration_seconds=round(clip.duration_seconds, 1),
            text=clip.text.strip(),
            score=round(clip.selection_score, 3),
            recommended=clip.rel_path in picks,
            overlap_risk=clip.rel_path in risky,
        )
        for clip in fresh
    )


def _default_picks(fresh: list, risky: set[str]) -> set[str]:
    """
    Choose which surviving clips arrive pre-selected.

    The planner's own recommendations come first, minus anything flagged for
    overlap risk: a pre-ticked checkbox is a recommendation, and recommending a
    clip taken from a stretch where two people talk over each other is how a
    reference voice quietly acquires someone else's. Risky clips stay listed
    and selectable -- when a source offers nothing else, a mediocre sample the
    operator chose knowingly beats no sample at all.

    When the filters thin the default picks out, the gaps are refilled with the
    qualifying survivor *furthest in time* from what is already picked -- not
    the next highest score. Spread is what made the planner's picks worth
    trusting, and topping up by score alone walks straight back into one dense
    monologue.
    """
    picked = [clip for clip in fresh if clip.recommended and clip.rel_path not in risky]
    pool = [
        clip
        for clip in fresh
        if clip not in picked
        and clip.rel_path not in risky
        and clip.selection_score >= MIN_RECOMMENDED_SCORE
    ]
    while pool and len(picked) < _DEFAULT_PICKS:
        if picked:
            chosen = max(
                pool,
                key=lambda clip: min(
                    abs(clip.source_begin_time_ms - other.source_begin_time_ms)
                    for other in picked
                ),
            )
        else:
            chosen = max(pool, key=lambda clip: clip.selection_score)
        picked.append(chosen)
        pool.remove(chosen)
    return {clip.rel_path for clip in picked}


def _build_source(
    item,
    *,
    speaker_id: int,
    speaker_name: str,
    person_public_id: str | None,
    evidence: str,
    candidates: list[ScoredVoiceprintSegment],
    clips: tuple[CandidateClip, ...],
    counts: _StatusCounts,
    is_matching_project: bool,
    deficits: tuple[str, ...],
) -> SampleSource:
    """Assemble one ranked source row from its harvestable candidates."""
    seconds = round(sum(_seconds(item.segment) for item in candidates), 1)
    best = max(candidate.score for candidate in candidates)
    reasons = _reasons(
        evidence=evidence,
        seconds=seconds,
        counts=counts,
        is_matching_project=is_matching_project,
    )
    priority = _priority(
        evidence=evidence,
        seconds=seconds,
        candidates=candidates,
        is_matching_project=is_matching_project,
        counts=counts,
        deficits=deficits,
    )
    return SampleSource(
        project_id=item.project_id,
        project_dir=item.project_dir,
        title=item.title,
        meeting_time=item.meeting_time,
        created_at=item.created_at,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        person_public_id=person_public_id,
        evidence=evidence,
        candidate_count=len(candidates),
        candidate_seconds=seconds,
        best_score=round(best, 3),
        matching_sample_count=counts.matching,
        quarantined_sample_count=counts.quarantined,
        other_sample_count=counts.other,
        priority=priority,
        reasons=reasons,
        clips=clips,
    )


def _harvestable(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    sampled_ranges: tuple[tuple[int, int], ...],
) -> list[ScoredVoiceprintSegment]:
    """
    Return this speaker's segments worth capturing that are not already stored.

    Supply is measured at the *recommended* bar rather than the selection
    floor: a source is only as good as the clips a capture run would default
    to, so counting barely-passable fragments would advertise audio the picker
    would never check.
    """
    fresh = [segment for segment in segments if not _overlaps(segment, sampled_ranges)]
    if not fresh:
        return []
    scored = select_voiceprint_segments(
        fresh, all_segments, sample_count=0, candidate_count=len(fresh)
    )
    return [item for item in scored if item.score >= MIN_RECOMMENDED_SCORE]


def _overlaps(
    segment: SentenceSegment, sampled_ranges: tuple[tuple[int, int], ...]
) -> bool:
    """Return whether a segment is an utterance the library already holds."""
    return _range_overlaps(segment.begin_time_ms, segment.end_time_ms, sampled_ranges)


def _range_overlaps(
    begin_ms: int, end_ms: int, sampled_ranges: tuple[tuple[int, int], ...]
) -> bool:
    """Return whether a time range is an utterance the library already holds."""
    for begin, end in sampled_ranges:
        if (
            begin_ms < end + _OVERLAP_TOLERANCE_MS
            and begin < end_ms + _OVERLAP_TOLERANCE_MS
        ):
            return True
    return False


def _reasons(
    *,
    evidence: str,
    seconds: float,
    counts: _StatusCounts,
    is_matching_project: bool,
) -> tuple[str, ...]:
    """Return the reason kinds that describe this source, for the caller to render."""
    reasons: list[str] = []
    if not is_matching_project:
        reasons.append(REASON_NEW_PROJECT)
    elif counts.matching:
        reasons.append(REASON_ALREADY_HARVESTED)
    if counts.quarantined and not counts.matching:
        reasons.append(REASON_RETRY_QUARANTINED)
    if seconds >= _SUPPLY_SATURATION_SECONDS:
        reasons.append(REASON_LARGE_SUPPLY)
    elif seconds < _THIN_SUPPLY_SECONDS:
        reasons.append(REASON_THIN_SUPPLY)
    if evidence == EVIDENCE_NAME:
        reasons.append(REASON_NAME_ONLY)
    return tuple(reasons)


def _priority(
    *,
    evidence: str,
    seconds: float,
    candidates: list[ScoredVoiceprintSegment],
    is_matching_project: bool,
    counts: _StatusCounts,
    deficits: tuple[str, ...],
) -> float:
    """
    Score a source by how much it closes *this person's* gaps.

    The weights are deliberately deficit-driven rather than a generic "best
    audio wins". A second room matters most to someone characterized by one
    recording; raw supply matters most to someone short on seconds. Ranking by
    audio quality alone would keep recommending the meeting already harvested.
    """
    supply = min(1.0, seconds / _SUPPLY_SATURATION_SECONDS)
    quality = sum(item.score for item in sorted(candidates, key=lambda c: -c.score)[:5])
    quality /= min(5, len(candidates))
    diversity = 0.0
    if not is_matching_project:
        diversity = 1.0 if DEFICIT_SINGLE_SOURCE in deficits else 0.6
    retry = 1.0 if counts.quarantined and not counts.matching else 0.0
    supply_weight = 0.45 if DEFICIT_SHORT_AUDIO in deficits else 0.35
    score = supply_weight * supply + 0.30 * diversity + 0.15 * quality + 0.10 * retry
    return round(_EVIDENCE_WEIGHT.get(evidence, 0.5) * score, 4)


def _attributed_speakers(
    speakers_dir: Path, *, person_public_id: str, person_name: str
) -> dict[int, tuple[str, str]]:
    """
    Return project speaker ids attributed to this person, with their evidence.

    A person-map link wins over a name match for the same speaker: the link
    names the library person, while the name only says two strings agree.
    """
    names = _read_speaker_map(speakers_dir / "speaker_map.json")
    persons = _read_speaker_map(speakers_dir / "speaker_person_map.json")
    found: dict[int, tuple[str, str]] = {}
    for speaker_id, mapped in persons.items():
        if mapped == person_public_id:
            found[speaker_id] = (
                EVIDENCE_PERSON_MAP,
                names.get(speaker_id, "") or person_name,
            )
    if not person_name or is_placeholder_name(person_name):
        return found
    target = name_fold(person_name)
    for speaker_id, name in names.items():
        if speaker_id in found or is_placeholder_name(name):
            continue
        # A speaker already linked to a *different* person is not this person,
        # however the display name reads -- the link is the stronger claim.
        if persons.get(speaker_id):
            continue
        if name_fold(name) == target:
            found[speaker_id] = (EVIDENCE_NAME, name)
    return found


def _read_speaker_map(path: Path) -> dict[int, str]:
    """Read a project speaker map keyed by integer speaker id."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: dict[int, str] = {}
    for key, value in payload.items():
        try:
            speaker_id = int(key)
        except TypeError, ValueError:
            continue
        if isinstance(value, str) and value.strip():
            mapping[speaker_id] = value.strip()
    return mapping


def _seconds(segment: SentenceSegment) -> float:
    """Return a segment's duration in seconds."""
    return max(0, segment.end_time_ms - segment.begin_time_ms) / 1000.0


def _skip_reason(error: Exception) -> str:
    """Return a stable reason kind for a project that could not be searched."""
    if isinstance(error, FileNotFoundError):
        return "no-transcript"
    return "unreadable"


def sample_source_payload(report: SampleSourceReport) -> dict[str, object]:
    """Build stable automation output for a sourcing report."""
    return {
        "person_public_id": report.person_public_id,
        "person_name": report.person_name,
        "deficits": list(report.deficits),
        "matching_sample_count": (
            report.health.matching_sample_count if report.health else 0
        ),
        "matching_seconds": report.health.matching_seconds if report.health else 0.0,
        "project_count": report.health.project_count if report.health else 0,
        "scanned_project_count": report.scanned_project_count,
        "total_candidate_seconds": report.total_candidate_seconds,
        "sources": [
            {
                "project_id": source.project_id,
                "title": source.title,
                "meeting_time": source.meeting_time,
                "speaker_id": source.speaker_id,
                "speaker_name": source.speaker_name,
                "person_public_id": source.person_public_id,
                "evidence": source.evidence,
                "candidate_count": source.candidate_count,
                "candidate_seconds": source.candidate_seconds,
                "best_score": source.best_score,
                "matching_sample_count": source.matching_sample_count,
                "quarantined_sample_count": source.quarantined_sample_count,
                "priority": source.priority,
                "reasons": list(source.reasons),
                "clips": [
                    {
                        "rel_path": clip.rel_path,
                        "begin_time_ms": clip.begin_time_ms,
                        "end_time_ms": clip.end_time_ms,
                        "duration_seconds": clip.duration_seconds,
                        "text": clip.text,
                        "score": clip.score,
                        "recommended": clip.recommended,
                        "overlap_risk": clip.overlap_risk,
                    }
                    for clip in source.clips
                ],
            }
            for source in report.sources
        ],
        "skipped": [
            {
                "project_id": item.project_id,
                "title": item.title,
                "reason": item.reason,
            }
            for item in report.skipped
        ],
    }
