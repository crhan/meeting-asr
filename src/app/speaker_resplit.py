"""Under-split rescue: re-split a polluted speaker track by voiceprint + clustering.

ASR diarization sometimes *under-splits*: several real people collapse into one
project speaker track. The existing automatic stabilization
(:mod:`app.speaker_stabilization`) can only move sentences *between speakers that
already exist in the project*, so a person who is present in the cross-project
voiceprint library but was never given their own track stays buried, and a person
who is not in the library at all is invisible.

This module is the pure-analysis engine that fills that gap. It does **not** write
any project state; it embeds the suspect track's sentences, then produces a
:class:`TrackResplitPlan` describing two kinds of moves:

- **promotions**: groups of sentences that confidently belong to a *library*
  person other than the track's current identity. The wiring layer reassigns each
  group either to that person's existing project speaker (if one exists) or to a
  freshly minted speaker track.
- **residue clusters**: coherent groups of sentences that match *no* library
  person and are clearly unlike the track's own identity — genuine out-of-library
  speaker candidates that should land in a review-visible "unknown" bucket, with an
  optional suggestion to merge into a nearby existing speaker.

Design notes (these are the corrections that make the approach sound):

1. **Anchor on clean library vectors, never the polluted track centroid.** A track
   that mixes N people has a meaningless centroid, so judging "who deviates" by the
   track centroid is circular. Instead we group sentences by their per-sentence best
   *library* match (the library person vectors are clean), and validate each group
   with its own multi-sentence centroid.
2. **Decide on groups, not single sentences.** A single short ECAPA embedding is
   noisy; a group centroid over several sentences is stable. Every move requires a
   minimum sentence count and total duration.
3. **Coherence gate for residue.** Out-of-library detection from noisy embeddings is
   unreliable, so residue must form a *coherent* cluster (high-threshold connected
   components) — scattered low-score singletons (usually just the dominant speaker
   in poor audio) never qualify.

The engine reuses the embedding/clustering primitives already built for cluster
quality diagnostics and the library-vector loading from speaker matching, so it
shares the same on-disk embedding cache (cache hits when cluster quality already
ran with ``score_all_segments=True``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.project_manager import project_paths
from app.speaker_cluster_quality import (
    SpeakerClusterClip,
    _connected_components,
    _cosine,
    _embed_selected_segments,
    _is_low_information_segment,
    _load_cluster_context,
    _mean_vector,
    _normalize,
    _read_embedding_cache,
    _write_embedding_cache,
)
from app.speaker_labeling import load_speaker_person_mapping
from app.speaker_matching import (
    _KnownSpeakerVector,
    _known_speaker_vectors,
    _ranked_matches,
)

# Embedding clip extraction parameters. Kept identical to the stabilization cluster
# pass so this engine reuses its cached per-sentence embeddings instead of recomputing.
RESPLIT_MAX_SECONDS = 12.0
RESPLIT_PADDING_SECONDS = 0.5

# Decision thresholds. Conservative by design; tuned against real dry-run output.
# Tuned against real dry-run output on a known under-split project. The asymmetry is
# deliberate: a *promotion* carries strong positive evidence (a group centroid that
# snaps onto one specific library person with a clear lead over the track's own
# identity), so two long sentences suffice — reliability tracks total voiced audio,
# not sentence count, hence the duration gate does the real work. *Residue* carries
# only negative evidence (matches nobody), which is noise-prone, so its floor is set
# low enough that a denoised centroid still weakly resembling the dominant speaker is
# left alone rather than pulled into a bucket.
DEFAULT_CANDIDATE_FLOOR = 0.50  # per-sentence library score to count as evidence
DEFAULT_PROMOTE_CENTROID_THRESHOLD = 0.62  # group centroid vs candidate library vector
DEFAULT_PROMOTE_LEAD_MARGIN = 0.10  # centroid must beat the track's own identity by this
DEFAULT_RESIDUE_MATCH_FLOOR = 0.40  # below this to *every* library person => unmatched
DEFAULT_RESIDUE_CLUSTER_THRESHOLD = 0.62  # connected-component edge among residue clips
DEFAULT_MERGE_THRESHOLD = 0.62  # residue cluster centroid vs another speaker => suggest
DEFAULT_MIN_GROUP_SENTENCES = 2  # min sentences per move (duration gate is the real bar)
DEFAULT_MIN_GROUP_SECONDS = 6.0
DEFAULT_MIN_SUSPECT_SENTENCES = 6  # only examine tracks large enough to hide a speaker


@dataclass(frozen=True, slots=True)
class ResplitParams:
    """Tunable decision thresholds for one re-split analysis."""

    candidate_floor: float = DEFAULT_CANDIDATE_FLOOR
    promote_centroid_threshold: float = DEFAULT_PROMOTE_CENTROID_THRESHOLD
    promote_lead_margin: float = DEFAULT_PROMOTE_LEAD_MARGIN
    residue_match_floor: float = DEFAULT_RESIDUE_MATCH_FLOOR
    residue_cluster_threshold: float = DEFAULT_RESIDUE_CLUSTER_THRESHOLD
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD
    min_group_sentences: int = DEFAULT_MIN_GROUP_SENTENCES
    min_group_seconds: float = DEFAULT_MIN_GROUP_SECONDS
    min_suspect_sentences: int = DEFAULT_MIN_SUSPECT_SENTENCES


@dataclass(frozen=True, slots=True)
class ResplitSentence:
    """One sentence selected for a move, identified the same way as reassignments."""

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        """Return the sentence duration in milliseconds."""
        return max(0, self.end_time_ms - self.begin_time_ms)


@dataclass(frozen=True, slots=True)
class CandidatePerson:
    """A group of track sentences whose best library match is one other person."""

    source_speaker_id: int
    person_id: int
    person_public_id: str
    name: str
    centroid_score: float
    assigned_score: float
    lead: float
    total_seconds: float
    existing_speaker_id: int | None
    decision: str  # "promote" | "below-centroid" | "below-lead" | "too-few"
    sentences: tuple[ResplitSentence, ...]


@dataclass(frozen=True, slots=True)
class ResidueCluster:
    """A coherent group of sentences that match no library person and not the track."""

    source_speaker_id: int
    assigned_score: float | None
    best_library_name: str | None
    best_library_score: float | None
    merge_target_speaker_id: int | None
    merge_score: float | None
    total_seconds: float
    decision: str  # "unknown-bucket" | "merge-suggested" | "too-few" | "fits-track"
    sentences: tuple[ResplitSentence, ...]


@dataclass(frozen=True, slots=True)
class TrackResplitPlan:
    """Pure analysis result for one project; never written to project state."""

    project_root: Path
    provider: str
    model: str
    params: ResplitParams
    library_size: int
    suspect_speaker_ids: tuple[int, ...]
    candidates: tuple[CandidatePerson, ...]
    residue_clusters: tuple[ResidueCluster, ...]

    @property
    def promotions(self) -> tuple[CandidatePerson, ...]:
        """Return only the candidate groups that passed every promotion gate."""
        return tuple(item for item in self.candidates if item.decision == "promote")


def analyze_project_resplit(
    project_dir: Path,
    *,
    store_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    params: ResplitParams | None = None,
) -> TrackResplitPlan:
    """Analyze a project for under-split tracks and propose re-split moves.

    Args:
        project_dir: Project root directory.
        store_dir: Optional voiceprint store directory (``None`` => default).
        provider: Optional embedding provider override.
        model: Optional embedding model key override.
        params: Optional decision thresholds (``None`` => conservative defaults).

    Returns:
        A pure :class:`TrackResplitPlan`; no project state is modified.
    """
    params = params or ResplitParams()
    context = _load_cluster_context(project_dir, provider, model)
    known = _known_speaker_vectors(store_dir, context.model)
    known_by_public = {
        item.person_public_id: item for item in known.values() if item.person_public_id
    }
    person_map = load_speaker_person_mapping(
        project_paths(project_dir).speakers_dir / "speaker_person_map.json"
    )

    cache = _read_embedding_cache(context.project_root)
    clips_by_speaker: dict[int, list[SpeakerClusterClip]] = {}
    for speaker_id, segments in sorted(context.segments_by_speaker.items()):
        usable = [s for s in segments if not _is_low_information_segment(s)]
        if not usable:
            continue
        clips_by_speaker[speaker_id] = _embed_selected_segments(
            context,
            speaker_id,
            usable,
            max_seconds=RESPLIT_MAX_SECONDS,
            padding_seconds=RESPLIT_PADDING_SECONDS,
            cache=cache,
            max_clips=None,
            require_audio_quality=False,
        )
    _write_embedding_cache(context.project_root, cache)

    speaker_ref = _speaker_reference_vectors(
        clips_by_speaker, person_map, known, known_by_public
    )
    existing_speaker_for_person = _existing_speaker_for_person(person_map)

    suspects = select_suspect_speakers(clips_by_speaker, params)
    candidates: list[CandidatePerson] = []
    residue_clusters: list[ResidueCluster] = []
    for speaker_id in suspects:
        clips = clips_by_speaker[speaker_id]
        assigned = _resolve_known(person_map.get(speaker_id), known, known_by_public)
        track_candidates, residual_clips = _candidate_persons(
            speaker_id,
            clips,
            assigned,
            known,
            existing_speaker_for_person,
            params,
        )
        candidates.extend(track_candidates)
        residue_clusters.extend(
            _residue_clusters(
                speaker_id,
                residual_clips,
                assigned,
                known,
                speaker_ref,
                params,
            )
        )
    return TrackResplitPlan(
        context.project_root,
        context.provider,
        context.model,
        params,
        len(known),
        tuple(suspects),
        tuple(candidates),
        tuple(residue_clusters),
    )


def select_suspect_speakers(
    clips_by_speaker: dict[int, list[SpeakerClusterClip]],
    params: ResplitParams,
) -> list[int]:
    """Return tracks large enough to plausibly hide more than one speaker."""
    return sorted(
        speaker_id
        for speaker_id, clips in clips_by_speaker.items()
        if len(clips) >= params.min_suspect_sentences
    )


def _candidate_persons(
    speaker_id: int,
    clips: list[SpeakerClusterClip],
    assigned: _KnownSpeakerVector | None,
    known: dict[int, _KnownSpeakerVector],
    existing_speaker_for_person: dict[str, int],
    params: ResplitParams,
) -> tuple[list[CandidatePerson], list[SpeakerClusterClip]]:
    """Group clips by their best library match and evaluate each group for promotion.

    Returns the per-person candidate groups plus the clips that were *not* claimed by
    any candidate group (the residual pool fed to residue detection).
    """
    assigned_public = assigned.person_public_id if assigned else None
    by_person: dict[int, list[SpeakerClusterClip]] = defaultdict(list)
    claimed: set[int] = set()
    for clip_index, clip in enumerate(clips):
        ranked = _ranked_matches(clip.vector, known, limit=1)
        if not ranked:
            continue
        best = ranked[0]
        if best.person_public_id == assigned_public:
            continue
        if best.score < params.candidate_floor:
            continue
        by_person[best.person_id].append(clip)
        claimed.add(clip_index)

    candidates: list[CandidatePerson] = []
    for person_id, group in by_person.items():
        person = known[person_id]
        centroid = _normalize(_mean_vector([clip.vector for clip in group]))
        centroid_score = _cosine(centroid, person.vector)
        assigned_score = _cosine(centroid, assigned.vector) if assigned else 0.0
        lead = centroid_score - assigned_score
        total_seconds = sum(_clip_duration_ms(clip) for clip in group) / 1000
        decision = _promotion_decision(
            group, centroid_score, lead, total_seconds, params
        )
        candidates.append(
            CandidatePerson(
                speaker_id,
                person_id,
                person.person_public_id,
                person.name,
                centroid_score,
                assigned_score,
                lead,
                total_seconds,
                existing_speaker_for_person.get(person.person_public_id),
                decision,
                tuple(_to_resplit_sentence(clip) for clip in group),
            )
        )

    promoted_indices = {
        clip_index
        for clip_index, clip in enumerate(clips)
        if clip_index in claimed and _clip_is_promoted(clip, candidates)
    }
    residual = [
        clip for clip_index, clip in enumerate(clips) if clip_index not in promoted_indices
    ]
    return candidates, residual


def _promotion_decision(
    group: list[SpeakerClusterClip],
    centroid_score: float,
    lead: float,
    total_seconds: float,
    params: ResplitParams,
) -> str:
    """Classify one candidate person group against the promotion gates."""
    if len(group) < params.min_group_sentences or total_seconds < params.min_group_seconds:
        return "too-few"
    if centroid_score < params.promote_centroid_threshold:
        return "below-centroid"
    if lead < params.promote_lead_margin:
        return "below-lead"
    return "promote"


def _clip_is_promoted(
    clip: SpeakerClusterClip, candidates: list[CandidatePerson]
) -> bool:
    """Return whether a clip belongs to a candidate group that was promoted."""
    identity = (clip.sentence_id, clip.begin_time_ms, clip.end_time_ms)
    for candidate in candidates:
        if candidate.decision != "promote":
            continue
        for sentence in candidate.sentences:
            if (sentence.sentence_id, sentence.begin_time_ms, sentence.end_time_ms) == identity:
                return True
    return False


def _residue_clusters(
    speaker_id: int,
    clips: list[SpeakerClusterClip],
    assigned: _KnownSpeakerVector | None,
    known: dict[int, _KnownSpeakerVector],
    speaker_ref: dict[int, list[float]],
    params: ResplitParams,
) -> list[ResidueCluster]:
    """Find coherent out-of-library clusters among residual clips.

    A clip enters the residual pool only when no library person matches it (best
    score below the floor). We then cluster those clips and keep a cluster only when
    its *centroid* also fails to match any library person — averaging denoises, so a
    centroid that snaps back onto the dominant library identity (e.g. the track's own
    person in poor audio) is rejected here rather than wrongly pulled into a bucket.
    """
    unmatched = [
        clip
        for clip in clips
        if _best_library_score(clip.vector, known) < params.residue_match_floor
    ]
    if len(unmatched) < params.min_group_sentences:
        return []
    results: list[ResidueCluster] = []
    for component in _connected_components(unmatched, params.residue_cluster_threshold):
        total_seconds = sum(_clip_duration_ms(clip) for clip in component) / 1000
        if (
            len(component) < params.min_group_sentences
            or total_seconds < params.min_group_seconds
        ):
            continue
        centroid = _normalize(_mean_vector([clip.vector for clip in component]))
        assigned_score = _cosine(centroid, assigned.vector) if assigned else None
        best_name, best_score = _best_library_match(centroid, known)
        if best_score is not None and best_score >= params.residue_match_floor:
            # The denoised centroid actually matches a library person; not residue.
            continue
        merge_id, merge_score = _nearest_other_speaker(centroid, speaker_id, speaker_ref)
        if merge_id is not None and merge_score >= params.merge_threshold:
            decision = "merge-suggested"
        else:
            decision = "unknown-bucket"
            merge_id, merge_score = None, None
        results.append(
            ResidueCluster(
                speaker_id,
                assigned_score,
                best_name,
                best_score,
                merge_id,
                merge_score,
                total_seconds,
                decision,
                tuple(_to_resplit_sentence(clip) for clip in component),
            )
        )
    return results


def _speaker_reference_vectors(
    clips_by_speaker: dict[int, list[SpeakerClusterClip]],
    person_map: dict[int, int | str],
    known: dict[int, _KnownSpeakerVector],
    known_by_public: dict[str, _KnownSpeakerVector],
) -> dict[int, list[float]]:
    """Return one representative vector per project speaker.

    Prefer the speaker's clean library vector when person-mapped; otherwise fall back
    to the speaker's own track centroid.
    """
    refs: dict[int, list[float]] = {}
    for speaker_id, clips in clips_by_speaker.items():
        mapped = _resolve_known(person_map.get(speaker_id), known, known_by_public)
        if mapped is not None:
            refs[speaker_id] = mapped.vector
        elif clips:
            refs[speaker_id] = _normalize(_mean_vector([clip.vector for clip in clips]))
    return refs


def _existing_speaker_for_person(person_map: dict[int, int | str]) -> dict[str, int]:
    """Map a voiceprint person public id to the project speaker already holding it."""
    mapping: dict[str, int] = {}
    for speaker_id, value in person_map.items():
        if isinstance(value, str) and value:
            mapping.setdefault(value, speaker_id)
    return mapping


def _resolve_known(
    value: int | str | None,
    known: dict[int, _KnownSpeakerVector],
    known_by_public: dict[str, _KnownSpeakerVector],
) -> _KnownSpeakerVector | None:
    """Resolve a person_map value (public id string or legacy int) to a known vector."""
    if value is None:
        return None
    if isinstance(value, str):
        return known_by_public.get(value)
    return known.get(int(value))


def _nearest_other_speaker(
    vector: list[float],
    speaker_id: int,
    speaker_ref: dict[int, list[float]],
) -> tuple[int | None, float | None]:
    """Return the closest project speaker other than ``speaker_id``."""
    candidates = [
        (other_id, _cosine(vector, ref))
        for other_id, ref in speaker_ref.items()
        if other_id != speaker_id
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def _best_library_match(
    vector: list[float], known: dict[int, _KnownSpeakerVector]
) -> tuple[str | None, float | None]:
    """Return the best library person name and score for a vector."""
    ranked = _ranked_matches(vector, known, limit=1)
    if not ranked:
        return None, None
    return ranked[0].name, ranked[0].score


def _best_library_score(vector: list[float], known: dict[int, _KnownSpeakerVector]) -> float:
    """Return the best library cosine score for a vector (0.0 when library empty)."""
    ranked = _ranked_matches(vector, known, limit=1)
    return ranked[0].score if ranked else 0.0


def resplit_plan_payload(plan: TrackResplitPlan) -> dict[str, object]:
    """Return a JSON-safe payload for a re-split plan (analysis output / audit)."""
    return {
        "project_root": str(plan.project_root),
        "provider": plan.provider,
        "model": plan.model,
        "library_size": plan.library_size,
        "params": {
            "candidate_floor": plan.params.candidate_floor,
            "promote_centroid_threshold": plan.params.promote_centroid_threshold,
            "promote_lead_margin": plan.params.promote_lead_margin,
            "residue_match_floor": plan.params.residue_match_floor,
            "residue_cluster_threshold": plan.params.residue_cluster_threshold,
            "merge_threshold": plan.params.merge_threshold,
            "min_group_sentences": plan.params.min_group_sentences,
            "min_group_seconds": plan.params.min_group_seconds,
            "min_suspect_sentences": plan.params.min_suspect_sentences,
        },
        "suspect_speaker_ids": list(plan.suspect_speaker_ids),
        "candidates": [_candidate_payload(item) for item in plan.candidates],
        "residue_clusters": [_residue_payload(item) for item in plan.residue_clusters],
    }


def _candidate_payload(candidate: CandidatePerson) -> dict[str, object]:
    """Return a JSON-safe payload for one candidate person group."""
    return {
        "source_speaker_id": candidate.source_speaker_id,
        "person_id": candidate.person_id,
        "person_public_id": candidate.person_public_id,
        "name": candidate.name,
        "centroid_score": candidate.centroid_score,
        "assigned_score": candidate.assigned_score,
        "lead": candidate.lead,
        "total_seconds": candidate.total_seconds,
        "existing_speaker_id": candidate.existing_speaker_id,
        "decision": candidate.decision,
        "sentences": [_sentence_payload(s) for s in candidate.sentences],
    }


def _residue_payload(cluster: ResidueCluster) -> dict[str, object]:
    """Return a JSON-safe payload for one residue cluster."""
    return {
        "source_speaker_id": cluster.source_speaker_id,
        "assigned_score": cluster.assigned_score,
        "best_library_name": cluster.best_library_name,
        "best_library_score": cluster.best_library_score,
        "merge_target_speaker_id": cluster.merge_target_speaker_id,
        "merge_score": cluster.merge_score,
        "total_seconds": cluster.total_seconds,
        "decision": cluster.decision,
        "sentences": [_sentence_payload(s) for s in cluster.sentences],
    }


def _sentence_payload(sentence: ResplitSentence) -> dict[str, object]:
    """Return a JSON-safe payload for one move-target sentence."""
    return {
        "sentence_id": sentence.sentence_id,
        "begin_time_ms": sentence.begin_time_ms,
        "end_time_ms": sentence.end_time_ms,
        "text": sentence.text,
    }


def _clip_duration_ms(clip: SpeakerClusterClip) -> int:
    """Return a clip's sentence duration in milliseconds."""
    return max(0, clip.end_time_ms - clip.begin_time_ms)


def _to_resplit_sentence(clip: SpeakerClusterClip) -> ResplitSentence:
    """Convert an embedded clip into a move-target sentence identity."""
    return ResplitSentence(
        clip.sentence_id, clip.begin_time_ms, clip.end_time_ms, clip.text
    )


__all__ = [
    "CandidatePerson",
    "ResidueCluster",
    "ResplitParams",
    "ResplitSentence",
    "TrackResplitPlan",
    "analyze_project_resplit",
    "resplit_plan_payload",
    "select_suspect_speakers",
]
