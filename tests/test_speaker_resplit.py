"""Tests for under-split re-split analysis and its stabilization wiring."""

from __future__ import annotations

import math

from app.speaker_cluster_quality import SpeakerClusterClip
from app.speaker_matching import _KnownSpeakerVector
from app.speaker_resplit import (
    CandidatePerson,
    ResidueCluster,
    ResplitParams,
    ResplitSentence,
    TrackResplitPlan,
    _candidate_persons,
    _existing_speaker_for_person,
    _promotion_decision,
    _residue_clusters,
    select_suspect_speakers,
)
from app.speaker_stabilization import _resplit_reassignments


def _unit(values: list[float]) -> list[float]:
    """Return an L2-normalized vector."""
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _clip(vector: list[float], *, sid: int, begin: int, end: int) -> SpeakerClusterClip:
    """Build a minimal embedded clip for engine tests."""
    return SpeakerClusterClip(
        speaker_id=1,
        index=sid,
        sentence_id=sid,
        begin_time_ms=begin,
        end_time_ms=end,
        text=f"sentence-{sid}",
        vector=vector,
    )


def _person(person_id: int, vector: list[float], public: str) -> _KnownSpeakerVector:
    """Build a known library person vector."""
    return _KnownSpeakerVector(person_id, f"person-{person_id}", _unit(vector), public)


A = [1.0, 0.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0, 0.0]
C = [0.0, 0.0, 1.0, 0.0]


def test_select_suspect_speakers_filters_by_size() -> None:
    """Only tracks with enough sentences are examined."""
    clips = {
        0: [_clip(A, sid=i, begin=i * 1000, end=i * 1000 + 800) for i in range(8)],
        1: [_clip(A, sid=100 + i, begin=i, end=i + 1) for i in range(3)],
    }
    params = ResplitParams(min_suspect_sentences=6)
    assert select_suspect_speakers(clips, params) == [0]


def test_promotion_decision_gates() -> None:
    """Each promotion gate maps to its own decision label."""
    group = [_clip(B, sid=i, begin=0, end=8000) for i in range(2)]
    params = ResplitParams(
        promote_centroid_threshold=0.62, promote_lead_margin=0.10, min_group_seconds=6.0
    )
    assert _promotion_decision(group, 0.80, 0.30, 16.0, False, params) == "promote"
    assert _promotion_decision(group, 0.50, 0.30, 16.0, False, params) == "below-centroid"
    assert _promotion_decision(group, 0.80, 0.05, 16.0, False, params) == "below-lead"
    assert _promotion_decision(group[:1], 0.80, 0.30, 16.0, False, params) == "too-few"
    assert _promotion_decision(group, 0.80, 0.30, 2.0, False, params) == "too-few"
    # The dominant guard fires first: a group that IS the track's primary speaker is
    # never promoted, even with otherwise-passing scores.
    assert _promotion_decision(group, 0.80, 0.30, 16.0, True, params) == "dominant"


def test_candidate_persons_promotes_other_library_person() -> None:
    """A minority group matching a different library person forms a promotable group.

    Realistic under-split shape: the track is mostly its assigned person (person 1) with
    a minority intruder (person 2). The intruder is the splinter to extract; the assigned
    majority is excluded up front and stays put."""
    known = {1: _person(1, A, "vpp-a"), 2: _person(2, B, "vpp-b")}
    assigned = known[1]  # the track currently belongs to person 1
    main = [_clip(A, sid=i, begin=i * 9000, end=i * 9000 + 9000) for i in range(6)]
    intruder = [_clip(B, sid=50 + i, begin=i * 9000, end=i * 9000 + 9000) for i in range(3)]
    params = ResplitParams(min_group_sentences=2, min_group_seconds=6.0)

    candidates, residual = _candidate_persons(
        speaker_id=1,
        clips=main + intruder,
        assigned=assigned,
        known=known,
        existing_speaker_for_person={},
        params=params,
    )

    assert len(candidates) == 1  # the assigned person is excluded up front
    candidate = candidates[0]
    assert candidate.person_id == 2
    assert candidate.decision == "promote"  # 3/9 minority => extract
    assert candidate.centroid_score > 0.62
    assert candidate.lead > 0.10
    assert candidate.existing_speaker_id is None
    assert len(residual) == 6  # promoted intruder clips leave; the assigned majority stays


def test_candidate_persons_keeps_dominant_group_when_unassigned() -> None:
    """On the relaxed run gate (assigned=None) the track's own dominant voice must not
    be promoted into a duplicate id, while a minority intruder still is.

    Without `assigned`, the up-front "skip the assigned person" exclusion is a no-op, so
    the track's main speaker reaches the candidate grouping. The dominant guard is what
    stops it from being minted+moved wholesale (which would empty the track and
    invalidate its voiceprint samples)."""
    known = {1: _person(1, A, "vpp-a"), 2: _person(2, B, "vpp-b")}
    main = [_clip(A, sid=i, begin=i * 9000, end=i * 9000 + 9000) for i in range(8)]
    intruder = [_clip(B, sid=100 + i, begin=i * 9000, end=i * 9000 + 9000) for i in range(2)]
    params = ResplitParams(min_group_sentences=2, min_group_seconds=6.0)

    candidates, residual = _candidate_persons(
        speaker_id=1,
        clips=main + intruder,
        assigned=None,  # relaxed gate: no accepted aggregate match for this track
        known=known,
        existing_speaker_for_person={},
        params=params,
    )

    by_person = {c.person_id: c for c in candidates}
    assert by_person[1].decision == "dominant"  # 8/10 of the track => source identity
    assert by_person[2].decision == "promote"  # 2/10 minority intruder => extract
    # The dominant group is not promoted, so its clips stay in the residual pool.
    assert len(residual) == 8


def test_candidate_persons_noop_on_coherent_track() -> None:
    """A track whose sentences all match its own identity yields nothing to move."""
    known = {1: _person(1, A, "vpp-a"), 2: _person(2, B, "vpp-b")}
    assigned = known[1]
    clips = [_clip(A, sid=i, begin=i * 9000, end=i * 9000 + 9000) for i in range(4)]
    params = ResplitParams(min_group_sentences=2)

    candidates, residual = _candidate_persons(
        1, clips, assigned, known, {}, params
    )

    assert candidates == []
    assert len(residual) == 4  # everything stays put


def test_residue_clusters_buckets_out_of_library_group() -> None:
    """A coherent group matching no library person becomes an unknown bucket."""
    known = {1: _person(1, A, "vpp-a"), 2: _person(2, B, "vpp-b")}
    clips = [_clip(C, sid=i, begin=i * 7000, end=i * 7000 + 7000) for i in range(2)]
    params = ResplitParams(
        residue_match_floor=0.40,
        residue_cluster_threshold=0.62,
        min_group_sentences=2,
        min_group_seconds=6.0,
    )

    clusters = _residue_clusters(
        speaker_id=1,
        clips=clips,
        assigned=known[1],
        known=known,
        speaker_ref={1: known[1].vector},  # only the source track exists
        params=params,
        track_clip_count=10,  # these 2 are a minority of the track
    )

    assert len(clusters) == 1
    assert clusters[0].decision == "unknown-bucket"
    assert clusters[0].merge_target_speaker_id is None
    assert len(clusters[0].sentences) == 2


def test_residue_clusters_skips_dominant_cluster() -> None:
    """A coherent out-of-library group that is the track majority is its primary
    speaker, not an outlier — never bucket the majority."""
    known = {1: _person(1, A, "vpp-a")}
    clips = [_clip(C, sid=i, begin=i * 7000, end=i * 7000 + 7000) for i in range(3)]
    params = ResplitParams(
        residue_match_floor=0.40,
        residue_cluster_threshold=0.62,
        min_group_sentences=2,
        min_group_seconds=6.0,
        dominant_track_fraction=0.5,
    )

    clusters = _residue_clusters(
        speaker_id=1,
        clips=clips,
        assigned=None,
        known=known,
        speaker_ref={},
        params=params,
        track_clip_count=3,  # the 3-clip cluster IS the whole track => skip
    )

    assert clusters == []


def test_residue_clusters_reject_when_centroid_snaps_to_library() -> None:
    """Clips below the floor individually but whose denoised centroid matches a known
    person are left alone, not pulled into a bucket."""
    known = {1: _person(1, A, "vpp-a")}
    # Two clips at cosine 0.45 to A, spread in azimuth so they cluster yet their mean
    # normalizes back above the 0.5 floor toward A.
    sin = math.sqrt(1 - 0.45**2)
    clip1 = [0.45, sin * math.cos(math.radians(30)), sin * math.sin(math.radians(30)), 0.0]
    clip2 = [0.45, sin * math.cos(math.radians(30)), -sin * math.sin(math.radians(30)), 0.0]
    clips = [
        _clip(clip1, sid=0, begin=0, end=7000),
        _clip(clip2, sid=1, begin=7000, end=14000),
    ]
    params = ResplitParams(
        residue_match_floor=0.50,
        residue_cluster_threshold=0.55,
        min_group_sentences=2,
        min_group_seconds=0.0,
    )

    clusters = _residue_clusters(
        1, clips, known[1], known, {1: known[1].vector}, params, track_clip_count=10
    )

    assert clusters == []  # centroid resembles person A => not residue


def test_residue_clusters_empty_library_is_noop() -> None:
    """With no library, every clip looks 'unmatched' — must not bucket a normal track."""
    clips = [_clip(C, sid=i, begin=i * 7000, end=i * 7000 + 7000) for i in range(3)]
    params = ResplitParams(min_group_sentences=2, min_group_seconds=0.0)

    clusters = _residue_clusters(
        speaker_id=1,
        clips=clips,
        assigned=None,
        known={},
        speaker_ref={},
        params=params,
        track_clip_count=3,
    )

    assert clusters == []


def test_existing_speaker_for_person_resolves_legacy_int_and_public() -> None:
    """Both legacy integer person ids and public-id strings key on the public id."""
    known = {7: _person(7, A, "vpp-a"), 9: _person(9, B, "vpp-b")}
    known_by_public = {"vpp-a": known[7], "vpp-b": known[9]}
    # speaker 0 mapped by legacy int (7), speaker 1 mapped by public id string.
    person_map: dict[int, int | str] = {0: 7, 1: "vpp-b"}

    mapping = _existing_speaker_for_person(person_map, known, known_by_public)

    assert mapping == {"vpp-a": 0, "vpp-b": 1}


def test_candidate_persons_routes_to_existing_track_for_mapped_person() -> None:
    """A promotion whose person already has a track reports that track id, not None."""
    known = {1: _person(1, A, "vpp-a"), 2: _person(2, B, "vpp-b")}
    assigned = known[1]
    clips = [_clip(B, sid=i, begin=i * 9000, end=i * 9000 + 9000) for i in range(3)]
    params = ResplitParams(min_group_sentences=2, min_group_seconds=6.0)

    candidates, _ = _candidate_persons(
        speaker_id=1,
        clips=clips,
        assigned=assigned,
        known=known,
        existing_speaker_for_person={"vpp-b": 5},  # person 2 already owns track 5
        params=params,
    )

    assert len(candidates) == 1
    assert candidates[0].existing_speaker_id == 5


def _sentences(count: int, *, base: int) -> tuple[ResplitSentence, ...]:
    """Build a tuple of distinct move-target sentences."""
    return tuple(
        ResplitSentence(
            sentence_id=base + i,
            begin_time_ms=(base + i) * 1000,
            end_time_ms=(base + i) * 1000 + 5000,
            text=f"s{base + i}",
        )
        for i in range(count)
    )


def _plan(
    candidates: tuple[CandidatePerson, ...],
    residue: tuple[ResidueCluster, ...],
) -> TrackResplitPlan:
    """Wrap candidates/residue in a minimal analysis plan."""
    from pathlib import Path

    return TrackResplitPlan(
        project_root=Path("."),
        provider="fake",
        model="fake",
        params=ResplitParams(),
        library_size=3,
        suspect_speaker_ids=(1,),
        candidates=candidates,
        residue_clusters=residue,
    )


def _candidate(
    *, name: str, public: str, existing: int | None, sentences: tuple[ResplitSentence, ...]
) -> CandidatePerson:
    """Build a promoted candidate person group."""
    return CandidatePerson(
        source_speaker_id=1,
        person_id=99,
        person_public_id=public,
        name=name,
        centroid_score=0.8,
        assigned_score=0.4,
        lead=0.4,
        total_seconds=15.0,
        existing_speaker_id=existing,
        decision="promote",
        sentences=sentences,
    )


def test_resplit_reassignments_mints_new_track_and_seeds() -> None:
    """A promotion for a library person without a track mints and seeds a new id."""
    candidate = _candidate(
        name="Shu", public="vpp-shu", existing=None, sentences=_sentences(2, base=10)
    )
    plan = _plan((candidate,), ())

    apply_plan = _resplit_reassignments(plan, existing_speaker_ids={0, 1})

    assert apply_plan.minted_speaker_ids == (2,)
    assert apply_plan.seed_names == {2: "Shu"}
    assert apply_plan.seed_public_ids == {2: "vpp-shu"}
    assert apply_plan.unknown_bucket_id is None
    assert {spec.new_speaker_id for spec in apply_plan.specs} == {2}
    assert all(spec.original_speaker_id == 1 for spec in apply_plan.specs)


def test_resplit_reassignments_routes_to_existing_track_without_seed() -> None:
    """A promotion for a person who already has a track routes there and seeds nothing."""
    candidate = _candidate(
        name="Bob", public="vpp-bob", existing=0, sentences=_sentences(3, base=20)
    )
    plan = _plan((candidate,), ())

    apply_plan = _resplit_reassignments(plan, existing_speaker_ids={0, 1})

    assert apply_plan.minted_speaker_ids == ()
    assert apply_plan.seed_names == {}
    assert {spec.new_speaker_id for spec in apply_plan.specs} == {0}


def test_resplit_reassignments_collapses_residue_into_one_anonymous_bucket() -> None:
    """All residue clusters share one minted, unseeded unknown bucket id."""
    residue = (
        ResidueCluster(
            source_speaker_id=1,
            assigned_score=0.2,
            best_library_name=None,
            best_library_score=0.1,
            merge_target_speaker_id=None,
            merge_score=None,
            total_seconds=12.0,
            decision="unknown-bucket",
            sentences=_sentences(2, base=30),
        ),
        ResidueCluster(
            source_speaker_id=1,
            assigned_score=0.2,
            best_library_name=None,
            best_library_score=0.1,
            merge_target_speaker_id=None,
            merge_score=None,
            total_seconds=10.0,
            decision="unknown-bucket",
            sentences=_sentences(2, base=40),
        ),
    )
    plan = _plan((), residue)

    apply_plan = _resplit_reassignments(plan, existing_speaker_ids={0, 1})

    assert apply_plan.unknown_bucket_id == 2
    assert apply_plan.minted_speaker_ids == (2,)
    assert apply_plan.seed_names == {}  # the bucket stays anonymous
    assert {spec.new_speaker_id for spec in apply_plan.specs} == {2}
    assert len(apply_plan.specs) == 4


def test_resplit_reassignments_dedupes_person_and_separates_bucket_id() -> None:
    """One new id per distinct person; the unknown bucket gets its own fresh id."""
    promo_a = _candidate(
        name="Shu", public="vpp-shu", existing=None, sentences=_sentences(2, base=10)
    )
    promo_a_again = _candidate(
        name="Shu", public="vpp-shu", existing=None, sentences=_sentences(2, base=50)
    )
    residue = (
        ResidueCluster(
            source_speaker_id=1,
            assigned_score=0.2,
            best_library_name=None,
            best_library_score=0.1,
            merge_target_speaker_id=None,
            merge_score=None,
            total_seconds=12.0,
            decision="unknown-bucket",
            sentences=_sentences(2, base=60),
        ),
    )
    plan = _plan((promo_a, promo_a_again), residue)

    apply_plan = _resplit_reassignments(plan, existing_speaker_ids={0, 1})

    # Same person -> single minted id (2); unknown bucket -> separate id (3).
    assert apply_plan.seed_names == {2: "Shu"}
    assert apply_plan.unknown_bucket_id == 3
    assert apply_plan.minted_speaker_ids == (2, 3)
