"""Tests for automatic speaker stabilization."""

from __future__ import annotations

from pathlib import Path

from app.sentence_reassignment import SentenceReassignmentApplyResult
from app.speaker_cluster_quality import (
    SpeakerClusterQualitySummary,
    SpeakerClusterReport,
    SpeakerClusterSampleScore,
)
from app.speaker_labeling import SentenceReassignmentSpec
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary
from app.speaker_sample_matching import (
    SpeakerSampleMatch,
    SpeakerSampleMatchReport,
    SpeakerSampleMatchSummary,
)
from app.speaker_stabilization import (
    SpeakerStabilizationIteration,
    SpeakerStabilizationSummary,
    _sentence_reassignments,
    stabilize_project_speakers,
)


def test_sentence_reassignments_use_identity_conflict_with_project_target(
    tmp_path: Path,
) -> None:
    """A strong sample conflict should become a concrete sentence reassignment."""
    sample_summary = _sample_summary(status="identity-conflict", best_other_person_id=2)
    cluster_summary = _cluster_summary(status="conflict", nearest_speaker_id=1)

    reassignments = _sentence_reassignments(sample_summary, cluster_summary)

    assert reassignments == [
        SentenceReassignmentSpec(
            sentence_id=10,
            begin_time_ms=1000,
            end_time_ms=3000,
            new_speaker_id=1,
            original_speaker_id=0,
        )
    ]


def test_sentence_reassignments_skip_cluster_contradiction() -> None:
    """Cluster conflicts pointing elsewhere should block automatic movement."""
    sample_summary = _sample_summary(status="identity-conflict", best_other_person_id=2)
    cluster_summary = _cluster_summary(status="conflict", nearest_speaker_id=3)

    assert _sentence_reassignments(sample_summary, cluster_summary) == []


def test_stabilize_project_speakers_applies_and_refreshes(
    monkeypatch, tmp_path: Path
) -> None:
    """Stabilization should apply conflicts and refresh diagnostics for the next pass."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    calls: list[str] = []

    diagnostics = [
        (
            _cluster_summary(status="conflict", nearest_speaker_id=1),
            _sample_summary("identity-conflict", 2),
        ),
        (
            _cluster_summary(status="ok", nearest_speaker_id=None),
            _sample_summary("identity-ok", None),
        ),
        (
            _cluster_summary(status="ok", nearest_speaker_id=None),
            _sample_summary("identity-ok", None),
        ),
    ]

    def fake_refresh(*args, **kwargs):
        calls.append("refresh")
        return diagnostics.pop(0)

    def fake_apply(project_dir_arg, reassignments, **kwargs):
        calls.append(f"apply:{len(reassignments)}")
        return SentenceReassignmentApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_samples=(),
            match_summary=_match_summary(project_dir_arg),
            rematch_skipped_reason=None,
        )

    monkeypatch.setattr("app.speaker_stabilization._refresh_diagnostics", fake_refresh)
    monkeypatch.setattr(
        "app.speaker_stabilization.apply_project_sentence_reassignments", fake_apply
    )
    monkeypatch.setattr(
        "app.speaker_stabilization.apply_project_speakers",
        lambda *args, **kwargs: calls.append("names"),
    )

    summary = stabilize_project_speakers(
        project_dir,
        store_dir=None,
        model=None,
        iterations=2,
        sample_workers=3,
        resplit=False,  # this test isolates the iterative pass; resplit is covered separately
    )

    assert calls == ["refresh", "apply:1", "names", "refresh", "refresh"]
    assert summary.reassignment_count == 1
    assert summary.final_match_summary is not None


def test_apply_resplit_phase_rerenders_named_for_residue_only(
    monkeypatch, tmp_path: Path
) -> None:
    """A residue-only move (no promotions, no accepted rematch) must still re-render the
    named exports; otherwise transcript_named.txt / subtitle_named.srt keep the moved
    sentences under their old speaker even though sentences.json changed."""
    from types import SimpleNamespace

    from app.speaker_resplit import (
        ResidueCluster,
        ResplitParams,
        ResplitSentence,
        TrackResplitPlan,
    )
    from app.speaker_stabilization import _apply_resplit_phase

    plan = TrackResplitPlan(
        project_root=tmp_path,
        provider="fake",
        model="fake-model",
        params=ResplitParams(),
        library_size=2,
        suspect_speaker_ids=(0,),
        candidates=(),  # no promotions => no seeds
        residue_clusters=(
            ResidueCluster(
                source_speaker_id=0,
                assigned_score=None,
                best_library_name=None,
                best_library_score=None,
                merge_target_speaker_id=None,
                merge_score=None,
                total_seconds=3.0,
                decision="unknown-bucket",
                sentences=(
                    ResplitSentence(
                        sentence_id=7, begin_time_ms=1000, end_time_ms=4000, text="x"
                    ),
                ),
            ),
        ),
    )
    empty_match = SpeakerMatchSummary(
        match_path=tmp_path / "m.json",
        provider="fake",
        model="fake-model",
        threshold=0.75,
        matches=[],  # rematch accepts nobody => empty accepted_mapping
    )
    apply_calls: list[dict] = []

    monkeypatch.setattr(
        "app.speaker_stabilization.analyze_project_resplit", lambda *a, **k: plan
    )
    monkeypatch.setattr(
        "app.speaker_stabilization.load_transcript_result",
        lambda *a, **k: SimpleNamespace(detected_speakers=[0, 1]),
    )
    monkeypatch.setattr(
        "app.speaker_stabilization.apply_project_sentence_reassignments",
        lambda *a, **k: SentenceReassignmentApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_samples=(),
            match_summary=empty_match,
            rematch_skipped_reason=None,
        ),
    )
    monkeypatch.setattr(
        "app.speaker_stabilization.apply_project_speakers",
        lambda project_dir, mappings, **k: apply_calls.append(mappings),
    )
    monkeypatch.setattr("app.speaker_stabilization.safe_write_json", lambda *a, **k: None)

    plan_out, minted, summary_out = _apply_resplit_phase(
        tmp_path, store_dir=None, model=None, params=None, progress=None
    )

    # Exactly one re-render, with an empty patch (re-render from updated sentences + map).
    assert apply_calls == [{}]
    assert minted == 1  # the single unknown bucket
    assert summary_out is empty_match


def test_final_match_summary_falls_back_to_resplit_phase(tmp_path: Path) -> None:
    """A run that only re-split (no iterative reassignment) must still report the
    post-resplit speakers via the re-split phase's own rematch summary."""
    resplit_match = _match_summary(tmp_path)
    summary = SpeakerStabilizationSummary(
        iterations=(),  # iterations=0 path (unanchored track) or zero reassignments
        minted_speaker_count=1,
        resplit_match_summary=resplit_match,
    )

    assert summary.final_match_summary is resplit_match


def test_iteration_match_summary_supersedes_resplit_phase(tmp_path: Path) -> None:
    """An iteration that applied moves re-reads post-resplit artifacts, so its summary
    wins over the earlier re-split phase summary."""
    resplit_match = _match_summary(tmp_path / "resplit")
    iteration_match = _match_summary(tmp_path / "iteration")
    iteration = SpeakerStabilizationIteration(
        index=1,
        reassignments=(),
        apply_result=SentenceReassignmentApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_samples=(),
            match_summary=iteration_match,
            rematch_skipped_reason=None,
        ),
        cluster_summary=None,
        sample_summary=None,
    )
    summary = SpeakerStabilizationSummary(
        iterations=(iteration,),
        resplit_match_summary=resplit_match,
    )

    assert summary.final_match_summary is iteration_match


def _sample_summary(
    status: str, best_other_person_id: int | None
) -> SpeakerSampleMatchSummary:
    """Build a minimal sample-match summary with two assigned project speakers."""
    conflict = SpeakerSampleMatch(
        speaker_id=0,
        sentence_id=10,
        begin_time_ms=1000,
        end_time_ms=3000,
        text="需要移动的句子",
        assigned_person_id=1,
        assigned_name="Alice",
        assigned_score=0.2,
        best_person_id=best_other_person_id,
        best_name="Bob" if best_other_person_id is not None else None,
        best_score=0.8 if best_other_person_id is not None else None,
        best_other_person_id=best_other_person_id,
        best_other_name="Bob" if best_other_person_id is not None else None,
        best_other_score=0.8 if best_other_person_id is not None else None,
        margin_score=-0.6 if best_other_person_id is not None else None,
        status=status,
    )
    return SpeakerSampleMatchSummary(
        report_path=Path("speaker_sample_matches.json"),
        provider="fake",
        model="fake-model",
        threshold=0.45,
        conflict_margin=0.08,
        ambiguous_margin=0.05,
        reports=[
            SpeakerSampleMatchReport(
                0, "Speaker A", 1, "Alice", 1, {status: 1}, [conflict]
            ),
            SpeakerSampleMatchReport(1, "Speaker B", 2, "Bob", 0, {}, []),
        ],
        verdict="identity-conflict",
    )


def _cluster_summary(
    status: str, nearest_speaker_id: int | None
) -> SpeakerClusterQualitySummary:
    """Build a minimal cluster summary containing the same sentence identity."""
    sample = SpeakerClusterSampleScore(
        index=1,
        sentence_id=10,
        begin_time_ms=1000,
        end_time_ms=3000,
        text="需要移动的句子",
        centroid_score=0.2,
        nearest_speaker_id=nearest_speaker_id,
        nearest_score=0.8 if nearest_speaker_id is not None else None,
        margin_score=-0.6 if nearest_speaker_id is not None else None,
        status=status,
    )
    report = SpeakerClusterReport(
        speaker_id=0,
        label="Speaker A",
        segment_count=1,
        clip_count=1,
        centroid_mean=0.2,
        centroid_min=0.2,
        warning_clip_count=0,
        critical_clip_count=1 if status == "conflict" else 0,
        intra_mean=None,
        intra_min=None,
        component_count=1,
        component_sizes=[1],
        nearest_speaker_id=nearest_speaker_id,
        nearest_score=sample.nearest_score,
        status="mixed" if status == "conflict" else "ok",
        warnings=[],
        samples=[sample],
    )
    return SpeakerClusterQualitySummary(
        report_path=Path("speaker_cluster_quality.json"),
        provider="fake",
        model="fake-model",
        same_speaker_threshold=0.6,
        merge_speaker_threshold=0.62,
        warning_score=0.7,
        critical_score=0.6,
        score_all_segments=True,
        reports=[report],
        close_pairs=[],
        verdict="test",
    )


def _match_summary(project_dir: Path) -> SpeakerMatchSummary:
    """Build an accepted aggregate match summary for name refresh."""
    return SpeakerMatchSummary(
        match_path=project_dir / "speakers" / "speaker_matches.json",
        provider="fake",
        model="fake-model",
        threshold=0.75,
        matches=[
            SpeakerMatch(
                1,
                "Speaker B",
                "Bob",
                0.9,
                True,
                1,
                accepted_person_id=2,
                accepted_person_public_id="vpp-0000000000000002",
            )
        ],
    )
