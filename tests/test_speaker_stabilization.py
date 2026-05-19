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
from app.speaker_sample_matching import SpeakerSampleMatch, SpeakerSampleMatchReport, SpeakerSampleMatchSummary
from app.speaker_stabilization import _sentence_reassignments, stabilize_project_speakers


def test_sentence_reassignments_use_identity_conflict_with_project_target(tmp_path: Path) -> None:
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


def test_stabilize_project_speakers_applies_and_refreshes(monkeypatch, tmp_path: Path) -> None:
    """Stabilization should apply conflicts and refresh diagnostics for the next pass."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    calls: list[str] = []

    diagnostics = [
        (_cluster_summary(status="conflict", nearest_speaker_id=1), _sample_summary("identity-conflict", 2)),
        (_cluster_summary(status="ok", nearest_speaker_id=None), _sample_summary("identity-ok", None)),
        (_cluster_summary(status="ok", nearest_speaker_id=None), _sample_summary("identity-ok", None)),
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
    monkeypatch.setattr("app.speaker_stabilization.apply_project_sentence_reassignments", fake_apply)
    monkeypatch.setattr("app.speaker_stabilization.apply_project_speakers", lambda *args, **kwargs: calls.append("names"))

    summary = stabilize_project_speakers(project_dir, store_dir=None, model=None, iterations=2, sample_workers=3)

    assert calls == ["refresh", "apply:1", "names", "refresh", "refresh"]
    assert summary.reassignment_count == 1
    assert summary.final_match_summary is not None


def _sample_summary(status: str, best_other_person_id: int | None) -> SpeakerSampleMatchSummary:
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
            SpeakerSampleMatchReport(0, "Speaker A", 1, "Alice", 1, {status: 1}, [conflict]),
            SpeakerSampleMatchReport(1, "Speaker B", 2, "Bob", 0, {}, []),
        ],
        verdict="identity-conflict",
    )


def _cluster_summary(status: str, nearest_speaker_id: int | None) -> SpeakerClusterQualitySummary:
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
