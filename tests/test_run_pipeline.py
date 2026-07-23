"""Tests for the artifact-gated project run pipeline."""

from __future__ import annotations

from pathlib import Path


from app.commands.project import (
    _identity_confidence_label,
    _run_polish_skip_reason,
    _run_summary_skip_reason,
)
from app.speaker_sample_matching import (
    SpeakerSampleMatchReport,
    SpeakerSampleMatchSummary,
)
from app.speaker_stabilization import (
    SpeakerStabilizationIteration,
    SpeakerStabilizationSummary,
)
from app.core.run_pipeline import (
    RunStage,
    execute_run_pipeline,
    pipeline_step_descriptions,
)
from app.project_manager import create_project, load_manifest, save_manifest


def test_pipeline_runs_stages_in_order_with_step_numbering() -> None:
    """Stages execute in plan order and see plan-derived step numbers."""
    seen: list[tuple[str, int, int]] = []

    def _make(key: str) -> RunStage:
        return RunStage(
            key=key,
            description=key,
            execute=lambda ctx, key=key: seen.append(
                (key, ctx.step_index, ctx.step_total)
            ),
        )

    stages = [
        _make("one"),
        RunStage(
            key="wide",
            description="wide",
            step_span=3,
            execute=lambda ctx: seen.append(("wide", ctx.step_index, ctx.step_total)),
        ),
        _make("last"),
    ]

    skipped = execute_run_pipeline(stages, None)

    assert skipped == {}
    assert seen == [("one", 1, 5), ("wide", 2, 5), ("last", 5, 5)]


def test_pipeline_skips_satisfied_stages_and_reports_reason() -> None:
    """A satisfied stage is skipped, later numbering is unaffected."""
    executed: list[str] = []
    stages = [
        RunStage(
            key="a", description="a", execute=lambda ctx: executed.append("a")
        ),
        RunStage(
            key="b",
            description="b",
            execute=lambda ctx: executed.append("b"),
            satisfied=lambda: "already there",
        ),
        RunStage(
            key="c",
            description="c",
            execute=lambda ctx: executed.append("c"),
            satisfied=lambda: None,
        ),
    ]

    skipped = execute_run_pipeline(stages, None)

    assert executed == ["a", "c"]
    assert skipped == {"b": "already there"}


def test_pipeline_step_descriptions_expand_sub_steps() -> None:
    """Multi-step stages contribute their sub-descriptions."""
    stages = [
        RunStage(key="a", description="A", execute=lambda ctx: None),
        RunStage(
            key="b",
            description="B",
            execute=lambda ctx: None,
            step_span=2,
            sub_descriptions=("B1", "B2"),
        ),
        RunStage(key="c", description="C", execute=lambda ctx: None, step_span=2),
    ]

    assert pipeline_step_descriptions(stages) == ("A", "B1", "B2", "C", "C")


def test_polish_skip_reason_follows_runtime_state(tmp_path: Path) -> None:
    """Accepted/pending/no-change polish states gate the run polish stage."""
    project_dir = _make_project(tmp_path)
    assert _run_polish_skip_reason(project_dir) is None

    manifest = load_manifest(project_dir)
    manifest.runtime = {"polish": {"status": "no_changes"}}
    save_manifest(project_dir, manifest)
    assert _run_polish_skip_reason(project_dir) == "previous polish found no changes"

    manifest = load_manifest(project_dir)
    manifest.runtime = {
        "polish": {"status": "proposal_ready", "proposal_json": "corrections/p.json"}
    }
    save_manifest(project_dir, manifest)
    # Proposal file missing -> stale state must NOT skip.
    assert _run_polish_skip_reason(project_dir) is None
    proposal = project_dir / "corrections" / "p.json"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("{}", encoding="utf-8")
    assert "pending review" in str(_run_polish_skip_reason(project_dir))

    manifest = load_manifest(project_dir)
    manifest.runtime = {"polish": {"status": "accepted"}}
    save_manifest(project_dir, manifest)
    # Accepted but corrected sentences missing -> re-run.
    assert _run_polish_skip_reason(project_dir) is None
    corrected = project_dir / "asr" / "sentences_corrected.json"
    corrected.parent.mkdir(parents=True, exist_ok=True)
    corrected.write_text("{}", encoding="utf-8")
    assert _run_polish_skip_reason(project_dir) == "polish already accepted"

    manifest = load_manifest(project_dir)
    manifest.runtime = {"polish": {"status": "failed"}}
    save_manifest(project_dir, manifest)
    assert _run_polish_skip_reason(project_dir) is None


def test_summary_skip_reason_gates_on_artifact(tmp_path: Path) -> None:
    """The summary stage is skipped only when meeting_summary.md exists."""
    project_dir = _make_project(tmp_path)
    assert _run_summary_skip_reason(project_dir) is None
    summary_path = project_dir / "exports" / "meeting_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("# memo", encoding="utf-8")
    assert _run_summary_skip_reason(project_dir) == "meeting summary exists"


def test_identity_confidence_label_summarizes_final_diagnostics(
    tmp_path: Path,
) -> None:
    """The run summary compresses sentence identity agreement into one line."""
    assert _identity_confidence_label(None) is None
    empty = SpeakerStabilizationSummary(iterations=())
    assert _identity_confidence_label(empty) is None

    stabilization = _stabilization_with_counts(
        [
            {"identity-ok": 18, "identity-conflict": 1},
            {"identity-ok": 6, "identity-ambiguous": 2, "no-assignment": 4},
        ]
    )
    label = _identity_confidence_label(stabilization)

    assert label == "24/27 sentences consistent (89%), 1 conflict, 2 ambiguous"

    unjudged = _stabilization_with_counts([{"no-assignment": 5}])
    assert _identity_confidence_label(unjudged) is None


def _stabilization_with_counts(
    status_counts_list: list[dict[str, int]],
) -> SpeakerStabilizationSummary:
    """Build a stabilization summary with the given per-speaker status counts."""
    reports = [
        SpeakerSampleMatchReport(
            speaker_id=index,
            label=f"Speaker {index}",
            assigned_person_id=None,
            assigned_name=None,
            sample_count=sum(counts.values()),
            status_counts=counts,
            samples=[],
        )
        for index, counts in enumerate(status_counts_list)
    ]
    sample_summary = SpeakerSampleMatchSummary(
        report_path=Path("speakers/speaker_sample_matches.json"),
        provider="local-speechbrain",
        model="ecapa",
        threshold=0.45,
        conflict_margin=0.08,
        ambiguous_margin=0.05,
        reports=reports,
        verdict="ok",
    )
    iteration = SpeakerStabilizationIteration(
        index=1,
        reassignments=(),
        apply_result=None,
        cluster_summary=None,  # type: ignore[arg-type] - unused by the label
        sample_summary=sample_summary,
    )
    return SpeakerStabilizationSummary(iterations=(iteration,))


def _make_project(tmp_path: Path) -> Path:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir
