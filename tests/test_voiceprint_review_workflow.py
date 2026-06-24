"""Tests for transactional Voiceprint Review workflow behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.presentation.tui import voiceprint_review_workflow
from app.voiceprint_embedding import VoiceprintEmbedSummary
from app.voiceprint_evaluation import (
    VoiceprintEvaluationSummary,
    VoiceprintProjectEvaluation,
    VoiceprintScoreChange,
)
from app.voiceprints import VoiceprintCaptureSummary, VoiceprintClip, VoiceprintSpeaker


def test_capture_quality_gate_confirms_then_excludes_low_quality_samples(
    monkeypatch, tmp_path: Path
) -> None:
    """Reviewed capture samples keep identity confirmation even when excluded."""
    store_dir = tmp_path / "voiceprints"
    db_path = store_dir / "voiceprints.sqlite"
    clip = VoiceprintClip(
        path=store_dir / "clips" / "p1" / "speaker_0" / "clip_001.wav",
        rel_path="p1/speaker_0/clip_001.wav",
        source_begin_time_ms=0,
        source_end_time_ms=1000,
        clip_begin_time_ms=0,
        clip_end_time_ms=1000,
        text="hello",
    )
    capture = VoiceprintCaptureSummary(
        store_dir,
        db_path,
        store_dir / "clips",
        [VoiceprintSpeaker(0, "Alice", None, None, [clip])],
        False,
    )
    monkeypatch.setattr(
        voiceprint_review_workflow,
        "list_voiceprint_samples",
        lambda _ref, _db: [
            SimpleNamespace(public_id="s1", clip_rel_path="p1/speaker_0/clip_001.wav")
        ],
    )
    monkeypatch.setattr(
        voiceprint_review_workflow,
        "analyze_voiceprint_quality",
        lambda **_kwargs: SimpleNamespace(
            people=[
                SimpleNamespace(
                    samples=[
                        SimpleNamespace(
                            sample_public_id="s1",
                            label="warning",
                        )
                    ]
                )
            ]
        ),
    )
    updates: list[tuple[str, str]] = []
    monkeypatch.setattr(
        voiceprint_review_workflow,
        "update_voiceprint_sample_status",
        lambda sample_id, status, _db: updates.append((sample_id, status)),
    )

    summary = voiceprint_review_workflow._apply_capture_quality_gate(
        capture, store_dir=store_dir, model="test-model"
    )

    assert summary.reviewed_sample_count == 1
    assert summary.excluded_sample_count == 1
    assert updates == [("s1", "verified-active"), ("s1", "verified-quarantined")]


def test_voiceprint_review_workflow_can_roll_back_pending_files(
    monkeypatch, tmp_path: Path
) -> None:
    """Rejecting a workflow should restore DB, project files, match file, and clips."""
    project_dir = tmp_path / "project"
    store_dir = tmp_path / "voiceprints"
    db_path = store_dir / "voiceprints.sqlite"
    clip_path = store_dir / "clips" / "project-1" / "speaker_0" / "clip_001.wav"
    manifest_path = project_dir / "project.json"
    match_path = project_dir / "speakers" / "speaker_matches.json"
    _write_file(db_path, "old db")
    _write_file(clip_path, "old clip")
    _write_file(manifest_path, "old manifest")
    _write_file(match_path, "old match")
    planned = _planned_capture(store_dir, db_path, clip_path)

    def fake_capture(project_dir: Path, **kwargs) -> VoiceprintCaptureSummary:
        _write_file(db_path, "new db")
        _write_file(clip_path, "new clip")
        _write_file(manifest_path, "new manifest")
        _write_file(match_path, "new match")
        return VoiceprintCaptureSummary(
            store_dir, db_path, store_dir / "clips", planned.speakers, False
        )

    monkeypatch.setattr(
        voiceprint_review_workflow, "persist_voiceprint_capture_selection", fake_capture
    )
    monkeypatch.setattr(
        voiceprint_review_workflow, "embed_voiceprint_samples", _fake_embed
    )
    monkeypatch.setattr(
        voiceprint_review_workflow,
        "_apply_capture_quality_gate",
        lambda *_args, **_kwargs: (
            voiceprint_review_workflow.VoiceprintQualityGateSummary()
        ),
    )
    monkeypatch.setattr(
        voiceprint_review_workflow, "evaluate_voiceprint_embedding", _fake_evaluation
    )

    summary = voiceprint_review_workflow.run_voiceprint_review_workflow(
        project_dir=project_dir,
        planned=planned,
        selected_clip_rel_paths=frozenset({planned.speakers[0].clips[0].rel_path}),
        store_dir=store_dir,
    )

    assert db_path.read_text(encoding="utf-8") == "new db"
    assert clip_path.read_text(encoding="utf-8") == "new clip"
    assert manifest_path.read_text(encoding="utf-8") == "new manifest"
    assert match_path.read_text(encoding="utf-8") == "new match"

    summary.transaction.rollback()

    assert db_path.read_text(encoding="utf-8") == "old db"
    assert clip_path.read_text(encoding="utf-8") == "old clip"
    assert manifest_path.read_text(encoding="utf-8") == "old manifest"
    assert match_path.read_text(encoding="utf-8") == "old match"
    assert not summary.transaction.backup_dir.exists()


def test_historical_risks_show_severity_project_id_and_review_command() -> None:
    """Historical voiceprint regressions should show warning and critical severity."""
    evaluation = VoiceprintEvaluationSummary(
        _fake_evaluation().current,
        (
            VoiceprintProjectEvaluation(
                Path("/projects/p-risk"),
                "p-risk",
                "历史风险项目",
                False,
                (
                    VoiceprintScoreChange(
                        0,
                        "Speaker A",
                        "骁程",
                        0.334,
                        "骁程",
                        0.334,
                        0.0,
                        "unchanged",
                        0.75,
                    ),
                    VoiceprintScoreChange(
                        3,
                        "Speaker D",
                        "敬悦",
                        0.734,
                        "武一",
                        0.802,
                        0.068,
                        "changed-best",
                        0.75,
                    ),
                    VoiceprintScoreChange(
                        4,
                        "Speaker E",
                        "米汤",
                        0.801,
                        "米汤",
                        0.700,
                        -0.101,
                        "declined",
                        0.75,
                    ),
                ),
            ),
            VoiceprintProjectEvaluation(
                Path("/projects/p-warn"),
                "p-warn",
                "轻微下降项目",
                False,
                (
                    VoiceprintScoreChange(
                        1,
                        "Speaker B",
                        "墨泪",
                        0.900,
                        "墨泪",
                        0.820,
                        -0.080,
                        "declined",
                        0.75,
                    ),
                ),
            ),
        ),
    )

    rendered = voiceprint_review_workflow._historical_evaluation_text(evaluation)

    assert "[bold red]critical 2[/]" in rendered
    assert "[yellow]warnings 1[/]" in rendered
    assert "[bold red]CRITICAL[/] p-risk" in rendered
    assert "[yellow]WARNING[/] p-warn" in rendered
    assert "review: meeting-asr project review p-risk" in rendered
    assert "review: meeting-asr project review p-warn" in rendered
    assert "Speaker D: 敬悦 0.734 -> 武一 0.802 (+0.068)" in rendered
    assert "Speaker E: 米汤 0.801 -> 米汤 0.700 (-0.101)" in rendered
    assert "Speaker B: 墨泪 0.900 -> 墨泪 0.820 (-0.080)" in rendered
    assert "Speaker A" not in rendered
    assert "changed-best" in rendered
    assert "threshold=0.750" in rendered


def test_current_project_changed_best_is_rendered_as_expected_improvement() -> None:
    """Current project speaker changes are the intended result of accepting embeddings."""
    evaluation = VoiceprintEvaluationSummary(
        VoiceprintProjectEvaluation(
            Path("project"),
            "project-1",
            "Project",
            True,
            (
                VoiceprintScoreChange(
                    1,
                    "Speaker B",
                    "华璟",
                    0.564,
                    "黄睿",
                    0.843,
                    0.279,
                    "changed-best",
                    0.75,
                ),
            ),
        ),
        (),
    )

    rendered = voiceprint_review_workflow._current_evaluation_text(evaluation)

    assert (
        "[green]Speaker B: 华璟 0.564 -> 黄睿 0.843 (+0.279) changed-best threshold=0.750[/]"
        in rendered
    )
    assert "[red]Speaker B" not in rendered


def test_historical_risk_details_are_capped_to_keep_actions_visible() -> None:
    """The result modal should summarize overflowing risk details."""
    evaluation = VoiceprintEvaluationSummary(
        _fake_evaluation().current,
        tuple(
            VoiceprintProjectEvaluation(
                Path(f"/projects/p-risk-{project_index}"),
                f"p-risk-{project_index}",
                f"风险项目 {project_index}",
                False,
                tuple(
                    VoiceprintScoreChange(
                        change_index,
                        f"Speaker {change_index}",
                        "Alice",
                        0.90,
                        "Bob",
                        0.90,
                        0.0,
                        "changed-best",
                        0.75,
                    )
                    for change_index in range(5)
                ),
            )
            for project_index in range(5)
        ),
    )

    rendered = voiceprint_review_workflow._historical_evaluation_text(evaluation)

    assert "p-risk-0" in rendered
    assert "p-risk-2" in rendered
    assert "p-risk-3" not in rendered
    assert "... 2 more risky project(s) omitted" in rendered
    assert "... 2 more risky change(s) omitted" in rendered


def _planned_capture(
    store_dir: Path, db_path: Path, clip_path: Path
) -> VoiceprintCaptureSummary:
    """Build a planned capture summary for one clip."""
    clip = VoiceprintClip(
        clip_path,
        "clips/project-1/speaker_0/clip_001.wav",
        1000,
        2000,
        1000,
        2000,
        "sample text",
    )
    return VoiceprintCaptureSummary(
        store_dir,
        db_path,
        store_dir / "clips",
        [VoiceprintSpeaker(0, "Alice", 1, "vpp-test", [clip])],
        True,
    )


def _fake_embed(**kwargs) -> VoiceprintEmbedSummary:
    """Return a deterministic embedding summary."""
    store_dir = Path(kwargs["store_dir"])
    return VoiceprintEmbedSummary(
        store_dir / "voiceprints.sqlite", "local-speechbrain", "test-model", 1, 0
    )


def _fake_evaluation(*args, **kwargs) -> VoiceprintEvaluationSummary:
    """Return a deterministic evaluation summary."""
    current = VoiceprintProjectEvaluation(
        Path("project"),
        "project-1",
        "Project",
        True,
        (
            VoiceprintScoreChange(
                0, "Speaker A", "Alice", 0.6, "Alice", 0.8, 0.2, "improved"
            ),
        ),
    )
    return VoiceprintEvaluationSummary(current, ())


def _write_file(path: Path, text: str) -> None:
    """Write a small text fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
