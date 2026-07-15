"""Tests for focused project speaker voiceprint learning."""

from __future__ import annotations

import json
from pathlib import Path

import app.commands.project as project_commands
import app.speaker_learning as speaker_learning
from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project
from app.speaker_learning import (
    SpeakerLearningMatch,
    SpeakerLearningResult,
    SpeakerLearningSummary,
    learn_project_speakers,
    speaker_learning_payload,
)
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary
from app.voiceprint_embedding import VoiceprintEmbedSummary
from app.voiceprints import (
    VoiceprintCapturedSample,
    VoiceprintCaptureDecision,
    VoiceprintCaptureSummary,
)

runner = CliRunner()


def test_speakers_learn_cli_defaults_to_read_only_json_plan(tmp_path: Path) -> None:
    """Without --apply, the CLI should validate and plan without creating a store."""
    project_dir = _project(tmp_path)
    store_dir = tmp_path / "voiceprints"

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "learn",
            str(project_dir),
            "--speaker-id",
            "0",
            "--store-dir",
            str(store_dir),
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["project_id"].startswith("p-")
    assert payload["status"] == "planned"
    assert payload["speakers"][0]["speaker_id"] == 0
    assert payload["speakers"][0]["status"] == "planned"
    assert not (store_dir / "voiceprints.sqlite").exists()


def test_speakers_learn_rejects_placeholder_before_apply(tmp_path: Path) -> None:
    """Learn must require a real confirmed name before touching the voiceprint store."""
    project_dir = _project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "未知发言人"}, ensure_ascii=False), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "learn",
            str(project_dir),
            "--speaker-id",
            "0",
            "--store-dir",
            str(store_dir),
            "--apply",
            "--embed",
            "--rematch",
            "--no-progress",
        ],
    )

    assert result.exit_code == 1
    assert "not confirmed and named" in result.output
    assert not (store_dir / "voiceprints.sqlite").exists()


def test_speakers_learn_cli_emits_json_before_needs_review_exit(
    monkeypatch, tmp_path: Path
) -> None:
    """Automation must receive the full result even when the command exits nonzero."""
    project_dir = _project(tmp_path)
    capture = _capture_summary(
        tmp_path,
        person_public_id="vpp-1111111111111111",
        dry_run=False,
    )
    match = SpeakerLearningMatch(
        status="matched",
        best_name="Alice",
        best_score=0.70,
        best_person_public_id="vpp-1111111111111111",
        accepted_person_public_id="vpp-1111111111111111",
    )
    fake = SpeakerLearningSummary(
        project_id=capture.project_id,
        status="needs_review",
        dry_run=False,
        threshold=0.75,
        capture=capture,
        embedding=VoiceprintEmbedSummary(
            capture.db_path, "local-speechbrain", "test-model", 1, 0
        ),
        matches=_match_summary(
            tmp_path,
            [_accepted_match(0, "Alice", "vpp-1111111111111111", score=0.70)],
        ),
        speakers=[
            SpeakerLearningResult(
                speaker_id=0,
                canonical_name="Alice",
                person_public_id="vpp-1111111111111111",
                existing_sample_count=0,
                capture_decision="captured",
                capture_reason="no_samples",
                captured_sample_ids=("vps-aaaaaaaaaaaaaaaa",),
                embedding_generated=True,
                before=None,
                after=match,
                score_delta=None,
                threshold=0.75,
                status="needs_review",
                reason="below_threshold",
                applied=False,
            )
        ],
    )
    monkeypatch.setattr(
        project_commands, "learn_project_speakers", lambda *a, **k: fake
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "learn",
            str(project_dir),
            "--speaker-id",
            "0",
            "--apply",
            "--embed",
            "--rematch",
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "needs_review"
    assert payload["speakers"][0]["reason"] == "below_threshold"


def test_speaker_learning_embeds_and_applies_only_selected_speaker(
    monkeypatch, tmp_path: Path
) -> None:
    """The closure must focus embedding and apply on the explicitly selected id."""
    person_id = "vpp-1111111111111111"
    other_person_id = "vpp-2222222222222222"
    planned = _capture_summary(tmp_path, person_public_id=None, dry_run=True)
    captured = _capture_summary(tmp_path, person_public_id=person_id, dry_run=False)
    before = _match_summary(tmp_path, [_no_candidate_match(0)])
    after = _match_summary(
        tmp_path,
        [
            _accepted_match(0, "Incidental Alias", person_id, score=0.91),
            _accepted_match(1, "Bob", other_person_id, score=0.93),
        ],
    )
    capture_calls: list[bool] = []

    def fake_capture(project_dir: Path, **kwargs) -> VoiceprintCaptureSummary:
        capture_calls.append(bool(kwargs["dry_run"]))
        return planned if kwargs["dry_run"] else captured

    embedded: dict[str, object] = {}

    def fake_embed(**kwargs) -> VoiceprintEmbedSummary:
        embedded.update(kwargs)
        return VoiceprintEmbedSummary(
            captured.db_path,
            "local-speechbrain",
            "test-model",
            1,
            0,
        )

    applied: dict[str, object] = {}

    def fake_apply(project_dir: Path, mappings, **kwargs):
        applied["project_dir"] = project_dir
        applied["mappings"] = mappings
        applied.update(kwargs)
        return (tmp_path / "map", tmp_path / "txt", tmp_path / "srt")

    monkeypatch.setattr(speaker_learning, "capture_voiceprints", fake_capture)
    monkeypatch.setattr(
        speaker_learning, "preview_project_speaker_matches", lambda *a, **k: before
    )
    monkeypatch.setattr(
        speaker_learning, "match_project_speakers", lambda *a, **k: after
    )
    monkeypatch.setattr(speaker_learning, "embed_voiceprint_samples", fake_embed)
    monkeypatch.setattr(speaker_learning, "apply_project_speakers", fake_apply)

    summary = learn_project_speakers(
        tmp_path / "project",
        speaker_ids={0},
        store_dir=tmp_path / "store",
        model=None,
        threshold=0.75,
        capture_sample_count=3,
        match_sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.5,
        min_samples=10,
        only_needed=True,
        apply_changes=True,
        embed=True,
        rematch=True,
    )

    assert capture_calls == [True, False]
    assert embedded["sample_ids"] == {101}
    assert embedded["rebuild"] is True
    assert summary.status == "matched"
    assert summary.speakers[0].status == "matched"
    assert summary.speakers[0].applied is True
    assert applied["mappings"] == {0: "Alice"}
    assert applied["person_public_mapping"] == {0: person_id}
    assert 1 not in applied["mappings"]


def test_speaker_learning_below_explicit_threshold_needs_review(
    monkeypatch, tmp_path: Path
) -> None:
    """Strong-margin acceptance below the requested threshold must not auto-apply."""
    person_id = "vpp-1111111111111111"
    planned = _capture_summary(tmp_path, person_public_id=None, dry_run=True)
    captured = _capture_summary(tmp_path, person_public_id=person_id, dry_run=False)
    below_threshold = _match_summary(
        tmp_path, [_accepted_match(0, "Alice", person_id, score=0.70)]
    )
    monkeypatch.setattr(
        speaker_learning,
        "capture_voiceprints",
        lambda project_dir, **kwargs: planned if kwargs["dry_run"] else captured,
    )
    monkeypatch.setattr(
        speaker_learning,
        "preview_project_speaker_matches",
        lambda *args, **kwargs: _match_summary(tmp_path, [_no_candidate_match(0)]),
    )
    monkeypatch.setattr(
        speaker_learning,
        "match_project_speakers",
        lambda *args, **kwargs: below_threshold,
    )
    monkeypatch.setattr(
        speaker_learning,
        "embed_voiceprint_samples",
        lambda **kwargs: VoiceprintEmbedSummary(
            captured.db_path, "local-speechbrain", "test-model", 1, 0
        ),
    )
    applied = False

    def fake_apply(*args, **kwargs):
        nonlocal applied
        applied = True

    monkeypatch.setattr(speaker_learning, "apply_project_speakers", fake_apply)

    summary = learn_project_speakers(
        tmp_path / "project",
        speaker_ids={0},
        store_dir=tmp_path / "store",
        model=None,
        threshold=0.75,
        capture_sample_count=3,
        match_sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.5,
        min_samples=10,
        only_needed=True,
        apply_changes=True,
        embed=True,
        rematch=True,
    )

    payload = speaker_learning_payload(summary)
    assert summary.status == "needs_review"
    assert summary.needs_review is True
    assert summary.speakers[0].status == "needs_review"
    assert summary.speakers[0].reason == "below_threshold"
    assert summary.speakers[0].applied is False
    assert payload["speakers"][0]["match"]["threshold"] == 0.75
    assert applied is False


def test_speaker_learning_wrong_person_never_applies(
    monkeypatch, tmp_path: Path
) -> None:
    """A high score for another stable person id is still an identity mismatch."""
    expected = "vpp-1111111111111111"
    wrong = "vpp-2222222222222222"
    planned = _capture_summary(tmp_path, person_public_id=None, dry_run=True)
    captured = _capture_summary(tmp_path, person_public_id=expected, dry_run=False)
    monkeypatch.setattr(
        speaker_learning,
        "capture_voiceprints",
        lambda project_dir, **kwargs: planned if kwargs["dry_run"] else captured,
    )
    monkeypatch.setattr(
        speaker_learning,
        "preview_project_speaker_matches",
        lambda *args, **kwargs: _match_summary(tmp_path, [_no_candidate_match(0)]),
    )
    monkeypatch.setattr(
        speaker_learning,
        "match_project_speakers",
        lambda *args, **kwargs: _match_summary(
            tmp_path, [_accepted_match(0, "Wrong", wrong, score=0.95)]
        ),
    )
    monkeypatch.setattr(
        speaker_learning,
        "embed_voiceprint_samples",
        lambda **kwargs: VoiceprintEmbedSummary(
            captured.db_path, "local-speechbrain", "test-model", 1, 0
        ),
    )
    monkeypatch.setattr(
        speaker_learning,
        "apply_project_speakers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("wrong identity must not be applied")
        ),
    )

    summary = learn_project_speakers(
        tmp_path / "project",
        speaker_ids={0},
        store_dir=tmp_path / "store",
        model=None,
        threshold=0.75,
        capture_sample_count=3,
        match_sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.5,
        min_samples=10,
        only_needed=True,
        apply_changes=True,
        embed=True,
        rematch=True,
    )

    assert summary.status == "needs_review"
    assert summary.speakers[0].reason == "identity_mismatch"


def _capture_summary(
    tmp_path: Path, *, person_public_id: str | None, dry_run: bool
) -> VoiceprintCaptureSummary:
    """Build a focused capture summary for one selected speaker."""
    sample = VoiceprintCapturedSample(
        sample_id=101,
        public_id="vps-aaaaaaaaaaaaaaaa",
        clip_path=tmp_path / "store" / "clips" / "clip.wav",
        embedded=False,
    )
    decision = VoiceprintCaptureDecision(
        speaker_id=0,
        name="Alice",
        person_id=11 if person_public_id else None,
        person_public_id=person_public_id,
        existing_sample_count=0,
        decision="capture" if dry_run else "captured",
        reason="no_samples",
        samples=[] if dry_run else [sample],
    )
    store = tmp_path / "store"
    return VoiceprintCaptureSummary(
        store_dir=store,
        db_path=store / "voiceprints.sqlite",
        clip_dir=store / "clips",
        speakers=[],
        dry_run=dry_run,
        project_id="p-test",
        decisions=[decision],
        only_needed=True,
        min_samples=10,
    )


def _project(tmp_path: Path) -> Path:
    """Create one named-speaker project for CLI planning tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"media")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Learning",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(
            {
                "full_text": "hello",
                "detected_speakers": [0],
                "sentences": [
                    {
                        "begin_time_ms": 0,
                        "end_time_ms": 3000,
                        "text": "This is a complete voiceprint sample.",
                        "speaker_id": 0,
                        "sentence_id": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "Alice"}), encoding="utf-8"
    )
    return project_dir


def _match_summary(tmp_path: Path, matches: list[SpeakerMatch]) -> SpeakerMatchSummary:
    """Build a deterministic match summary."""
    return SpeakerMatchSummary(
        match_path=tmp_path / "project" / "speakers" / "speaker_matches.json",
        provider="local-speechbrain",
        model="test-model",
        threshold=0.75,
        matches=matches,
    )


def _no_candidate_match(speaker_id: int) -> SpeakerMatch:
    """Build a no-candidate match row."""
    return SpeakerMatch(
        speaker_id=speaker_id,
        label=f"Speaker {speaker_id}",
        name=None,
        score=0.0,
        accepted=False,
        sample_count=1,
        threshold=0.75,
    )


def _accepted_match(
    speaker_id: int, name: str, person_public_id: str, *, score: float
) -> SpeakerMatch:
    """Build an accepted match row, including strong-margin below-threshold cases."""
    return SpeakerMatch(
        speaker_id=speaker_id,
        label=f"Speaker {speaker_id}",
        name=name,
        score=score,
        accepted=True,
        sample_count=1,
        best_name=name,
        best_score=score,
        accepted_name=name,
        threshold=0.75,
        best_person_id=speaker_id + 10,
        best_person_public_id=person_public_id,
        accepted_person_id=speaker_id + 10,
        accepted_person_public_id=person_public_id,
        accept_reason="threshold" if score >= 0.75 else "strong-margin",
    )
