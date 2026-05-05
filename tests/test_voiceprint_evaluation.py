"""Tests for voiceprint embedding impact evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from app import voiceprint_evaluation
from app.project_manager import create_project
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary
from app.voiceprint_evaluation import evaluate_voiceprint_embedding


def test_evaluate_voiceprint_embedding_updates_current_and_flags_history(monkeypatch, tmp_path: Path) -> None:
    """Evaluation should persist current rematch and dry-run historical regressions."""
    projects_dir = tmp_path / "projects"
    current = _project(projects_dir, tmp_path / "current.mp4", "Current")
    historical = _project(projects_dir, tmp_path / "historical.mp4", "Historical")
    _write_matches(current, best_name="Alice", score=0.60)
    _write_matches(historical, best_name="Alice", score=0.82)
    calls: list[Path] = []

    def fake_match_project_speakers(project_dir: Path, **kwargs) -> SpeakerMatchSummary:
        calls.append(project_dir)
        return _summary(project_dir, best_name="Alice", score=0.79)

    def fake_preview_project_speaker_matches(project_dir: Path, **kwargs) -> SpeakerMatchSummary:
        return _summary(project_dir, best_name="Alice", score=0.70)

    monkeypatch.setattr(voiceprint_evaluation, "match_project_speakers", fake_match_project_speakers)
    monkeypatch.setattr(voiceprint_evaluation, "preview_project_speaker_matches", fake_preview_project_speaker_matches)

    summary = evaluate_voiceprint_embedding(
        current,
        store_dir=None,
        provider=None,
        endpoint=None,
        model="test-model",
    )

    assert calls == [current]
    assert summary.current.improved_count == 1
    assert round(summary.current.changes[0].delta or 0.0, 3) == 0.19
    assert summary.historical_project_count == 1
    assert summary.historical[0].project_dir == historical
    assert summary.historical_risk_count == 1


def _project(projects_dir: Path, source: Path, title: str) -> Path:
    """Create one minimal project."""
    source.write_bytes(b"source")
    project_dir = projects_dir / title.lower()
    create_project(
        source,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _write_matches(project_dir: Path, *, best_name: str, score: float) -> None:
    """Write a minimal speaker match file."""
    path = project_dir / "speakers" / "speaker_matches.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "threshold": 0.75,
                "matches": [
                    {
                        "speaker_id": 0,
                        "label": "Speaker A",
                        "name": best_name if score >= 0.75 else None,
                        "accepted": score >= 0.75,
                        "best_name": best_name,
                        "best_score": score,
                        "score": score,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _summary(project_dir: Path, *, best_name: str, score: float) -> SpeakerMatchSummary:
    """Build a speaker match summary."""
    accepted = score >= 0.75
    match = SpeakerMatch(
        0,
        "Speaker A",
        best_name if accepted else None,
        score,
        accepted,
        1,
        best_name,
        score,
        best_name if accepted else None,
        0.75,
    )
    return SpeakerMatchSummary(project_dir / "speakers" / "speaker_matches.json", "local-speechbrain", "test-model", 0.75, [match])
