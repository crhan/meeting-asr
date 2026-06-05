"""Tests for the low-confidence crosstalk/noise voiceprint tier."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.commands.project import (
    _ensure_named_outputs_for_nonblocking_run,
    _project_has_unresolved_match,
    _voiceprint_match_cli_line,
)
from app.project_manager import create_project, load_manifest
from app.speaker_crosstalk import CrosstalkParams, is_crosstalk
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_CROSSTALK,
    voiceprint_match_status,
)
from app.speaker_matching import (
    SpeakerMatch,
    SpeakerMatchSummary,
    VoiceprintCandidate,
    _flag_crosstalk_matches,
    _KnownSpeakerVector,
    match_project_speakers,
)

runner = CliRunner()


def _match(
    *,
    accepted: bool = False,
    sample_count: int = 2,
    best_score: float | None = 0.30,
    candidate_scores: tuple[float, ...] = (0.30, 0.28),
    crosstalk: bool = False,
) -> SpeakerMatch:
    candidates = tuple(
        VoiceprintCandidate(person_id=index, name=f"p{index}", score=score)
        for index, score in enumerate(candidate_scores)
    )
    return SpeakerMatch(
        speaker_id=0,
        label="Speaker A",
        name=None,
        score=best_score or 0.0,
        accepted=accepted,
        sample_count=sample_count,
        best_name=("p0" if candidates else None),
        best_score=best_score,
        candidates=candidates,
        crosstalk=crosstalk,
    )


# --- classifier matrix ------------------------------------------------------


def test_is_crosstalk_flags_few_low_unconcentrated() -> None:
    """Few samples + very low score + non-converging candidates → crosstalk."""
    assert is_crosstalk(_match(), CrosstalkParams()) is True


def test_is_crosstalk_skips_accepted() -> None:
    """An auto-accepted speaker is never crosstalk."""
    assert is_crosstalk(_match(accepted=True), CrosstalkParams()) is False


def test_is_crosstalk_skips_real_below_threshold() -> None:
    """A real person just below the bar (decent score) is not crosstalk."""
    assert (
        is_crosstalk(
            _match(best_score=0.70, candidate_scores=(0.70, 0.20)), CrosstalkParams()
        )
        is False
    )


def test_is_crosstalk_skips_clear_weak_frontrunner() -> None:
    """A weak but clearly leading candidate is a real person, not noise."""
    assert (
        is_crosstalk(
            _match(best_score=0.30, candidate_scores=(0.30, 0.10)), CrosstalkParams()
        )
        is False
    )


def test_is_crosstalk_skips_many_samples() -> None:
    """A well-represented speaker is never crosstalk regardless of score."""
    assert is_crosstalk(_match(sample_count=8), CrosstalkParams()) is False


def test_is_crosstalk_skips_no_candidate() -> None:
    """An empty-library row (no candidate) stays no-candidate, not crosstalk."""
    assert (
        is_crosstalk(
            _match(best_score=None, candidate_scores=()), CrosstalkParams()
        )
        is False
    )


def test_is_crosstalk_thresholds_are_configurable() -> None:
    """Raising the floor flags more; lowering max_samples excludes more."""
    real_ish = _match(best_score=0.70, candidate_scores=(0.70, 0.68))
    assert is_crosstalk(real_ish, CrosstalkParams()) is False
    assert is_crosstalk(real_ish, CrosstalkParams(score_floor=0.8)) is True
    assert is_crosstalk(_match(sample_count=2), CrosstalkParams(max_samples=1)) is False


def test_is_crosstalk_disabled_is_noop() -> None:
    """Disabled params never flag anything."""
    assert is_crosstalk(_match(), CrosstalkParams(enabled=False)) is False


# --- status + post-pass -----------------------------------------------------


def test_voiceprint_status_reads_persisted_crosstalk_flag() -> None:
    """A persisted crosstalk flag surfaces as the crosstalk status."""
    flagged = {"accepted": False, "crosstalk": True, "best_name": "p0", "best_score": 0.3}
    plain = {"accepted": False, "crosstalk": False, "best_name": "p0", "best_score": 0.3}
    assert voiceprint_match_status(flagged) == MATCH_STATUS_CROSSTALK
    assert voiceprint_match_status(plain) == MATCH_STATUS_BELOW_THRESHOLD


def test_flag_crosstalk_matches_marks_only_qualifying() -> None:
    """The post-pass flags crosstalk clusters and leaves real ones untouched."""
    noise = _match()
    real = _match(best_score=0.70, candidate_scores=(0.70, 0.20))
    flagged = _flag_crosstalk_matches([noise, real], CrosstalkParams())
    assert flagged[0].crosstalk is True
    assert flagged[1].crosstalk is False


def test_flag_crosstalk_matches_disabled_is_noop() -> None:
    """Disabled params leave every row unflagged."""
    flagged = _flag_crosstalk_matches([_match()], CrosstalkParams(enabled=False))
    assert flagged[0].crosstalk is False


# --- integration through the matcher ----------------------------------------


def _crosstalk_project(tmp_path: Path) -> Path:
    """A project whose single speaker has few short utterances."""
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
    sentences = {
        "full_text": "报一下数字。三千二。",
        "detected_speakers": [0],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1500,
                "text": "报一下数字。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 3500,
                "text": "三千二。",
                "speaker_id": 0,
                "sentence_id": 2,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False), encoding="utf-8"
    )
    return project_dir


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((1000).to_bytes(2, "little", signed=True) * 160)
    return output_path


def _fake_embed_speaker0(path: Path, *, provider: str | None) -> list[float]:
    """Speaker 0 probes embed to a fixed axis; library vectors sit far off it."""
    return [1.0, 0.0]


def test_match_flags_crosstalk_non_blocking(monkeypatch, tmp_path: Path) -> None:
    """A low-sample, low-score, ambiguous speaker is flagged crosstalk and unblocks."""
    project_dir = _crosstalk_project(tmp_path)
    before = (project_dir / "asr" / "sentences.json").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "app.speaker_matching.extract_audio_clip", _fake_extract_audio_clip
    )
    monkeypatch.setattr(
        "app.speaker_matching.embed_audio_file", _fake_embed_speaker0
    )
    # Two library people, both weakly and almost equally similar to [1, 0]:
    # best score ~0.30, lead ~0.02 -> ambiguous, below the 0.5 floor.
    monkeypatch.setattr(
        "app.speaker_matching._known_speaker_vectors",
        lambda store_dir, model: {
            5: _KnownSpeakerVector(5, "甲", [0.30, 0.95394], "vpp-0000000000000005"),
            6: _KnownSpeakerVector(6, "乙", [0.28, 0.95996], "vpp-0000000000000006"),
        },
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(tmp_path / "voiceprints"),
            "--threshold",
            "0.99",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(
        (project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8")
    )
    row = payload["matches"][0]
    assert row["crosstalk"] is True
    assert row["status"] == MATCH_STATUS_CROSSTALK
    # Non-blocking: the only unmatched speaker is crosstalk, so nothing is unresolved.
    assert _project_has_unresolved_match(project_dir) is False
    # Non-destructive: the speaker stays anonymous and no sentence moved.
    assert not (project_dir / "speakers" / "speaker_map.json").exists()
    assert (project_dir / "asr" / "sentences.json").read_text(encoding="utf-8") == before


# --- complete-run anonymous outputs -----------------------------------------


def _summary(project_dir: Path, matches: list[SpeakerMatch]) -> SpeakerMatchSummary:
    return SpeakerMatchSummary(
        match_path=project_dir / "speakers" / "speaker_matches.json",
        provider="local-speechbrain",
        model="m",
        threshold=0.99,
        matches=matches,
    )


def test_ensure_named_outputs_renders_anonymous_for_crosstalk_only(
    tmp_path: Path,
) -> None:
    """A non-blocking crosstalk-only run still gets its 'ready' named outputs."""
    project_dir = _crosstalk_project(tmp_path)
    matches = _summary(project_dir, [_match(crosstalk=True)])

    _ensure_named_outputs_for_nonblocking_run(project_dir, matches, {})

    assert (project_dir / "exports" / "transcript_named.txt").exists()
    assert (project_dir / "exports" / "subtitle_named.srt").exists()


def test_ensure_named_outputs_skips_blocking_run(tmp_path: Path) -> None:
    """A real below-threshold speaker keeps the run blocking; no anon render."""
    project_dir = _crosstalk_project(tmp_path)
    matches = _summary(project_dir, [_match(crosstalk=False)])

    _ensure_named_outputs_for_nonblocking_run(project_dir, matches, {})

    assert not (project_dir / "exports" / "transcript_named.txt").exists()


def test_ensure_named_outputs_noop_when_something_was_named(tmp_path: Path) -> None:
    """When a speaker was auto-named the workflow already rendered; helper no-ops."""
    project_dir = _crosstalk_project(tmp_path)
    matches = _summary(project_dir, [_match(crosstalk=True)])

    _ensure_named_outputs_for_nonblocking_run(project_dir, matches, {0: "X"})

    assert not (project_dir / "exports" / "transcript_named.txt").exists()


# --- persisted setting honored by rematch (Codex P2) ------------------------


def _patch_low_score_library(monkeypatch) -> None:
    """Two library people both weakly/ambiguously similar to speaker 0's probe."""
    monkeypatch.setattr(
        "app.speaker_matching.extract_audio_clip", _fake_extract_audio_clip
    )
    monkeypatch.setattr("app.speaker_matching.embed_audio_file", _fake_embed_speaker0)
    monkeypatch.setattr(
        "app.speaker_matching._known_speaker_vectors",
        lambda store_dir, model: {
            5: _KnownSpeakerVector(5, "甲", [0.30, 0.95394], "vpp-0000000000000005"),
            6: _KnownSpeakerVector(6, "乙", [0.28, 0.95996], "vpp-0000000000000006"),
        },
    )


def _match_once(project_dir: Path, store: Path, params: CrosstalkParams | None):
    return match_project_speakers(
        project_dir,
        store_dir=store,
        provider=None,
        model=None,
        threshold=0.99,
        sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.5,
        crosstalk_params=params,
    )


def test_no_crosstalk_persists_and_survives_rematch(monkeypatch, tmp_path: Path) -> None:
    """--no-crosstalk persists, so an implicit rematch keeps the tier disabled."""
    project_dir = _crosstalk_project(tmp_path)
    store = tmp_path / "voiceprints"
    _patch_low_score_library(monkeypatch)

    first = _match_once(project_dir, store, CrosstalkParams(enabled=False))
    assert all(not m.crosstalk for m in first.matches)
    assert load_manifest(project_dir).speakers["crosstalk"]["enabled"] is False

    # The rematch path (stabilization/resplit/review) passes no params; it must
    # honor the persisted disabled setting instead of re-enabling the tier.
    rematch = _match_once(project_dir, store, None)
    assert all(not m.crosstalk for m in rematch.matches)


def test_crosstalk_default_persists_enabled_and_rematch_keeps_flag(
    monkeypatch, tmp_path: Path
) -> None:
    """Default (enabled) persists, so an implicit rematch keeps flagging crosstalk."""
    project_dir = _crosstalk_project(tmp_path)
    store = tmp_path / "voiceprints"
    _patch_low_score_library(monkeypatch)

    first = _match_once(project_dir, store, CrosstalkParams())
    assert any(m.crosstalk for m in first.matches)
    assert load_manifest(project_dir).speakers["crosstalk"]["enabled"] is True

    rematch = _match_once(project_dir, store, None)
    assert any(m.crosstalk for m in rematch.matches)


# --- CLI line shows crosstalk, not no-candidate (Codex P3) ------------------


def test_cli_line_renders_crosstalk_status() -> None:
    """A crosstalk row reads as crosstalk, not the misleading no-candidate."""
    line = _voiceprint_match_cli_line(_match(crosstalk=True))
    assert "status=crosstalk" in line
    assert "no-candidate" not in line
