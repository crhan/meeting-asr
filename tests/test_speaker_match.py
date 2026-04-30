"""Tests for matching project speakers with voiceprints."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL

runner = CliRunner()


def test_project_speakers_match_writes_suggestions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Project speakers should match against embedded voiceprints."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    _patch_audio_embedding(monkeypatch)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    result = runner.invoke(
        app,
        ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir), "--threshold", "0.7"],
    )

    payload = json.loads((project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "Provider: local-speechbrain" in result.output
    assert f"Model: {LOCAL_SPEECHBRAIN_MODEL}" in result.output
    assert "Speaker A -> 欧丁" in result.output
    assert "Speaker B -> 敬悦" in result.output
    assert payload["matches"][0]["name"] == "欧丁"
    assert payload["matches"][0]["accepted"] is True
    assert payload["provider"] == "local-speechbrain"
    assert payload["model"] == LOCAL_SPEECHBRAIN_MODEL


def test_project_speakers_match_can_apply_matches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Accepted matches should be able to write named project outputs."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    _patch_audio_embedding(monkeypatch)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    result = runner.invoke(
        app,
        ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir), "--apply"],
    )

    transcript = (project_dir / "exports" / "transcript_named.txt").read_text(encoding="utf-8")
    assert result.exit_code == 0
    assert "Applied accepted speaker matches." in result.output
    assert "欧丁" in transcript
    assert "敬悦" in transcript


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for speaker matching tests."""
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


def _write_named_speaker_inputs(project_dir: Path) -> None:
    """Write normalized transcript and speaker map fixtures."""
    sentences = {
        "full_text": "大家好。收到。",
        "detected_speakers": [0, 1],
        "sentences": [
            {"begin_time_ms": 0, "end_time_ms": 3000, "text": "我是欧丁。", "speaker_id": 0, "sentence_id": 1},
            {"begin_time_ms": 4000, "end_time_ms": 7000, "text": "我是敬悦。", "speaker_id": 1, "sentence_id": 2},
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "敬悦"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _patch_audio_embedding(monkeypatch) -> None:
    """Patch audio extraction and embedding with deterministic test doubles."""
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.speaker_matching.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file)
    monkeypatch.setattr("app.speaker_matching.embed_audio_file", _fake_embed_audio_file)


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Write a fake clip payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(f"{input_path}:{start_seconds}:{duration_seconds}".encode())
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None, endpoint: str | None) -> list[float]:
    """Return deterministic vectors based on speaker id in the path."""
    return [0.0, 1.0] if "speaker_1" in str(path) else [1.0, 0.0]
