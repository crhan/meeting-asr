"""Tests for matching project speakers with voiceprints."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project
from app.speaker_matching import _KnownSpeakerVector
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL

runner = CliRunner()


@pytest.mark.requires_ffmpeg
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
    assert "Speaker A status=matched name=欧丁" in result.output
    assert "Speaker B status=matched name=敬悦" in result.output
    assert payload["matches"][0]["name"] == "欧丁"
    assert payload["matches"][0]["accepted"] is True
    assert payload["matches"][0]["accepted_name"] == "欧丁"
    assert payload["matches"][0]["best_name"] == "欧丁"
    assert isinstance(payload["matches"][0]["accepted_person_id"], int)
    assert payload["matches"][0]["accepted_person_id"] == payload["matches"][0]["best_person_id"]
    assert payload["matches"][0]["accepted_person_public_id"] == payload["matches"][0]["best_person_public_id"]
    assert payload["matches"][0]["candidates"][0]["person_public_id"].startswith("vpp-")
    assert payload["matches"][0]["best_score"] == payload["matches"][0]["score"]
    assert payload["matches"][0]["threshold"] == 0.7
    assert payload["matches"][0]["status"] == "matched"
    assert payload["matches"][0]["candidates"][0]["name"] == "欧丁"
    assert isinstance(payload["matches"][0]["candidates"][0]["person_id"], int)
    assert payload["provider"] == "local-speechbrain"
    assert payload["model"] == LOCAL_SPEECHBRAIN_MODEL


def test_project_speakers_match_keeps_below_threshold_best_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Low-score candidates should stay explainable without being auto-applied."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    _patch_audio_embedding(monkeypatch)
    monkeypatch.setattr(
        "app.speaker_matching._known_speaker_vectors",
        lambda store_dir, model: {7: _KnownSpeakerVector(7, "墨泪", [0.8, 0.6], "vpp-0000000000000007")},
    )

    result = runner.invoke(
        app,
        ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir), "--threshold", "0.9"],
    )

    payload = json.loads((project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8"))
    first = payload["matches"][0]
    assert result.exit_code == 0
    assert "Speaker A status=below-threshold best=墨泪 score=0.800 threshold=0.900" in result.output
    assert "Speaker A -> unknown" not in result.output
    assert "Recommended next step:" in result.output
    assert "meeting-asr project speakers review" in result.output
    assert "Advanced/scripted alternative" in result.output
    assert "--map 0=Name" in result.output
    assert first["name"] is None
    assert first["accepted_name"] is None
    assert first["accepted"] is False
    assert first["best_name"] == "墨泪"
    assert first["best_score"] == 0.8
    assert first["score"] == 0.8
    assert first["threshold"] == 0.9
    assert first["status"] == "below-threshold"
    assert first["best_person_id"] == 7
    assert first["best_person_public_id"] == "vpp-0000000000000007"
    assert first["accepted_person_id"] is None
    assert first["accepted_person_public_id"] is None
    assert first["candidates"] == [
        {
            "person_id": 7,
            "person_public_id": "vpp-0000000000000007",
            "name": "墨泪",
            "score": 0.8,
        }
    ]


@pytest.mark.requires_ffmpeg
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
    person_map = json.loads((project_dir / "speakers" / "speaker_person_map.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "Applied accepted speaker matches." in result.output
    assert "欧丁" in transcript
    assert "敬悦" in transcript
    assert set(person_map) == {"0", "1"}
    assert all(str(value).startswith("vpp-") for value in person_map.values())


@pytest.mark.requires_ffmpeg
def test_project_speakers_match_reuses_project_probe_embedding_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Repeated matching should not re-embed unchanged project speaker probes."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    calls: list[Path] = []
    _write_named_speaker_inputs(project_dir)
    _patch_audio_embedding(monkeypatch, calls=calls)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    first = runner.invoke(app, ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir)])
    after_first = len(calls)
    second = runner.invoke(app, ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir)])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert after_first > 0
    assert len(calls) == after_first
    assert (project_dir / "tmp" / "voiceprint_match" / "probe_embeddings.json").exists()


def test_project_speakers_match_allows_empty_voiceprint_library(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Empty voiceprint stores should produce review-only unknown matches."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "empty-voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.speaker_matching.embed_audio_file", _raise_unexpected_embedding)

    result = runner.invoke(
        app,
        ["project", "speakers", "match", str(project_dir), "--store-dir", str(store_dir)],
    )

    payload = json.loads((project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "Speaker A status=no-candidate threshold=0.750" in result.output
    assert "Speaker B status=no-candidate threshold=0.750" in result.output
    assert "below-threshold" not in result.output
    assert payload["matches"][0]["name"] is None
    assert payload["matches"][0]["accepted"] is False
    assert payload["matches"][0]["score"] == 0.0
    assert payload["matches"][0]["best_name"] is None
    assert payload["matches"][0]["best_score"] is None
    assert payload["matches"][0]["accepted_name"] is None
    assert payload["matches"][0]["threshold"] == 0.75
    assert payload["matches"][0]["status"] == "no-candidate"
    assert not (project_dir / "tmp" / "voiceprint_match").exists()


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


def _patch_audio_embedding(monkeypatch, *, calls: list[Path] | None = None) -> None:
    """Patch audio extraction and embedding with deterministic test doubles."""
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.speaker_matching.extract_audio_clip", _fake_extract_audio_clip)
    if calls is None:
        monkeypatch.setattr("app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file)
        monkeypatch.setattr("app.speaker_matching.embed_audio_file", _fake_embed_audio_file)
        return
    monkeypatch.setattr("app.voiceprint_embedding.embed_audio_file", _tracking_fake_embed_audio_file(calls))
    monkeypatch.setattr("app.speaker_matching.embed_audio_file", _tracking_fake_embed_audio_file(calls))


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Write a fake clip payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((1000).to_bytes(2, "little", signed=True) * 160)
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
    """Return deterministic vectors based on speaker id in the path."""
    return [0.0, 1.0] if "speaker_1" in str(path) else [1.0, 0.0]


def _tracking_fake_embed_audio_file(calls: list[Path]):
    """Return an embedding fake that records each call."""

    def fake(path: Path, *, provider: str | None) -> list[float]:
        calls.append(path)
        return _fake_embed_audio_file(path, provider=provider)

    return fake


def _raise_unexpected_embedding(path: Path, *, provider: str | None) -> list[float]:
    """Fail when an empty voiceprint library tries to embed project probes."""
    raise AssertionError(f"Unexpected embedding call for {path}.")
