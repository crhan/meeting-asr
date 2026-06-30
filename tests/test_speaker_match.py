"""Tests for matching project speakers with voiceprints."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project
from app.speaker_matching import (
    _KnownProjectVector,
    _KnownSpeakerVector,
    _ranked_matches,
)
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
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(store_dir),
            "--threshold",
            "0.7",
        ],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8")
    )
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
    assert (
        payload["matches"][0]["accepted_person_id"]
        == payload["matches"][0]["best_person_id"]
    )
    assert (
        payload["matches"][0]["accepted_person_public_id"]
        == payload["matches"][0]["best_person_public_id"]
    )
    assert payload["matches"][0]["candidates"][0]["person_public_id"].startswith("vpp-")
    assert payload["matches"][0]["best_score"] == payload["matches"][0]["score"]
    assert payload["matches"][0]["threshold"] == 0.7
    assert payload["matches"][0]["status"] == "matched"
    assert payload["matches"][0]["candidates"][0]["name"] == "欧丁"
    assert isinstance(payload["matches"][0]["candidates"][0]["person_id"], int)
    assert payload["matches"][0]["candidates"][0]["score_source"] == "person-centroid"
    assert "probe_segments" in payload["matches"][0]["diagnostics"]
    assert payload["matches"][0]["diagnostics"]["probe_sample_count"] == 1
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
        lambda store_dir, model: {
            7: _KnownSpeakerVector(7, "墨泪", [0.8, 0.6], "vpp-0000000000000007")
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
            str(store_dir),
            "--threshold",
            "0.9",
        ],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8")
    )
    first = payload["matches"][0]
    assert result.exit_code == 0
    assert (
        "Speaker A status=below-threshold best=墨泪 score=0.800 threshold=0.900"
        in result.output
    )
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
            "score_source": "person-centroid",
            "sample_count": 0,
            "project_count": 0,
        }
    ]


def test_project_speakers_match_accepts_strong_margin_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A below-threshold candidate with a large top-2 margin is safe to accept."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    _patch_audio_embedding(monkeypatch)
    monkeypatch.setattr(
        "app.speaker_matching._known_speaker_vectors",
        lambda store_dir, model: {
            7: _KnownSpeakerVector(
                7,
                "墨泪",
                [0.66, math.sqrt(1 - 0.66**2)],
                "vpp-0000000000000007",
            ),
            8: _KnownSpeakerVector(
                8,
                "欧丁",
                [0.30, math.sqrt(1 - 0.30**2)],
                "vpp-0000000000000008",
            ),
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
            str(store_dir),
            "--threshold",
            "0.75",
        ],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8")
    )
    first = payload["matches"][0]
    assert result.exit_code == 0
    assert "Speaker A status=matched name=墨泪 score=0.660" in result.output
    assert first["accepted"] is True
    assert first["accepted_name"] == "墨泪"
    assert first["best_score"] == 0.66
    assert first["threshold"] == 0.75
    assert first["margin_score"] == 0.36000000000000004
    assert first["accept_reason"] == "strong-margin"
    assert first["status"] == "matched"
    assert first["diagnostics"]["accept_reason"] == "strong-margin"
    assert first["diagnostics"]["margin_score"] == 0.36000000000000004
    assert first["diagnostics"]["best_score_source"] == "person-centroid"


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
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(store_dir),
            "--apply",
        ],
    )

    transcript = (project_dir / "exports" / "transcript_named.txt").read_text(
        encoding="utf-8"
    )
    person_map = json.loads(
        (project_dir / "speakers" / "speaker_person_map.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.exit_code == 0
    assert "Applied accepted speaker matches." in result.output
    assert "欧丁" in transcript
    assert "敬悦" in transcript
    assert set(person_map) == {"0", "1"}
    assert all(str(value).startswith("vpp-") for value in person_map.values())


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
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])

    first = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(store_dir),
        ],
    )
    after_first = len(calls)
    second = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(store_dir),
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert after_first > 0
    assert len(calls) == after_first
    assert (project_dir / "tmp" / "voiceprint_match" / "probe_embeddings.json").exists()


def test_project_speakers_match_prefers_quality_probe_segments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Probe sampling should not blindly choose the longest low-information clip."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_quality_probe_inputs(project_dir)
    starts: list[float] = []

    def fake_extract(
        input_path: Path,
        output_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
    ) -> Path:
        starts.append(start_seconds)
        return _fake_extract_audio_clip(
            input_path,
            output_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )

    monkeypatch.setattr("app.speaker_matching.extract_audio_clip", fake_extract)
    monkeypatch.setattr("app.speaker_matching.embed_audio_file", _fake_embed_audio_file)
    monkeypatch.setattr(
        "app.speaker_matching._known_speaker_vectors",
        lambda store_dir, model: {
            7: _KnownSpeakerVector(7, "欧丁", [1.0, 0.0], "vpp-0000000000000007")
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
            str(store_dir),
            "--sample-count",
            "1",
            "--padding-seconds",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert starts == [35.0]


def test_ranked_matches_uses_stable_project_centroid() -> None:
    """Project centroids should rescue cross-project drift from a diluted mean."""
    probe = [1.0, 0.0]
    known = {
        7: _KnownSpeakerVector(
            7,
            "米汤",
            [0.60, 0.80],
            "vpp-0000000000000007",
            (
                _KnownProjectVector("old", [0.60, 0.80], 3),
                _KnownProjectVector(
                    "current", [0.92, math.sqrt(1 - 0.92**2)], 2
                ),
            ),
            5,
            2,
        ),
        8: _KnownSpeakerVector(
            8,
            "其他人",
            [0.75, math.sqrt(1 - 0.75**2)],
            "vpp-0000000000000008",
            (),
            2,
            1,
        ),
    }

    candidates = _ranked_matches(probe, known, limit=2)

    assert candidates[0].name == "米汤"
    assert candidates[0].score == 0.92
    assert candidates[0].score_source == "project-centroid"
    assert candidates[0].sample_count == 5
    assert candidates[0].project_count == 2


def test_project_speakers_match_allows_empty_voiceprint_library(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Empty voiceprint stores should produce review-only unknown matches."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "empty-voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr(
        "app.speaker_matching.embed_audio_file", _raise_unexpected_embedding
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "match",
            str(project_dir),
            "--store-dir",
            str(store_dir),
        ],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_matches.json").read_text(encoding="utf-8")
    )
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
            {
                "begin_time_ms": 0,
                "end_time_ms": 3000,
                "text": "我是欧丁。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 4000,
                "end_time_ms": 7000,
                "text": "我是敬悦。",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "敬悦"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_quality_probe_inputs(project_dir: Path) -> None:
    """Write inputs where the longest segment is low-information filler."""
    sentences = {
        "full_text": "嗯嗯嗯。这里开始讲具体方案和风险边界。",
        "detected_speakers": [0],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 30000,
                "text": "嗯嗯嗯",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 35000,
                "end_time_ms": 41000,
                "text": "这里开始讲具体方案和风险边界。",
                "speaker_id": 0,
                "sentence_id": 2,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _patch_audio_embedding(monkeypatch, *, calls: list[Path] | None = None) -> None:
    """Patch audio extraction and embedding with deterministic test doubles."""
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr(
        "app.speaker_matching.extract_audio_clip", _fake_extract_audio_clip
    )
    if calls is None:
        monkeypatch.setattr(
            "app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file
        )
        monkeypatch.setattr(
            "app.speaker_matching.embed_audio_file", _fake_embed_audio_file
        )
        return
    monkeypatch.setattr(
        "app.voiceprint_embedding.embed_audio_file",
        _tracking_fake_embed_audio_file(calls),
    )
    monkeypatch.setattr(
        "app.speaker_matching.embed_audio_file", _tracking_fake_embed_audio_file(calls)
    )


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
