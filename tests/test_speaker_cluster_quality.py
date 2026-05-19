"""Tests for project speaker cluster quality diagnostics."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.models import SentenceSegment
from app.project_manager import create_project
from app.speaker_cluster_quality import _select_segments

runner = CliRunner()


def test_project_speakers_cluster_flags_mixed_speaker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Cluster diagnostics should flag one speaker bucket with multiple voices."""
    project_dir = _sample_project(tmp_path)
    _write_cluster_inputs(project_dir)
    _patch_cluster_embedding(monkeypatch)

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "cluster",
            str(project_dir),
            "--sample-count",
            "4",
            "--same-speaker-threshold",
            "0.8",
            "--merge-threshold",
            "0.95",
            "--no-progress",
        ],
    )

    report_path = project_dir / "speakers" / "speaker_cluster_quality.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    speaker_a = payload["speakers"][0]
    assert result.exit_code == 0
    assert "Speaker A status=mixed" in result.output
    assert "multi_component" in result.output
    assert speaker_a["label"] == "Speaker A"
    assert speaker_a["status"] == "mixed"
    assert speaker_a["component_count"] == 2
    assert speaker_a["component_sizes"] == [2, 2]
    assert "multi_component" in speaker_a["warnings"]
    assert speaker_a["samples"][0]["sentence_id"] == 1
    assert speaker_a["samples"][0]["centroid_score"] is not None
    assert speaker_a["samples"][0]["status"] in {"ok", "warning", "critical"}


def test_project_speakers_cluster_can_score_all_segments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """All-segment scoring should not force all segments into centroid anchors."""
    project_dir = _sample_project(tmp_path)
    _write_cluster_inputs(project_dir)
    _patch_cluster_embedding(monkeypatch)

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "cluster",
            str(project_dir),
            "--sample-count",
            "2",
            "--all-segments",
            "--no-progress",
        ],
    )

    payload = json.loads((project_dir / "speakers" / "speaker_cluster_quality.json").read_text(encoding="utf-8"))
    speaker_a = payload["speakers"][0]
    assert result.exit_code == 0
    assert payload["scoring_mode"] == "all-segments"
    assert speaker_a["clip_count"] == 2
    assert len(speaker_a["samples"]) == 4


def test_cluster_sample_selection_prefers_informative_text() -> None:
    """Anchor selection should avoid low-information utterances when better text exists."""
    segments = [
        SentenceSegment(0, 12_000, "嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯", 0, 1),
        SentenceSegment(13_000, 16_000, "这个方案需要重新设计验证路径", 0, 2),
        SentenceSegment(17_000, 20_000, "我们先看日志再确认根因", 0, 3),
        SentenceSegment(21_000, 22_000, "好", 0, 4),
        SentenceSegment(23_000, 28_000, "那个几个事情啊就是", 0, 5),
    ]

    selected = _select_segments(segments, 2)

    assert [segment.sentence_id for segment in selected] == [2, 3]


def test_project_speakers_cluster_can_emit_json_without_writing_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """JSON diagnostics should be usable without mutating the project."""
    project_dir = _sample_project(tmp_path)
    _write_cluster_inputs(project_dir)
    _patch_cluster_embedding(monkeypatch)

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "cluster",
            str(project_dir),
            "--sample-count",
            "2",
            "--no-write-report",
            "--json",
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["provider"] == "local-speechbrain"
    assert payload["speakers"][0]["label"] == "Speaker A"
    assert not (project_dir / "speakers" / "speaker_cluster_quality.json").exists()


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for speaker cluster tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Cluster Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _write_cluster_inputs(project_dir: Path) -> None:
    """Write a transcript where Speaker A contains two acoustic clusters."""
    sentences = {
        "full_text": "alpha beta gamma delta other other",
        "detected_speakers": [0, 1],
        "sentences": [
            {"begin_time_ms": 0, "end_time_ms": 3000, "text": "alpha", "speaker_id": 0, "sentence_id": 1},
            {"begin_time_ms": 4000, "end_time_ms": 7000, "text": "beta", "speaker_id": 0, "sentence_id": 2},
            {"begin_time_ms": 8000, "end_time_ms": 11000, "text": "gamma", "speaker_id": 0, "sentence_id": 3},
            {"begin_time_ms": 12000, "end_time_ms": 15000, "text": "delta", "speaker_id": 0, "sentence_id": 4},
            {"begin_time_ms": 16000, "end_time_ms": 19000, "text": "other", "speaker_id": 1, "sentence_id": 5},
            {"begin_time_ms": 20000, "end_time_ms": 23000, "text": "other", "speaker_id": 1, "sentence_id": 6},
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")


def _patch_cluster_embedding(monkeypatch) -> None:
    """Patch audio extraction and cluster embedding with deterministic fakes."""
    monkeypatch.setattr("app.speaker_cluster_quality.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.speaker_cluster_quality.embed_audio_file", _fake_embed_audio_file)


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Write a fake WAV clip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((1000).to_bytes(2, "little", signed=True) * 160)
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
    """Return deterministic vectors from the cluster probe path."""
    path_text = str(path)
    if "speaker_0/clip_003" in path_text or "speaker_0/clip_004" in path_text:
        return [0.0, 1.0]
    if "speaker_1" in path_text:
        return [0.0, 1.0]
    return [1.0, 0.0]
