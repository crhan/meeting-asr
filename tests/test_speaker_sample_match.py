"""Tests for per-sample speaker identity matching."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project
from app.speaker_matching import _KnownSpeakerVector

runner = CliRunner()


def test_project_speakers_sample_match_flags_identity_conflict(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Sample matching should flag a sentence closer to another known person."""
    project_dir = _sample_project(tmp_path)
    _write_sample_match_inputs(project_dir)
    _patch_sample_matching(monkeypatch)

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "sample-match",
            str(project_dir),
            "--threshold",
            "0.75",
            "--no-progress",
        ],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_sample_matches.json").read_text(
            encoding="utf-8"
        )
    )
    speaker_a = payload["speakers"][0]
    conflict = speaker_a["samples"][1]
    assert result.exit_code == 0
    assert "Sample match report:" in result.output
    assert "Speaker A assigned=欧丁 samples=3 ok=1 conflict=1" in result.output
    assert payload["verdict"].startswith("identity-conflict")
    assert conflict["status"] == "identity-conflict"
    assert conflict["assigned_name"] == "欧丁"
    assert conflict["best_name"] == "敬悦"
    assert conflict["assigned_score"] == 0.0
    assert conflict["best_score"] == 1.0
    assert conflict["margin_score"] == -1.0
    assert speaker_a["samples"][2]["status"] == "low-info"


def test_project_speakers_sample_match_reuses_embedding_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Repeated sample matching should not re-embed unchanged samples."""
    project_dir = _sample_project(tmp_path)
    calls: list[Path] = []
    _write_sample_match_inputs(project_dir)
    _patch_sample_matching(monkeypatch, calls=calls)

    first = runner.invoke(
        app, ["project", "speakers", "sample-match", str(project_dir), "--no-progress"]
    )
    after_first = len(calls)
    second = runner.invoke(
        app, ["project", "speakers", "sample-match", str(project_dir), "--no-progress"]
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert after_first > 0
    assert len(calls) == after_first
    assert (
        project_dir / "tmp" / "voiceprint_sample_match" / "sample_embeddings.json"
    ).exists()


def test_project_speakers_sample_match_reuses_cluster_embedding_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Sample matching should reuse compatible vectors produced by cluster diagnostics."""
    project_dir = _sample_project(tmp_path)
    calls: list[Path] = []
    _write_sample_match_inputs(project_dir)
    _patch_sample_matching(monkeypatch, calls=calls)

    first = runner.invoke(
        app, ["project", "speakers", "sample-match", str(project_dir), "--no-progress"]
    )
    sample_cache_path = (
        project_dir / "tmp" / "voiceprint_sample_match" / "sample_embeddings.json"
    )
    cluster_cache_path = (
        project_dir / "tmp" / "speaker_cluster" / "clip_embeddings.json"
    )
    cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_cache_path.write_text(
        sample_cache_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    sample_cache_path.unlink()
    calls.clear()

    second = runner.invoke(
        app, ["project", "speakers", "sample-match", str(project_dir), "--no-progress"]
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert calls == []
    assert sample_cache_path.exists()


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for sample match tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Sample Match Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _write_sample_match_inputs(project_dir: Path) -> None:
    """Write transcript and explicit speaker-person assignments."""
    sentences = {
        "full_text": "欧丁正常。敬悦混入。嗯。敬悦正常。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 3000,
                "text": "欧丁正常发言。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 4000,
                "end_time_ms": 7000,
                "text": "敬悦混入发言。",
                "speaker_id": 0,
                "sentence_id": 2,
            },
            {
                "begin_time_ms": 8000,
                "end_time_ms": 8500,
                "text": "嗯。",
                "speaker_id": 0,
                "sentence_id": 3,
            },
            {
                "begin_time_ms": 9000,
                "end_time_ms": 12000,
                "text": "敬悦正常发言。",
                "speaker_id": 1,
                "sentence_id": 4,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "speakers" / "speaker_person_map.json").write_text(
        json.dumps({"0": 1, "1": 2}, ensure_ascii=False),
        encoding="utf-8",
    )


def _patch_sample_matching(monkeypatch, *, calls: list[Path] | None = None) -> None:
    """Patch sample matching dependencies with deterministic doubles."""
    monkeypatch.setattr(
        "app.speaker_sample_matching.extract_audio_clip", _fake_extract_audio_clip
    )
    monkeypatch.setattr(
        "app.speaker_sample_matching.trim_embedding_audio_silence", _fake_trim_audio
    )
    monkeypatch.setattr(
        "app.speaker_sample_matching._known_speaker_vectors",
        _fake_known_speaker_vectors,
    )
    if calls is None:
        monkeypatch.setattr(
            "app.speaker_sample_matching.embed_audio_file", _fake_embed_audio_file
        )
        return
    monkeypatch.setattr(
        "app.speaker_sample_matching.embed_audio_file",
        _tracking_fake_embed_audio_file(calls),
    )


def _fake_known_speaker_vectors(
    store_dir: Path | None, model: str
) -> dict[int, _KnownSpeakerVector]:
    """Return two deterministic known speakers."""
    return {
        1: _KnownSpeakerVector(1, "欧丁", [1.0, 0.0], "vpp-ou-ding"),
        2: _KnownSpeakerVector(2, "敬悦", [0.0, 1.0], "vpp-jing-yue"),
    }


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


def _fake_trim_audio(input_path: Path, output_path: Path) -> Path:
    """Pretend audio preprocessing produced an embedding-ready clip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(input_path.read_bytes())
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
    """Return deterministic vectors from the sentence identity in the path."""
    path_text = str(path)
    if "sentence_2_4000_7000" in path_text or "speaker_1" in path_text:
        return [0.0, 1.0]
    return [1.0, 0.0]


def _tracking_fake_embed_audio_file(calls: list[Path]):
    """Return an embedding fake that records each call."""

    def fake(path: Path, *, provider: str | None) -> list[float]:
        calls.append(path)
        return _fake_embed_audio_file(path, provider=provider)

    return fake


def test_project_speakers_sample_match_flags_foreign_in_unnamed_cluster(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """In an unnamed cluster, only a confirmed speaker's sentence is foreign.

    Speaker 0 has no assigned identity. Its first sentence matches 欧丁, who is
    NOT confirmed anywhere, so it must stay put. Its second sentence matches
    敬悦, who IS confirmed on speaker 1, so it is a foreign sentence eligible
    for reassignment.
    """
    project_dir = _sample_project(tmp_path)
    _write_unnamed_cluster_inputs(project_dir)
    _patch_sample_matching(monkeypatch)

    result = runner.invoke(
        app,
        ["project", "speakers", "sample-match", str(project_dir), "--no-progress"],
    )

    payload = json.loads(
        (project_dir / "speakers" / "speaker_sample_matches.json").read_text(
            encoding="utf-8"
        )
    )
    unnamed = next(s for s in payload["speakers"] if s["speaker_id"] == 0)
    matches_unconfirmed = unnamed["samples"][0]
    matches_confirmed = unnamed["samples"][1]
    assert result.exit_code == 0
    assert unnamed["assigned_person_id"] is None
    # Matches 欧丁, but 欧丁 is not confirmed on any speaker -> left in place.
    assert matches_unconfirmed["best_name"] == "欧丁"
    assert matches_unconfirmed["status"] == "no-assignment"
    # Matches 敬悦, confirmed on speaker 1 -> foreign, carried in best_other.
    assert matches_confirmed["status"] == "identity-foreign"
    assert matches_confirmed["best_other_name"] == "敬悦"
    assert payload["verdict"].startswith("identity-foreign")


def _write_unnamed_cluster_inputs(project_dir: Path) -> None:
    """Write a transcript with one unnamed cluster and one confirmed speaker.

    Speaker 1 is mapped to 敬悦, so 敬悦 is a confirmed in-project identity.
    Speaker 0 has no mapping (an unnamed, below-threshold cluster): its first
    sentence embeds to 欧丁 (unconfirmed → kept) and its second embeds to 敬悦
    (confirmed → foreign). The fake embedder returns 敬悦's vector for the
    ``sentence_2_4000_7000`` clip and for any ``speaker_1`` clip.
    """
    sentences = {
        "full_text": "这句声纹像欧丁但欧丁不在场。这句其实是敬悦的串话。敬悦本人正常发言。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 3000,
                "text": "这句声纹像欧丁但欧丁并不在场。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 4000,
                "end_time_ms": 7000,
                "text": "这句其实是敬悦的串话被并进了本簇。",
                "speaker_id": 0,
                "sentence_id": 2,
            },
            {
                "begin_time_ms": 9000,
                "end_time_ms": 12000,
                "text": "敬悦本人在自己簇里正常发言。",
                "speaker_id": 1,
                "sentence_id": 3,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "speakers" / "speaker_person_map.json").write_text(
        json.dumps({"1": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
