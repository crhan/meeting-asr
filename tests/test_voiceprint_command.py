"""Tests for cross-project voiceprint commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.config import save_config_values
from app.project_manager import create_project, load_manifest
from app.voiceprint_embedding import BAILIAN_VOICEPRINT_MODEL, LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings, list_voiceprint_samples_for_project

runner = CliRunner()


def test_voiceprint_capture_writes_xdg_store_and_sqlite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Capture should store WAV clips outside the project and index them in SQLite."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    manifest = load_manifest(project_dir)
    assert result.exit_code == 0
    assert "Captured voiceprint samples: 2" in result.output
    assert "Next steps:" in result.output
    assert f"meeting-asr voiceprint embed --store-dir {store_dir.resolve()}" in result.output
    assert "meeting-asr voiceprint list" in result.output
    assert (store_dir / "voiceprints.sqlite").exists()
    assert (store_dir / "clips" / manifest.project_id / "speaker_0" / "clip_001.wav").exists()
    assert not (project_dir / "speakers" / "voiceprints").exists()
    assert manifest.speakers["voiceprints"]["sample_count"] == 2
    project_samples = list_voiceprint_samples_for_project(manifest.project_id, get_voiceprint_db_path(store_dir))
    assert len(project_samples) == 2
    assert {sample.project_id for sample in project_samples} == {manifest.project_id}

    list_result = runner.invoke(app, ["voiceprint", "list", "--store-dir", str(store_dir)])
    speaker_id = _speaker_id_from_list(list_result.output, "欧丁")
    show_result = runner.invoke(app, ["voiceprint", "show", speaker_id, "--store-dir", str(store_dir)])
    show_by_name_result = runner.invoke(app, ["voiceprint", "show", "欧丁", "--store-dir", str(store_dir)])

    assert list_result.exit_code == 0
    assert "Speakers: 2 | Samples: 2 | Embedded samples: 0/2" in list_result.output
    assert speaker_id.isdecimal()
    assert "ID" in list_result.output
    assert "Speaker" in list_result.output
    assert "Embedded" in list_result.output
    assert "欧丁" in list_result.output
    assert show_result.exit_code == 0
    assert show_by_name_result.exit_code == 0
    assert "[1] 欧丁" in show_result.output
    assert f"speaker_id: {speaker_id}" in show_result.output
    assert "sample_id:" in show_result.output
    assert manifest.project_id in show_result.output
    assert "clip_001.wav" in show_result.output


def test_voiceprint_browse_summary_uses_global_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Browse summary should expose the same global library data as the TUI."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    result = runner.invoke(app, ["voiceprint", "browse", "--summary", "--store-dir", str(store_dir)])
    tui_result = runner.invoke(app, ["voiceprint", "browse", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert f"Voiceprint library: {store_dir.resolve() / 'voiceprints.sqlite'}" in result.output
    assert "Speakers: 2 | Samples: 2 | Embedded: 0/2" in result.output
    assert "欧丁 id=" in result.output
    assert tui_result.exit_code != 0
    assert "requires an interactive terminal" in tui_result.output


def test_voiceprint_capture_skips_anonymous_speaker_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Anonymous fallback labels should not become voiceprint identities."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_partially_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    manifest = load_manifest(project_dir)
    list_result = runner.invoke(app, ["voiceprint", "list", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert "Captured voiceprint samples: 1" in result.output
    assert (store_dir / "clips" / manifest.project_id / "speaker_0" / "clip_001.wav").exists()
    assert not (store_dir / "clips" / manifest.project_id / "speaker_2").exists()
    assert "Speakers: 1 | Samples: 1 | Embedded samples: 0/1" in list_result.output
    assert "欧丁" in list_result.output
    assert "Speaker C" not in list_result.output


def test_voiceprint_play_dry_run_prints_clip_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Play should target one numbered clip without modifying the store."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    result = runner.invoke(
        app,
        ["voiceprint", "play", "欧丁", "--sample", "1", "--store-dir", str(store_dir), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "clip_001.wav" in result.output


def test_voiceprint_embed_stores_sample_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Embedding should store one vector per captured sample."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    result = runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])
    list_result = runner.invoke(app, ["voiceprint", "list", "--store-dir", str(store_dir)])
    embeddings = list_voiceprint_embeddings(LOCAL_SPEECHBRAIN_MODEL, get_voiceprint_db_path(store_dir))

    assert result.exit_code == 0
    assert "Provider: local-speechbrain" in result.output
    assert f"Model: {LOCAL_SPEECHBRAIN_MODEL}" in result.output
    assert "Embedded: 2" in result.output
    assert len(embeddings) == 2
    assert "Embedded samples: 2/2" in list_result.output


def test_voiceprint_embed_uses_configured_provider_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Embedding should resolve provider defaults from global config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    save_config_values({"voiceprint.embedding_provider": "bailian"})
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr("app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )

    result = runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])
    embeddings = list_voiceprint_embeddings(BAILIAN_VOICEPRINT_MODEL, get_voiceprint_db_path(store_dir))

    assert result.exit_code == 0
    assert "Provider: bailian" in result.output
    assert f"Model: {BAILIAN_VOICEPRINT_MODEL}" in result.output
    assert len(embeddings) == 2


def test_voiceprint_delete_sample_removes_row_and_clip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Deleting one sample should remove its row and exact WAV file."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )
    clip_path = store_dir / "clips" / load_manifest(project_dir).project_id / "speaker_0" / "clip_001.wav"

    result = runner.invoke(app, ["voiceprint", "delete-sample", "欧丁", "--sample", "1", "--store-dir", str(store_dir)])
    show_result = runner.invoke(app, ["voiceprint", "show", "欧丁", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert "Deleted sample:" in result.output
    assert "clip file: deleted" in result.output
    assert not clip_path.exists()
    assert show_result.exit_code == 1


def test_voiceprint_delete_speaker_removes_all_samples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Deleting one speaker should remove its rows and exact WAV files."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        ["voiceprint", "capture", str(project_dir), "--sample-count", "1", "--store-dir", str(store_dir)],
    )
    clip_path = store_dir / "clips" / load_manifest(project_dir).project_id / "speaker_1" / "clip_001.wav"

    speaker_id = _speaker_id_from_list(
        runner.invoke(app, ["voiceprint", "list", "--store-dir", str(store_dir)]).output,
        "敬悦",
    )

    result = runner.invoke(app, ["voiceprint", "delete-speaker", speaker_id, "--store-dir", str(store_dir), "--yes"])
    list_result = runner.invoke(app, ["voiceprint", "list", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert f"Deleted speaker: 敬悦 (id {speaker_id})" in result.output
    assert not clip_path.exists()
    assert "敬悦" not in list_result.output
    assert "Speakers: 1 | Samples: 1 | Embedded samples: 0/1" in list_result.output
    assert "欧丁" in list_result.output


def test_voiceprint_capture_dry_run_does_not_write_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Dry-run should plan global paths without writing clips or SQLite."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Planned voiceprint samples: 2" in result.output
    assert "Next steps:" not in result.output
    assert "meeting-asr voiceprint embed" not in result.output
    assert not (store_dir / "voiceprints.sqlite").exists()
    assert not (store_dir / "clips").exists()


def test_voiceprint_path_prints_xdg_paths(tmp_path: Path) -> None:
    """Path command should expose store, database, and clip roots."""
    store_dir = tmp_path / "voiceprints"

    result = runner.invoke(app, ["voiceprint", "path", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert f"Store: {store_dir.resolve()}" in result.output
    assert f"Database: {store_dir.resolve() / 'voiceprints.sqlite'}" in result.output
    assert f"Clips: {store_dir.resolve() / 'clips'}" in result.output


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for voiceprint tests."""
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
    """Write normalized transcript and speaker mapping fixtures."""
    sentences = {
        "full_text": "大家好。收到。",
        "detected_speakers": [0, 1],
        "sentences": [
            {"begin_time_ms": 0, "end_time_ms": 1000, "text": "短句。", "speaker_id": 0, "sentence_id": 1},
            {"begin_time_ms": 2000, "end_time_ms": 8000, "text": "这是一段更适合作为样本的话。", "speaker_id": 0, "sentence_id": 2},
            {"begin_time_ms": 9000, "end_time_ms": 12000, "text": "收到，我补充一下。", "speaker_id": 1, "sentence_id": 3},
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "敬悦"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_partially_named_speaker_inputs(project_dir: Path) -> None:
    """Write a transcript where one speaker still has the anonymous fallback name."""
    sentences = {
        "full_text": "大家好。还有一个人。",
        "detected_speakers": [0, 2],
        "sentences": [
            {"begin_time_ms": 0, "end_time_ms": 3000, "text": "我是欧丁。", "speaker_id": 0, "sentence_id": 1},
            {"begin_time_ms": 4000, "end_time_ms": 7000, "text": "这个人还没有确认。", "speaker_id": 2, "sentence_id": 2},
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "2": "Speaker C"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Write a fake WAV payload for tests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{input_path}:{start_seconds:.3f}:{duration_seconds:.3f}".encode()
    output_path.write_bytes(payload)
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None, endpoint: str | None) -> list[float]:
    """Return deterministic vectors based on the speaker path."""
    return [0.0, 1.0] if "speaker_1" in str(path) else [1.0, 0.0]


def _speaker_id_from_list(output: str, name: str) -> str:
    """Extract a speaker id from ``voiceprint list`` output."""
    for line in output.splitlines():
        if name not in line:
            continue
        columns = [column.strip() for column in line.split("|")]
        if len(columns) >= 2 and columns[1] == name and columns[0].isdecimal():
            return columns[0]
        cells = [cell.strip() for cell in line.split("│") if cell.strip()]
        if len(cells) >= 2 and cells[1] == name and cells[0].isdecimal():
            return cells[0]
    raise AssertionError(f"speaker not found in list output: {name}")
