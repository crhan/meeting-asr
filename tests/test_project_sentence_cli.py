"""CLI tests for sentence-level project locators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_xdg_data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default voiceprint lookups inside the test sandbox."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_sentence_command_locates_corrected_sentence_text(tmp_path: Path) -> None:
    """The locator prints the reviewed sentence text and a web deep-link."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir / "asr" / "sentences.json", _sentences_payload())
    corrected = _sentences_payload()
    corrected["sentences"][1]["text"] = "第二句修正"
    _write_sentences(project_dir / "asr" / "sentences_corrected.json", corrected)

    result = runner.invoke(
        app, ["project", "speakers", "sentence", str(project_dir), "2"]
    )

    assert result.exit_code == 0, result.output
    assert "Sentence: #2" in result.output
    assert "Speaker: 1 (Speaker B)" in result.output
    assert "/speakers?sentence=2" in result.output
    assert "第二句修正" in result.output


def test_sentence_command_reassigns_by_id_and_rebuilds_named_outputs(
    tmp_path: Path,
) -> None:
    """Reassignment by locator should update sentence files and speaker outputs."""
    project_dir = _make_project(tmp_path)
    payload = _sentences_payload()
    _write_sentences(project_dir / "asr" / "sentences.json", payload)
    _write_sentences(project_dir / "asr" / "sentences_corrected.json", payload)

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "sentence",
            str(project_dir),
            "2",
            "--to-speaker",
            "2",
            "--name",
            "宵恩",
            "--no-rematch",
        ],
    )

    assert result.exit_code == 0, result.output
    raw = json.loads((project_dir / "asr" / "sentences.json").read_text("utf-8"))
    corrected = json.loads(
        (project_dir / "asr" / "sentences_corrected.json").read_text("utf-8")
    )
    assert raw["sentences"][1]["speaker_id"] == 2
    assert corrected["sentences"][1]["speaker_id"] == 2
    speaker_map = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text("utf-8")
    )
    assert speaker_map["2"] == "宵恩"
    assert "宵恩:" in (
        project_dir / "exports" / "transcript_named_corrected.txt"
    ).read_text("utf-8")


def _make_project(tmp_path: Path) -> Path:
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


def _sentences_payload() -> dict:
    return {
        "full_text": "第一句。第二句。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "第一句",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 2500,
                "text": "第二句",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }


def _write_sentences(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
