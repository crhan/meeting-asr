"""Tests for editor-driven project vocabulary correction."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project

runner = CliRunner()


def test_project_correct_edit_writes_corrected_outputs_and_learns_context(tmp_path: Path) -> None:
    """Editing the review file should write corrected artifacts and lexicon context."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")
    lexicon_db = tmp_path / "lexicon.sqlite"

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
            "--yes",
            "--lexicon-db",
            str(lexicon_db),
            "--category",
            "system",
        ],
    )

    assert result.exit_code == 0
    assert "Vocabulary correction accepted." in result.output
    assert "Changed sentences: 1" in result.output
    assert "Learned contexts: 1" in result.output
    assert "艾赛" in (project_dir / "asr" / "sentences.json").read_text(encoding="utf-8")
    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in (
        project_dir / "exports" / "transcript_named_corrected.txt"
    ).read_text(encoding="utf-8")
    assert (project_dir / "exports" / "subtitle_named_corrected.srt").exists()
    assert (project_dir / "corrections" / "asr_hotwords.json").exists()
    hotwords = json.loads((project_dir / "corrections" / "asr_hotwords.json").read_text(encoding="utf-8"))
    assert hotwords["dashscope_vocabulary"] == [{"text": "iSee", "weight": 4}]
    assert _fetch_one(lexicon_db, "SELECT canonical FROM terms") == "iSee"
    assert _fetch_one(lexicon_db, "SELECT alias FROM aliases") == "艾赛"
    assert _fetch_one(lexicon_db, "SELECT category FROM terms") == "system"


def test_project_correct_edit_no_open_only_creates_review_file(tmp_path: Path) -> None:
    """No-open mode should let users inspect the generated review file without applying changes."""
    project_dir = _sample_project(tmp_path)

    result = runner.invoke(app, ["project", "correct", "edit", str(project_dir), "--no-open"])
    review_files = list((project_dir / "tmp" / "corrections").glob("review_*.md"))

    assert result.exit_code == 0
    assert "Changed sentences: 0" in result.output
    assert review_files
    assert "meeting-asr: sentence_id=1" in review_files[0].read_text(encoding="utf-8")
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()


def test_project_correct_edit_can_leave_proposal_pending(tmp_path: Path) -> None:
    """Without acceptance, edit should produce proposal files but not final artifacts."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Vocabulary correction proposal ready." in result.output
    assert "Correction proposal left pending." in result.output
    assert list((project_dir / "tmp" / "corrections").glob("proposal_*.json"))
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()


def test_project_correct_accept_applies_latest_proposal(tmp_path: Path) -> None:
    """Accept command should apply a pending proposal and learn contexts."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")
    lexicon_db = tmp_path / "lexicon.sqlite"
    runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    result = runner.invoke(
        app,
        ["project", "correct", "accept", str(project_dir), "--lexicon-db", str(lexicon_db)],
    )

    assert result.exit_code == 0
    assert "Vocabulary correction accepted." in result.output
    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert _fetch_one(lexicon_db, "SELECT canonical FROM terms") == "iSee"


def test_project_correct_edit_can_use_existing_review_file(tmp_path: Path) -> None:
    """Existing edited review files should be reusable for proposal generation."""
    project_dir = _sample_project(tmp_path)
    runner.invoke(app, ["project", "correct", "edit", str(project_dir), "--no-open"])
    review_file = next((project_dir / "tmp" / "corrections").glob("review_*.md"))
    review_file.write_text(review_file.read_text(encoding="utf-8").replace("艾赛", "iSee"), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--review-file",
            str(review_file),
            "--from-original",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Sample changes: 1" in result.output


def test_project_transcript_show_can_select_corrected_output(tmp_path: Path) -> None:
    """Corrected transcript artifacts should be viewable through project transcript show."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")

    runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
            "--yes",
        ],
    )
    result = runner.invoke(app, ["project", "transcript", "show", str(project_dir), "--kind", "corrected"])

    assert result.exit_code == 0
    assert "iSee" in result.output


def _sample_project(tmp_path: Path) -> Path:
    """Create a project fixture with one mapped speaker and one ASR error."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "projects" / "demo"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    sentences = {
        "full_text": "我们看一下艾赛系统。",
        "detected_speakers": [0],
        "sentences": [
            {
                "begin_time_ms": 1000,
                "end_time_ms": 3000,
                "text": "我们看一下艾赛系统。",
                "speaker_id": 0,
                "sentence_id": 1,
            }
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text('{"0": "敬悦"}\n', encoding="utf-8")
    return project_dir


def _editor_script(tmp_path: Path, old: str, new: str) -> Path:
    """Write an editor script that replaces text in the review file."""
    script = tmp_path / f"editor_{old}_{new}.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "path = Path(sys.argv[1])",
                f"path.write_text(path.read_text(encoding='utf-8').replace({old!r}, {new!r}), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return script


def _fetch_one(db_path: Path, query: str) -> str:
    """Fetch a single SQLite string value."""
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(query).fetchone()
    assert row is not None
    return str(row[0])
