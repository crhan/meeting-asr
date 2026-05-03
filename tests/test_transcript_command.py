"""Tests for transcript viewing commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project, load_manifest

runner = CliRunner()


def test_project_transcript_show_prefers_named_output(tmp_path: Path) -> None:
    """Project-scoped transcript command should show generated project output."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "show", str(project_dir)])

    assert result.exit_code == 0
    assert result.output == "欧丁: 你好\n"


def test_project_transcript_show_accepts_project_id(tmp_path: Path) -> None:
    """Transcript commands should resolve project ids from the project store."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(
        app,
        ["project", "transcript", "show", manifest.project_id, "--projects-dir", str(projects_dir)],
    )

    assert result.exit_code == 0
    assert result.output == "欧丁: 你好\n"


def test_transcript_path_prefers_named_srt(tmp_path: Path) -> None:
    """SRT mode should prefer named subtitles after speaker apply."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "path", str(project_dir), "--kind", "srt"])

    assert result.exit_code == 0
    assert result.output.strip() == str(project_dir / "exports" / "subtitle_named.srt")


def test_transcript_list_shows_available_artifacts(tmp_path: Path) -> None:
    """List mode should expose where each artifact lives."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "list", str(project_dir)])

    assert result.exit_code == 0
    assert "Artifacts: 6/6 available" in result.output
    assert "named" in result.output
    assert "exports/transcript_named.txt" in result.output
    assert "asr/raw_result.json" in result.output
    assert "asr/sentences.json" in result.output


def test_transcript_list_prints_json(tmp_path: Path) -> None:
    """List mode should expose stable JSON for scripts and agents."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "list", str(project_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["project"] == str(project_dir)
    assert payload["count"] == 6
    assert payload["available_count"] == 6
    named = next(item for item in payload["artifacts"] if item["kind"] == "named")
    assert named["available"] is True
    assert named["path"] == str(project_dir / "exports" / "transcript_named.txt")
    assert str(project_dir / "exports" / "transcript_named.txt") in named["candidates"]


def test_transcript_list_prints_plain_rows(tmp_path: Path) -> None:
    """List mode should offer stable plain output for scripts."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "list", str(project_dir), "--plain"])

    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "kind\tstatus\tpath"
    assert "named\tavailable\texports/transcript_named.txt" in result.output
    assert "srt\tavailable\texports/subtitle_named.srt" in result.output
    assert "╭" not in result.output


def test_transcript_list_shows_missing_artifacts(tmp_path: Path) -> None:
    """List mode should show expected locations for absent artifacts."""
    project_dir = _sample_project(tmp_path)

    result = runner.invoke(app, ["project", "transcript", "list", str(project_dir)])

    assert result.exit_code == 0
    assert "Artifacts: 0/6 available" in result.output
    assert "missing" in result.output
    assert "expected: exports/transcript.txt" in result.output
    assert "exports/subtitle_named.srt or exports/subtitle.srt" in result.output


def test_transcript_show_can_select_plain_output(tmp_path: Path) -> None:
    """Kind selection should allow deterministic script usage."""
    project_dir = _sample_project(tmp_path)
    _write_transcript_outputs(project_dir)

    result = runner.invoke(app, ["project", "transcript", "show", str(project_dir), "--kind", "plain"])

    assert result.exit_code == 0
    assert result.output == "纯文本\n"


def test_top_level_transcript_command_is_not_registered() -> None:
    """Transcript viewing should live under project scope."""
    result = runner.invoke(app, ["transcript", "show"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def _sample_project(tmp_path: Path, *, projects_dir: Path | None = None) -> Path:
    """Create a minimal project for transcript command tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = (projects_dir or tmp_path) / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=projects_dir or tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _write_transcript_outputs(project_dir: Path) -> None:
    """Write all transcript artifacts used by tests."""
    exports_dir = project_dir / "exports"
    asr_dir = project_dir / "asr"
    exports_dir.mkdir(exist_ok=True)
    asr_dir.mkdir(exist_ok=True)
    (exports_dir / "transcript.txt").write_text("纯文本\n", encoding="utf-8")
    (exports_dir / "transcript_speakers.txt").write_text("Speaker A: 你好\n", encoding="utf-8")
    (exports_dir / "transcript_named.txt").write_text("欧丁: 你好\n", encoding="utf-8")
    (exports_dir / "subtitle.srt").write_text("anonymous srt\n", encoding="utf-8")
    (exports_dir / "subtitle_named.srt").write_text("named srt\n", encoding="utf-8")
    (asr_dir / "raw_result.json").write_text("{}\n", encoding="utf-8")
    (asr_dir / "sentences.json").write_text('{"sentences": []}\n', encoding="utf-8")
