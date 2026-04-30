"""Tests for project lifecycle helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from app.cli import app
from app.project_manager import (
    _parse_project_oss_upload,
    apply_project_speakers,
    create_project,
    init_project_git,
    load_manifest,
    resolve_project_source_path,
)

runner = CliRunner()


def test_create_project_writes_manifest_and_copies_source(tmp_path: Path) -> None:
    """Project creation should establish the directory boundary."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "projects" / "supplier-ai"

    create_project(
        source,
        title="供应商管理AI治理",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time="2026-04-29T15:07:42+08:00",
        hash_source=True,
    )

    loaded = load_manifest(project_dir)
    copied_source = project_dir / "source" / "meeting.mp4"
    assert copied_source.read_bytes() == b"fake video"
    assert loaded.source.path == "source/meeting.mp4"
    assert loaded.source.original_path == str(source.resolve())
    assert resolve_project_source_path(project_dir, loaded) == copied_source.resolve()
    assert loaded.source.sha256 is not None


def test_create_project_reports_copy_progress(tmp_path: Path) -> None:
    """Project creation should expose copy progress without printing from the core."""
    source = tmp_path / "meeting.mp4"
    payload = b"fake video"
    source.write_bytes(payload)
    project_dir = tmp_path / "projects" / "supplier-ai"
    events = []

    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
        progress=events.append,
    )

    assert events[0].description == "Copying source media"
    assert events[0].total == len(payload)
    assert sum(event.advance for event in events) == len(payload)


def test_project_create_command_defaults_to_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project creation without --projects-dir should use XDG data home."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")

    result = runner.invoke(app, ["project", "create", str(source), "--title", "Demo"])

    project_dirs = [path for path in (data_home / "meeting-asr" / "projects").iterdir() if path.is_dir()]
    assert result.exit_code == 0
    assert len(project_dirs) == 1
    assert "Project created." in result.output
    assert f"cd {project_dirs[0].resolve()}" in result.output
    assert "meeting-asr project transcribe" in result.output
    assert "meeting-asr project transcribe ." not in result.output


def test_project_create_output_quotes_copyable_cd_command(tmp_path: Path) -> None:
    """Project creation output should be pasteable when paths contain spaces."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project with space"

    result = runner.invoke(app, ["project", "create", str(source), "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert f"cd '{project_dir.resolve()}'" in result.output
    assert "meeting-asr project status" in result.output
    assert "meeting-asr project status ." not in result.output


def test_project_list_command_reads_default_projects_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project list should use the same XDG parent as project create."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    create_project(
        source,
        title="Demo",
        projects_dir=None,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
    )
    projects_dir = data_home / "meeting-asr" / "projects"
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())

    result = runner.invoke(app, ["project", "list"])

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "Demo" in result.output
    assert "created" in result.output
    assert str(project_dir.resolve()) in result.output


def test_project_list_command_accepts_projects_dir(tmp_path: Path) -> None:
    """Project list should scan the requested projects parent only."""
    projects_dir = tmp_path / "projects"
    ignored_dir = projects_dir / "not-a-project"
    ignored_dir.mkdir(parents=True)
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = projects_dir / "demo"
    create_project(
        source,
        title="Demo",
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )

    result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir)])

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "Demo" in result.output
    assert str(project_dir.resolve()) in result.output
    assert "not-a-project" not in result.output


def test_project_list_command_handles_empty_projects_dir(tmp_path: Path) -> None:
    """Project list should treat a missing projects parent as empty."""
    projects_dir = tmp_path / "missing"

    result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir)])

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "No projects found." in result.output


def test_project_status_command_reads_manifest(tmp_path: Path) -> None:
    """The project status command should expose key manifest fields."""
    project_dir = _sample_project(tmp_path)

    result = runner.invoke(app, ["project", "status", str(project_dir)])

    assert result.exit_code == 0
    assert "Title: Demo" in result.output
    assert "Source: source/meeting.mp4" in result.output


def test_project_status_defaults_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project commands should not require '.' inside a project directory."""
    project_dir = _sample_project(tmp_path)
    monkeypatch.chdir(project_dir)

    result = runner.invoke(app, ["project", "status"])

    assert result.exit_code == 0
    assert "Title: Demo" in result.output
    assert f"Project: {project_dir.resolve()}" in result.output


def test_legacy_absolute_source_path_still_resolves(tmp_path: Path) -> None:
    """Older manifests may point to an absolute external source."""
    source = tmp_path / "old.mp4"
    source.write_bytes(b"old video")
    project_dir = _sample_project(tmp_path)
    manifest_path = project_dir / "project.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source"]["path"] = str(source.resolve())
    payload["source"].pop("original_path", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert resolve_project_source_path(project_dir, load_manifest(project_dir)) == source.resolve()


def test_top_level_transcribe_command_is_not_registered() -> None:
    """Transcription must go through project lifecycle."""
    result = runner.invoke(app, ["transcribe", "meeting.mp4"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_project_oss_upload_rejects_unknown_value() -> None:
    """Invalid upload modes should fail early."""
    with pytest.raises(typer.BadParameter, match="auto, true, or false"):
        _parse_project_oss_upload("maybe", file_url=None)


def test_apply_project_speakers_writes_project_outputs(tmp_path: Path) -> None:
    """Speaker naming should stay inside speakers/ and exports/."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    mapping_path, transcript_path, srt_path = apply_project_speakers(project_dir, {0: "欧丁"})

    assert mapping_path == project_dir / "speakers" / "speaker_map.json"
    assert transcript_path == project_dir / "exports" / "transcript_named.txt"
    assert srt_path == project_dir / "exports" / "subtitle_named.srt"
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")


def test_project_speakers_inspect_shows_mapped_names(tmp_path: Path) -> None:
    """Speaker inspect should show human names after speaker apply."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {0: "欧丁", 1: "敬悦"})

    result = runner.invoke(app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"])

    assert result.exit_code == 0
    assert "Speaker A (speaker_id=0)" in result.output
    assert "Name: 欧丁" in result.output
    assert "Speaker B (speaker_id=1)" in result.output
    assert "Name: 敬悦" in result.output


def test_project_speakers_inspect_shows_voiceprint_matches(tmp_path: Path) -> None:
    """Speaker inspect should show accepted voiceprint match suggestions."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(
            {
                "provider": "local-speechbrain",
                "model": "speechbrain-spkrec-ecapa-voxceleb",
                "threshold": 0.75,
                "matches": [
                    {
                        "speaker_id": 1,
                        "label": "Speaker B",
                        "name": "敬悦",
                        "score": 0.775052,
                        "accepted": True,
                        "sample_count": 23,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"])

    assert result.exit_code == 0
    assert "Speaker B (speaker_id=1)" in result.output
    assert "Voiceprint match: 敬悦 score=0.775 accepted" in result.output


def test_project_speakers_apply_prompts_for_names(tmp_path: Path) -> None:
    """Speaker apply should support the human review flow."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--sample-count", "1"],
        input="欧丁\n敬悦\n",
    )

    transcript_path = project_dir / "exports" / "transcript_named.txt"
    mapping = json.loads((project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "Name for Speaker A" in result.output
    assert "Name for Speaker B" in result.output
    assert mapping == {"0": "欧丁", "1": "敬悦"}
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")
    assert "meeting-asr project speakers preview" in result.output
    assert "meeting-asr voiceprint capture" in result.output
    assert f"open {transcript_path.resolve()}" in result.output


def test_project_speakers_apply_can_show_more_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Speaker apply should let users ask for more evidence before naming."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    remembered: list[str] = []
    monkeypatch.setattr("app.commands.project._remember_prompt_history", remembered.append)

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--sample-count", "1"],
        input="/more\n欧丁\n敬悦\n",
    )

    mapping = json.loads((project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "More samples for Speaker A" in result.output
    assert "再补一句。" in result.output
    assert remembered == ["/more"]
    assert mapping == {"0": "欧丁", "1": "敬悦"}


def test_project_speakers_preview_prefers_named_subtitle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preview after speaker naming should use the named subtitle."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "exports").mkdir(exist_ok=True)
    (project_dir / "exports" / "subtitle.srt").write_text("anonymous", encoding="utf-8")
    (project_dir / "exports" / "subtitle_named.srt").write_text("named", encoding="utf-8")
    captured: dict[str, Path] = {}

    def fake_build_preview_command(*, video: Path, subtitle: Path, start_seconds: float) -> list[str]:
        captured["subtitle"] = subtitle
        return ["player", str(subtitle)]

    monkeypatch.setattr("app.commands.project.build_preview_command", fake_build_preview_command)

    result = runner.invoke(app, ["project", "speakers", "preview", str(project_dir), "--dry-run"])

    assert result.exit_code == 0
    assert captured["subtitle"] == project_dir.resolve() / "exports" / "subtitle_named.srt"
    assert "subtitle_named.srt" in result.output


def test_project_git_init_writes_safe_ignore_file(tmp_path: Path) -> None:
    """Optional Git tracking should ignore heavy generated artifacts."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    project_dir = _sample_project(tmp_path)

    gitignore_path = init_project_git(project_dir)

    content = gitignore_path.read_text(encoding="utf-8")
    assert "source/" in content
    assert "audio/" in content


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for tests."""
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


def _write_sample_sentences(path: Path) -> None:
    """Write a normalized sentences.json fixture."""
    payload = {
        "full_text": "大家好。收到。",
        "detected_speakers": [0, 1],
        "sentences": [
            {"begin_time_ms": 0, "end_time_ms": 1000, "text": "大家好。", "speaker_id": 0, "sentence_id": 1},
            {"begin_time_ms": 1200, "end_time_ms": 1800, "text": "收到。", "speaker_id": 1, "sentence_id": 2},
            {"begin_time_ms": 1900, "end_time_ms": 2500, "text": "再补一句。", "speaker_id": 0, "sentence_id": 3},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
