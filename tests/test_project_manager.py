"""Tests for project lifecycle helpers."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from app.cli import app
from app.commands import project as project_commands
from app.core.project_refs import list_projects, resolve_project_ref
from app.models import SentenceSegment
from app.project_manager import (
    _invalidate_downstream_artifacts,
    _parse_project_oss_upload,
    apply_project_speakers,
    create_project,
    find_project_by_source,
    init_project_git,
    load_manifest,
    ProjectMeetingSummary,
    ProjectTranscribeSummary,
    project_paths,
    resolve_project_source_path,
    save_manifest,
    summarize_project,
)
from app.meeting_summary import MeetingSummary
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary

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

    assert events[0].description == "Hashing source media"
    assert events[0].total == len(payload)
    assert any(event.description == "Copying source media" for event in events)
    assert sum(event.advance for event in events) == len(payload) * 2


def test_project_id_is_stable_for_same_source_content(tmp_path: Path) -> None:
    """Project id should not depend on creation date or title text."""
    source_a = tmp_path / "meeting-a.mp4"
    source_b = tmp_path / "meeting-b.mp4"
    source_a.write_bytes(b"same video")
    source_b.write_bytes(b"same video")
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"

    create_project(
        source_a,
        title="First Title",
        projects_dir=tmp_path / "projects",
        project_dir=project_a,
        meeting_time=None,
        hash_source=False,
    )
    create_project(
        source_b,
        title="Second Title",
        projects_dir=tmp_path / "projects",
        project_dir=project_b,
        meeting_time=None,
        hash_source=False,
    )

    manifest_a = load_manifest(project_a)
    manifest_b = load_manifest(project_b)
    assert manifest_a.project_id == manifest_b.project_id
    assert manifest_a.project_id.startswith("p-")
    assert manifest_a.source.sha256 == manifest_b.source.sha256


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
    manifest = load_manifest(project_dirs[0])
    assert "Project No.:" not in result.output
    assert f"meeting-asr project transcribe {manifest.project_id}" in result.output
    assert f"meeting-asr project review {manifest.project_id}" in result.output
    assert manifest.project_id in result.output
    assert "meeting-asr project transcribe ." not in result.output


def test_project_create_output_uses_copyable_next_steps(tmp_path: Path) -> None:
    """Project creation output should not require cd into the project."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project with space"

    result = runner.invoke(app, ["project", "create", str(source), "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert "cd " not in result.output
    assert "meeting-asr project status" in result.output
    assert "meeting-asr project status ." not in result.output


def test_project_run_applies_accepted_voiceprint_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should continue through automatic speaker matching."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 2, 3)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [
                SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2),
                SpeakerMatch(1, "Speaker B", "敬悦", 0.88, True, 1),
            ],
        )

    def fake_summarize_project(project_dir, model=None, update_title=True, progress=None):
        manifest = load_manifest(project_dir)
        manifest.title = "自动会议标题"
        save_manifest(project_dir, manifest)
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(project_dir, "自动会议标题", summary_path, json_path, "qwen-test", True)

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(project_commands, "match_project_speakers", fake_match_project_speakers)

    result = runner.invoke(app, ["project", "run", str(source), "--projects-dir", str(projects_dir)])

    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)
    transcript = project_dir / "exports" / "transcript_named.txt"
    assert result.exit_code == 0
    assert "Project automation completed." in result.output
    assert "Title" in result.output
    assert "自动会议标题" in result.output
    assert "exports/meeting_summary.md" in result.output
    assert str((project_dir / "exports" / "meeting_summary.md").resolve()) in result.output
    assert "Voiceprint matches" in result.output
    assert "2/2 accepted" in result.output
    assert f"meeting-asr project correct edit {manifest.project_id}" in result.output
    assert f"meeting-asr project transcript show {manifest.project_id} --kind corrected" in result.output
    assert "meeting-asr project review" not in result.output
    assert "欧丁" in transcript.read_text(encoding="utf-8")
    assert "敬悦" in transcript.read_text(encoding="utf-8")


def test_project_run_reports_review_when_matches_are_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should emit a concrete review handoff when automation cannot finish."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 2, 3)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [
                SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2),
                SpeakerMatch(1, "Speaker B", None, 0.12, False, 1),
            ],
        )

    def fake_summarize_project(project_dir, model=None, update_title=True, progress=None):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(project_dir, "自动会议标题", summary_path, json_path, "qwen-test", False)

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(project_commands, "match_project_speakers", fake_match_project_speakers)

    result = runner.invoke(app, ["project", "run", str(source), "--projects-dir", str(projects_dir)])
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert "Project automation needs review." in result.output
    assert "Voiceprint matches" in result.output
    assert "1/2 accepted" in result.output
    assert "partial" in result.output
    assert f"meeting-asr project review {manifest.project_id}" in result.output
    assert f"meeting-asr project correct edit {manifest.project_id}" in result.output
    assert "Agent prompt:" in result.output


def test_summarize_project_writes_summary_and_updates_auto_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project summary should write artifacts and replace a filename-derived title."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title=None,
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    monkeypatch.setattr(
        "app.project_manager.generate_meeting_summary",
        lambda result, settings, model: MeetingSummary(
            "AI 转型研讨",
            "讨论 AI 转型目标和落地路径。",
            ["目标", "路径"],
            ["补充方案"],
            "qwen-test",
        ),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings",
        lambda require_oss=False: object(),
    )

    summary = summarize_project(project_dir, model=None, update_title=True)
    manifest = load_manifest(project_dir)

    assert summary.title_updated is True
    assert manifest.title == "AI 转型研讨"
    assert manifest.outputs["meeting_summary"] == "exports/meeting_summary.md"
    assert manifest.outputs["meeting_summary_json"] == "exports/meeting_summary.json"
    assert manifest.asr["summary_model"] == "qwen-test"
    assert "讨论 AI 转型目标" in summary.summary_path.read_text(encoding="utf-8")


def test_summarize_project_preserves_manual_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A manually supplied title should not be overwritten by summarization."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="手工标题",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    monkeypatch.setattr(
        "app.project_manager.generate_meeting_summary",
        lambda result, settings, model: MeetingSummary("自动标题", "摘要", [], [], "qwen-test"),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings",
        lambda require_oss=False: object(),
    )

    summary = summarize_project(project_dir, model=None, update_title=True)

    assert summary.title_updated is False
    assert load_manifest(project_dir).title == "手工标题"

def test_project_summarize_command_prints_absolute_summary_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project summary command should print directly openable artifact paths."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )

    def fake_summarize_project(project_dir, model=None, update_title=True, progress=None):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# Demo\n", encoding="utf-8")
        json_path.write_text('{"model":"qwen-test"}\n', encoding="utf-8")
        return ProjectMeetingSummary(project_dir, "Demo", summary_path, json_path, "qwen-test", False)

    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        ["project", "summarize", manifest.project_id, "--projects-dir", str(projects_dir), "--no-progress"],
    )

    assert result.exit_code == 0
    assert f"Summary: {(project_dir / 'exports' / 'meeting_summary.md').resolve()}" in result.output
    assert f"Summary JSON: {(project_dir / 'exports' / 'meeting_summary.json').resolve()}" in result.output


def test_retranscribe_invalidates_downstream_artifacts(tmp_path: Path) -> None:
    """A fresh ASR result should not keep stale summary, speaker, or correction outputs."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "exports" / "transcript.txt").write_text("base\n", encoding="utf-8")
    apply_project_speakers(project_dir, {0: "敬悦", 1: "欧丁"})
    stale_paths = [
        project_dir / "exports" / "meeting_summary.md",
        project_dir / "exports" / "meeting_summary.json",
        project_dir / "asr" / "sentences_corrected.json",
        project_dir / "exports" / "transcript_corrected.txt",
        project_dir / "exports" / "transcript_named_corrected.txt",
        project_dir / "exports" / "subtitle_named_corrected.srt",
        project_dir / "corrections" / "asr_hotwords.json",
        project_dir / "corrections" / "applied.json",
        project_dir / "speakers" / "speaker_matches.json",
    ]
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    manifest = load_manifest(project_dir)
    manifest.asr["summary_model"] = "qwen-test"
    manifest.speakers["matches"] = "speakers/speaker_matches.json"
    manifest.speakers["voiceprints"] = {"sample_count": 1}
    manifest.outputs.update(
        {
            "meeting_summary": "exports/meeting_summary.md",
            "meeting_summary_json": "exports/meeting_summary.json",
            "corrected_sentences": "asr/sentences_corrected.json",
            "corrected_transcript": "exports/transcript_corrected.txt",
            "corrected_named_transcript": "exports/transcript_named_corrected.txt",
            "corrected_named_subtitle": "exports/subtitle_named_corrected.srt",
            "asr_hotwords": "corrections/asr_hotwords.json",
            "vocabulary_corrections": "corrections/applied.json",
        }
    )

    _invalidate_downstream_artifacts(project_paths(project_dir), manifest)

    assert (project_dir / "asr" / "sentences.json").exists()
    assert (project_dir / "exports" / "transcript.txt").exists()
    assert not (project_dir / "exports" / "transcript_named.txt").exists()
    assert not (project_dir / "speakers" / "speaker_map.json").exists()
    assert all(not path.exists() for path in stale_paths)
    assert "summary_model" not in manifest.asr
    assert not {"mapped", "matches", "voiceprints"} & set(manifest.speakers)
    assert not set(manifest.outputs) & {
        "meeting_summary",
        "meeting_summary_json",
        "named_transcript",
        "named_subtitle",
        "corrected_sentences",
        "corrected_transcript",
        "corrected_named_transcript",
        "corrected_named_subtitle",
        "asr_hotwords",
        "vocabulary_corrections",
    }


def test_resolve_project_ref_accepts_path_id_title_and_unique_partial(tmp_path: Path) -> None:
    """Project references should use stable ids, paths, titles, or unique title fragments."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Project Ref Demo")
    manifest = load_manifest(project_dir)

    assert resolve_project_ref(project_dir, projects_dir) == project_dir.resolve()
    assert resolve_project_ref(manifest.project_id, projects_dir) == project_dir.resolve()
    assert resolve_project_ref("Project Ref Demo", projects_dir) == project_dir.resolve()
    assert resolve_project_ref("Ref Demo", projects_dir) == project_dir.resolve()


def test_project_list_order_does_not_follow_updated_at(tmp_path: Path) -> None:
    """Project list order should reflect project identity chronology, not later edits."""
    projects_dir = tmp_path / "projects"
    older_source = tmp_path / "older.mp4"
    newer_source = tmp_path / "newer.mp4"
    older_source.write_bytes(b"older video")
    newer_source.write_bytes(b"newer video")
    older = projects_dir / "older"
    newer = projects_dir / "newer"
    create_project(
        older_source,
        title="Older Project",
        projects_dir=projects_dir,
        project_dir=older,
        meeting_time=None,
        hash_source=False,
    )
    create_project(
        newer_source,
        title="Newer Project",
        projects_dir=projects_dir,
        project_dir=newer,
        meeting_time=None,
        hash_source=False,
    )
    older_manifest = load_manifest(older)
    newer_manifest = load_manifest(newer)
    older_manifest.created_at = "2026-05-01T10:00:00+08:00"
    older_manifest.updated_at = "2026-05-03T10:00:00+08:00"
    newer_manifest.created_at = "2026-05-02T10:00:00+08:00"
    newer_manifest.updated_at = "2026-05-02T10:00:00+08:00"
    save_manifest(older, older_manifest)
    save_manifest(newer, newer_manifest)

    projects = list_projects(projects_dir).projects

    assert [project.title for project in projects] == ["Newer Project", "Older Project"]


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

    result = runner.invoke(app, ["project", "list"])
    project_id = list_projects(projects_dir).projects[0].project_id

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "Use Project ID or Directory" in result.output
    assert "Project ID" in result.output
    assert project_id in result.output
    assert "No." not in result.output
    assert "State" in result.output
    assert "Demo" in result.output
    assert "Created" in result.output
    assert "transcribe 1" not in result.output


def test_project_list_command_prints_json(tmp_path: Path) -> None:
    """Project list should have a stable machine-readable form."""
    projects_dir = tmp_path / "projects"
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

    result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["projects_dir"] == str(projects_dir.resolve())
    assert payload["count"] == 1
    assert payload["projects"][0]["title"] == "Demo"
    assert payload["projects"][0]["project_dir"] == str(project_dir.resolve())
    assert payload["projects"][0]["status"] == "created"
    assert payload["projects"][0]["workflow"]["state"] == "Created"
    assert payload["projects"][0]["workflow"]["artifacts"] == []
    assert payload["projects"][0]["workflow"]["next_command_short"] == (
        f"transcribe {load_manifest(project_dir).project_id}"
    )


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
    assert load_manifest(project_dir).project_id in result.output
    assert "Created" in result.output
    assert "not-a-project" not in result.output


def test_project_list_command_handles_empty_projects_dir(tmp_path: Path) -> None:
    """Project list should treat a missing projects parent as empty."""
    projects_dir = tmp_path / "missing"

    result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir)])

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "No projects found." in result.output


def test_project_update_command_changes_title_and_meeting_time(tmp_path: Path) -> None:
    """Project update should change editable manifest metadata."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Old Title")
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "project",
            "update",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--title",
            "New Title",
            "--meeting-time",
            "2026-05-02T10:00:00+08:00",
        ],
    )
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert "Project updated." in result.output
    assert "Title: New Title" in result.output
    assert manifest.title == "New Title"
    assert manifest.source.meeting_time == "2026-05-02T10:00:00+08:00"


def test_project_update_requires_a_field(tmp_path: Path) -> None:
    """Project update without updates should fail with a clear message."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Demo")
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "update", manifest.project_id, "--projects-dir", str(projects_dir)])

    assert result.exit_code == 1
    assert "Nothing to update" in result.output


def test_project_delete_moves_project_to_trash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project delete should move the project to Meeting-ASR trash by default."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Delete Me")
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "delete", manifest.project_id, "--projects-dir", str(projects_dir), "--yes"])
    list_result = runner.invoke(app, ["project", "list", "--projects-dir", str(projects_dir)])
    trash_root = tmp_path / "data" / "meeting-asr" / "trash" / "projects"
    trashed = [path for path in trash_root.iterdir() if path.is_dir()]

    assert result.exit_code == 0
    assert "Project moved to trash." in result.output
    assert not project_dir.exists()
    assert len(trashed) == 1
    assert (trashed[0] / "project.json").exists()
    assert "Delete Me" not in list_result.output


def test_project_delete_permanent_removes_project(tmp_path: Path) -> None:
    """Permanent delete should physically remove the project only when requested."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Delete Me")
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        ["project", "delete", manifest.project_id, "--projects-dir", str(projects_dir), "--permanent", "--yes"],
    )

    assert result.exit_code == 0
    assert "Project permanently deleted." in result.output
    assert not project_dir.exists()


def test_project_create_command_reuses_existing_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Creating a project from the same source should reuse the existing project."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")

    first = runner.invoke(app, ["project", "create", str(source), "--title", "Demo"])
    second = runner.invoke(app, ["project", "create", str(source), "--title", "Demo"])

    projects_dir = data_home / "meeting-asr" / "projects"
    project_dirs = [path for path in projects_dir.iterdir() if path.is_dir()]
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert len(project_dirs) == 1
    assert "Project created." in first.output
    assert "Project already exists; reusing it." in second.output
    assert f"meeting-asr project review {load_manifest(project_dirs[0]).project_id}" in second.output


def test_default_project_directory_uses_project_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default project roots should use the stable project id directly."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")

    result = runner.invoke(app, ["project", "create", str(source), "--title", "Demo"])

    projects_dir = data_home / "meeting-asr" / "projects"
    project_dirs = [path for path in projects_dir.iterdir() if path.is_dir()]
    manifest = load_manifest(project_dirs[0])
    assert result.exit_code == 0
    assert project_dirs == [projects_dir / manifest.project_id]
    assert manifest.project_id.startswith("p-")


def test_find_project_by_source_prefers_more_complete_duplicate(tmp_path: Path) -> None:
    """Duplicate source lookup should pick the project with more completed work."""
    projects_dir = tmp_path / "projects"
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")
    created_project = projects_dir / "created"
    named_project = projects_dir / "named"
    create_project(
        source,
        title="Created",
        projects_dir=projects_dir,
        project_dir=created_project,
        meeting_time=None,
        hash_source=False,
    )
    create_project(
        source,
        title="Named",
        projects_dir=projects_dir,
        project_dir=named_project,
        meeting_time=None,
        hash_source=False,
    )
    named_manifest = load_manifest(named_project)
    named_manifest.status = "named"
    save_manifest(named_project, named_manifest)

    assert find_project_by_source(source, projects_dir) == named_project.resolve()


def test_project_status_command_reads_manifest(tmp_path: Path) -> None:
    """The project status command should expose key manifest fields."""
    project_dir = _sample_project(tmp_path)

    result = runner.invoke(app, ["project", "status", str(project_dir)])

    assert result.exit_code == 0
    assert "Title: Demo" in result.output
    assert "State: Created" in result.output
    assert "Next: transcribe" in result.output
    assert "Artifacts: -" in result.output
    assert "Source: source/meeting.mp4" in result.output


def test_project_status_command_prints_json(tmp_path: Path) -> None:
    """Project status should be script-friendly."""
    project_dir = _sample_project(tmp_path)
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "status", str(project_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["project"] == str(project_dir.resolve())
    assert payload["project_id"] == manifest.project_id
    assert payload["title"] == "Demo"
    assert payload["workflow"]["state"] == "Created"
    assert payload["source"] == "source/meeting.mp4"


def test_project_status_accepts_project_id(tmp_path: Path) -> None:
    """Project status should resolve project ids from the projects parent."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "status", manifest.project_id, "--projects-dir", str(projects_dir)])

    assert result.exit_code == 0
    assert f"Project: {project_dir.resolve()}" in result.output
    assert "Title: Demo" in result.output


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


def test_project_speakers_inspect_marks_voiceprint_conflicts(tmp_path: Path) -> None:
    """Speaker inspect should flag conflicts between manual names and accepted matches."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {1: "敬悦"})
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
                        "name": "墨泪",
                        "score": 0.80123,
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
    assert "Name: 敬悦" in result.output
    assert "Voiceprint match: 墨泪 score=0.801 accepted CONFLICT" in result.output


def test_speaker_match_summary_colors_review_states() -> None:
    """Voiceprint match summaries should use color to separate review states."""
    accepted = project_commands._speaker_match_summary({"name": "敬悦", "score": 0.775052, "accepted": True})
    review = project_commands._speaker_match_summary({"name": "unknown", "accepted": False})
    conflict = project_commands._speaker_match_summary(
        {"label": "Speaker B", "name": "墨泪", "score": 0.80123, "accepted": True},
        mapped_name="敬悦",
    )

    assert "\x1b[32m" in accepted
    assert "\x1b[33m" in review
    assert "\x1b[31m" in conflict
    assert "\x1b[1m" in conflict
    assert "CONFLICT" in conflict


def test_project_speakers_review_summary_shows_tui_queue(tmp_path: Path) -> None:
    """Speaker review should expose a non-interactive queue summary."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {1: "敬悦"})
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "speaker_id": 1,
                        "name": "墨泪",
                        "score": 0.80123,
                        "accepted": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "review",
            str(project_dir),
            "--summary",
            "--store-dir",
            str(tmp_path / "voiceprints"),
        ],
    )

    assert result.exit_code == 0
    assert "Speaker review queue:" in result.output
    assert "Known people: 0" in result.output
    assert "Speaker B speaker_id=1 status=conflict name=敬悦 match=墨泪" in result.output


def test_project_review_summary_accepts_project_id(tmp_path: Path) -> None:
    """Project-level review should resolve an AutoRun project id."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    result = runner.invoke(
        app,
        [
            "project",
            "review",
            manifest.project_id,
            "--summary",
            "--projects-dir",
            str(projects_dir),
            "--store-dir",
            str(tmp_path / "voiceprints"),
        ],
    )

    assert result.exit_code == 0
    assert f"Speaker review queue: {project_dir.resolve()}" in result.output
    assert "Speaker A speaker_id=0" in result.output


def test_project_review_summary_without_project_lists_history(tmp_path: Path) -> None:
    """Project-level review without PROJECT should expose the historical project list."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "review", "--summary", "--projects-dir", str(projects_dir)])

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert manifest.project_id in result.output


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


def test_project_speakers_apply_can_preview_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Speaker apply should let users play the current speaker before naming."""
    project_dir = _sample_project(tmp_path)
    sentences_path = project_dir / "asr" / "sentences.json"
    _write_sample_sentences(sentences_path)
    payload = json.loads(sentences_path.read_text(encoding="utf-8"))
    payload["sentences"].append(
        {
            "begin_time_ms": 10_000,
            "end_time_ms": 18_000,
            "text": "这是一段更适合试听的句子。",
            "speaker_id": 0,
            "sentence_id": 4,
        }
    )
    sentences_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    played: list[list[str]] = []
    remembered: list[str] = []
    captured: dict[str, Path | float] = {}
    audio_segments: list[str] = []
    audio_clip = project_dir / "tmp" / "speaker_apply_preview" / "Speaker_A" / "preview.wav"

    def fake_build_audio_preview_command(
        *,
        media: Path,
        start_seconds: float,
        duration_seconds: float | None = None,
    ) -> list[str]:
        captured["audio"] = media
        captured["audio_start"] = start_seconds
        captured["audio_duration"] = duration_seconds or 0.0
        return ["audio-player"]

    def fake_build_audio_preview_clip(
        *,
        preview_context,
        speaker_label: str,
        segments,
    ) -> Path:
        assert preview_context.project_root == project_dir.resolve()
        assert speaker_label == "Speaker A"
        audio_segments.extend(segment.text for segment in segments)
        return audio_clip

    def fake_run_preview_command(command: list[str]) -> None:
        played.append(command)

    monkeypatch.setattr("app.commands.project.build_audio_preview_command", fake_build_audio_preview_command)
    monkeypatch.setattr("app.commands.project._build_speaker_apply_audio_preview_clip", fake_build_audio_preview_clip)
    monkeypatch.setattr("app.commands.project._run_speaker_apply_preview_command", fake_run_preview_command)
    monkeypatch.setattr("app.commands.project._remember_prompt_history", remembered.append)

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--sample-count", "1"],
        input="/more\n/audio\n欧丁\n敬悦\n",
    )

    mapping = json.loads((project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert played == [["audio-player"]]
    assert remembered == ["/more", "/audio"]
    assert captured["audio"] == audio_clip
    assert captured["audio_start"] == 0.0
    assert captured["audio_duration"] == 0.0
    assert audio_segments == ["再补一句。"]
    assert "Preview sample for Speaker A: [00:00:01.900 - 00:00:02.500] 再补一句。" in result.output
    assert "Starting audio preview for Speaker A with 1 displayed sample(s)." in result.output
    assert "Controls: Space/P pauses, Q/Esc stops early, Ctrl-C also stops." in result.output
    assert "/video" not in result.output
    assert mapping == {"0": "欧丁", "1": "敬悦"}


def test_speaker_apply_audio_preview_clip_uses_visible_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audio preview clips should be built from the visible sample batch only."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    context = project_commands.SpeakerApplyPreviewContext(
        project_root=tmp_path,
        video=source,
    )
    segments = [
        SentenceSegment(1000, 1600, "第一段。", 0, 1),
        SentenceSegment(5000, 9000, "第二段。", 0, 2),
    ]
    calls: list[tuple[Path, float, float]] = []

    def fake_extract_audio_clip(
        input_path: Path,
        output_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
    ) -> Path:
        assert input_path == source
        calls.append((output_path, start_seconds, duration_seconds))
        _write_test_wav(output_path)
        return output_path

    monkeypatch.setattr(project_commands, "extract_audio_clip", fake_extract_audio_clip)

    output = project_commands._build_speaker_apply_audio_preview_clip(
        preview_context=context,
        speaker_label="Speaker A",
        segments=segments,
    )

    assert output == tmp_path / "tmp" / "speaker_apply_preview" / "Speaker_A" / "preview.wav"
    assert [(start, duration) for _, start, duration in calls] == [(0.0, 4.0), (4.0, 6.0)]
    assert [path.name for path, _, _ in calls] == ["clip_001.wav", "clip_002.wav"]
    with wave.open(str(output), "rb") as reader:
        assert reader.getnframes() > 2 * 160


def test_speaker_apply_preview_runner_stops_on_q(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preview controls should terminate the player from the CLI process."""
    process = _FakePreviewProcess()
    stdin = _FakePreviewStdin("q")

    monkeypatch.setattr(project_commands.sys, "stdin", stdin)
    monkeypatch.setattr(project_commands.subprocess, "Popen", lambda command, stdin=None: process)
    monkeypatch.setattr(project_commands.termios, "tcgetattr", lambda fd: ["old"])
    monkeypatch.setattr(project_commands.termios, "tcsetattr", lambda fd, when, settings: None)
    monkeypatch.setattr(project_commands.tty, "setcbreak", lambda fd: None)
    monkeypatch.setattr(project_commands.select, "select", lambda read, write, error, timeout: (read, [], []))

    project_commands._run_speaker_apply_preview_command(["player"])

    assert process.terminated is True
    assert "Preview stopped." in capsys.readouterr().out


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


def test_project_speakers_preview_prefers_corrected_named_subtitle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preview after vocabulary correction should use corrected named subtitles."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "exports").mkdir(exist_ok=True)
    (project_dir / "exports" / "subtitle_named.srt").write_text("named", encoding="utf-8")
    (project_dir / "exports" / "subtitle_named_corrected.srt").write_text("corrected", encoding="utf-8")
    captured: dict[str, Path] = {}

    def fake_build_preview_command(*, video: Path, subtitle: Path, start_seconds: float) -> list[str]:
        captured["subtitle"] = subtitle
        return ["player", str(subtitle)]

    monkeypatch.setattr("app.commands.project.build_preview_command", fake_build_preview_command)

    result = runner.invoke(app, ["project", "speakers", "preview", str(project_dir), "--dry-run"])

    assert result.exit_code == 0
    assert captured["subtitle"] == project_dir.resolve() / "exports" / "subtitle_named_corrected.srt"
    assert "subtitle_named_corrected.srt" in result.output


def test_project_git_init_writes_safe_ignore_file(tmp_path: Path) -> None:
    """Optional Git tracking should ignore heavy generated artifacts."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    project_dir = _sample_project(tmp_path)

    gitignore_path = init_project_git(project_dir)

    content = gitignore_path.read_text(encoding="utf-8")
    assert "source/" in content
    assert "audio/" in content


def _sample_project(tmp_path: Path, *, projects_dir: Path | None = None, title: str = "Demo") -> Path:
    """Create a minimal project for tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = (projects_dir or tmp_path) / "project"
    create_project(
        source,
        title=title,
        projects_dir=projects_dir or tmp_path,
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


def _write_test_wav(path: Path) -> None:
    """Write a tiny mono 16kHz WAV fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 160)


class _FakePreviewStdin:
    """Tiny TTY-like stdin for preview control tests."""

    def __init__(self, text: str) -> None:
        self.text = text

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0

    def read(self, size: int) -> str:
        value = self.text[:size]
        self.text = self.text[size:]
        return value


class _FakePreviewProcess:
    """Tiny process-like object for preview control tests."""

    pid = 12345

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.terminated:
            return -15
        if self.killed:
            return -9
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return self.poll() or 0
