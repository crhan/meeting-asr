"""Tests for project lifecycle helpers."""

from __future__ import annotations

import json
import shutil
import wave
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from app import project_manager
from app.cli import app
from app.commands import project as project_commands
from app.asr_hotwords import AsrHotwordResolution
from app.config import Settings, set_config_value
from app.core.progress import emit_progress
from app.core.project_refs import list_projects, resolve_project_ref
from app.core.project_models import ProjectTranscribeOptions
from app.correction_types import CorrectionEditSummary
from app.infra.dashscope_asr import TranscriptionPollEvent
from app.lexicon_models import LexiconContext
from app.lexicon_store import record_lexicon_contexts
from app.models import SentenceSegment, TranscriptResult
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
    resolve_project_audio_path,
    resolve_project_source_path,
    save_manifest,
    summarize_project,
)
from app.presentation.cli.project_list import _project_list_timestamp
from app.meeting_summary import MeetingSummary
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_default_lexicon_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep project tests away from the developer's real XDG state."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    isolated_db = tmp_path / "lexicon" / "lexicon.sqlite"
    monkeypatch.setattr(
        "app.transcript_corrections.default_lexicon_db_path", lambda: isolated_db
    )
    monkeypatch.setattr(
        "app.lexicon_store.default_lexicon_db_path", lambda: isolated_db
    )


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
    assert loaded.source.meeting_time == "2026-04-29T15:07:42+08:00"
    assert resolve_project_source_path(project_dir, loaded) == copied_source.resolve()
    assert loaded.source.sha256 is not None
    assert loaded.title_source == "manual"
    assert loaded.title_model is None


def test_resolve_project_audio_path_prefers_asr_audio_timeline(tmp_path: Path) -> None:
    """ASR-timed clip extraction should use project audio, not original media."""
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source")
    project_dir = tmp_path / "projects" / "audio-path"
    create_project(
        source,
        title="Audio Path",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {
        "path": "audio/audio.flac",
        "format": "flac",
        "duration_seconds": 10.0,
    }
    save_manifest(project_dir, manifest)

    assert (
        resolve_project_audio_path(project_dir, load_manifest(project_dir))
        == audio_path.resolve()
    )


def test_resolve_project_audio_path_falls_back_to_source(tmp_path: Path) -> None:
    """Old projects without extracted audio should still be playable."""
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source")
    project_dir = tmp_path / "projects" / "source-fallback"
    create_project(
        source,
        title="Source Fallback",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    manifest = load_manifest(project_dir)

    assert resolve_project_audio_path(
        project_dir, manifest
    ) == resolve_project_source_path(project_dir, manifest)


def test_prepare_project_audio_removes_staged_video_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audio preparation should prune the project video copy without touching user input."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "projects" / "pruned-video"
    create_project(
        source,
        title="Pruned Video",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    copied_source = project_dir / "source" / "meeting.mp4"

    def fake_extract_audio_for_asr(
        input_path: Path, output_path: Path, *, audio_format: str
    ) -> Path:
        """
        Write a fake extracted audio file.

        Args:
            input_path: Source media path.
            output_path: Destination audio path.
            audio_format: Requested audio format.

        Returns:
            Destination path.
        """
        assert Path(input_path) == copied_source.resolve()
        assert audio_format == "flac"
        Path(output_path).write_bytes(b"audio")
        return Path(output_path)

    monkeypatch.setattr(
        project_manager, "extract_audio_for_asr", fake_extract_audio_for_asr
    )

    audio_path = project_manager.prepare_project_audio(project_dir, audio_format="flac")
    manifest = load_manifest(project_dir)

    assert audio_path == project_dir / "audio" / "audio.flac"
    assert audio_path.exists()
    assert source.exists()
    assert not copied_source.exists()
    assert (
        manifest.runtime["source_cleanup"]["removed_project_source"]
        == "source/meeting.mp4"
    )
    assert manifest.runtime["source_cleanup"]["kept_audio"] == "audio/audio.flac"


def test_prepare_project_audio_keeps_staged_audio_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audio preparation should not delete an already-audio project source."""
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"fake wav")
    project_dir = tmp_path / "projects" / "kept-audio"
    create_project(
        source,
        title="Kept Audio",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    copied_source = project_dir / "source" / "meeting.wav"

    def fake_extract_audio_for_asr(
        input_path: Path, output_path: Path, *, audio_format: str
    ) -> Path:
        """
        Write a fake normalized audio file.

        Args:
            input_path: Source media path.
            output_path: Destination audio path.
            audio_format: Requested audio format.

        Returns:
            Destination path.
        """
        assert Path(input_path) == copied_source.resolve()
        Path(output_path).write_bytes(b"audio")
        return Path(output_path)

    monkeypatch.setattr(
        project_manager, "extract_audio_for_asr", fake_extract_audio_for_asr
    )

    project_manager.prepare_project_audio(project_dir, audio_format="flac")
    manifest = load_manifest(project_dir)

    assert copied_source.exists()
    assert source.exists()
    assert "source_cleanup" not in manifest.runtime


def test_ensure_project_audio_reuses_audio_when_source_was_pruned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reruns should use existing project audio even when the staged video is gone."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "projects" / "audio-rerun"
    create_project(
        source,
        title="Audio Rerun",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.write_bytes(b"audio")
    (project_dir / "source" / "meeting.mp4").unlink()
    manifest = load_manifest(project_dir)
    events = []

    def fail_extract_audio_for_asr(*args, **kwargs) -> Path:
        raise AssertionError("existing project audio should be reused")

    monkeypatch.setattr(
        project_manager, "extract_audio_for_asr", fail_extract_audio_for_asr
    )

    resolved = project_manager._ensure_project_audio(
        project_paths(project_dir), manifest, "wav", events.append
    )

    saved = load_manifest(project_dir)
    assert resolved == audio_path.resolve()
    assert saved.audio["path"] == "audio/audio.flac"
    assert saved.audio["format"] == "flac"
    assert events[-1].description == "Using existing project audio"


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

    project_dirs = [
        path
        for path in (data_home / "meeting-asr" / "projects").iterdir()
        if path.is_dir()
    ]
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

    result = runner.invoke(
        app, ["project", "create", str(source), "--project-dir", str(project_dir)]
    )

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

    def fake_summarize_project(
        project_dir, model=None, update_title=True, progress=None
    ):
        manifest = load_manifest(project_dir)
        manifest.title = "自动会议标题"
        save_manifest(project_dir, manifest)
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(
            project_dir, "自动会议标题", summary_path, json_path, "qwen-test", True
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )

    result = runner.invoke(
        app, ["project", "run", str(source), "--projects-dir", str(projects_dir)]
    )

    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)
    transcript = project_dir / "exports" / "transcript_named.txt"
    assert result.exit_code == 0
    assert "Project automation completed." in result.output
    assert "Title" in result.output
    assert "自动会议标题" in result.output
    assert "exports/meeting_summary.md" in result.output
    assert (
        str((project_dir / "exports" / "meeting_summary.md").resolve()) in result.output
    )
    assert "Voiceprint matches" in result.output
    assert "2/2 matched | below-threshold 0 | no-candidate 0" in result.output
    assert f"meeting-asr voiceprint review {manifest.project_id}" in result.output
    assert f"meeting-asr project correct edit {manifest.project_id}" in result.output
    assert (
        f"meeting-asr project transcript show {manifest.project_id} --kind corrected"
        in result.output
    )
    assert "欧丁" in transcript.read_text(encoding="utf-8")
    assert "敬悦" in transcript.read_text(encoding="utf-8")


def test_project_run_stabilizes_sentence_speakers_after_voiceprint_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should trigger sentence-level stabilization after accepted identity matches."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"
    calls: dict[str, object] = {}

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
                SpeakerMatch(
                    0,
                    "Speaker A",
                    "欧丁",
                    0.91,
                    True,
                    2,
                    accepted_person_id=1,
                    accepted_person_public_id="vpp-0000000000000001",
                ),
                SpeakerMatch(
                    1,
                    "Speaker B",
                    "敬悦",
                    0.88,
                    True,
                    1,
                    accepted_person_id=2,
                    accepted_person_public_id="vpp-0000000000000002",
                ),
            ],
        )

    def fake_stabilize_project_speakers(project_dir, **kwargs):
        calls["project_dir"] = project_dir
        calls["kwargs"] = kwargs
        return SimpleNamespace(reassignment_count=2, final_match_summary=None)

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )
    monkeypatch.setattr(
        project_commands, "stabilize_project_speakers", fake_stabilize_project_speakers
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--no-polish",
            "--no-summarize",
            "--no-progress",
            "--speaker-sample-workers",
            "7",
        ],
    )

    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    assert result.exit_code == 0
    assert calls["project_dir"] == project_dir
    assert calls["kwargs"] == {
        "store_dir": None,
        "model": None,
        "iterations": 2,
        "sample_workers": 7,
        "progress": None,
    }
    assert "reassigned 2 sentence(s)" in result.output


def test_project_run_agent_log_prints_project_id_before_polling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Agent logs should expose project identity before long ASR polling."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        manifest = load_manifest(project_dir)
        emit_progress(
            progress,
            "Waiting for DashScope ASR",
            log_kind="stage",
            stage="ASR polling",
            project_id=manifest.project_id,
            project_path=str(project_dir),
            input_file=str(source),
            timestamp="2026-05-06T10:00:00+08:00",
        )
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 3)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--no-polish",
            "--no-summarize",
            "--agent-log",
            "--no-progress",
        ],
    )

    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)
    assert result.exit_code == 0
    assert f"project_id={manifest.project_id}" in result.output
    assert "stage=project created" in result.output
    assert "stage=ASR polling" in result.output
    assert str(project_dir) in result.output


def test_project_run_human_output_suppresses_structured_agent_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default human output should not print stage/heartbeat log lines."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        manifest = load_manifest(project_dir)
        emit_progress(
            progress,
            "Waiting for DashScope ASR",
            log_kind="stage",
            stage="ASR polling",
            project_id=manifest.project_id,
            project_path=str(project_dir),
            input_file=str(source),
            timestamp="2026-05-06T10:00:00+08:00",
        )
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 3)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--no-polish",
            "--no-summarize",
        ],
    )

    assert result.exit_code == 0
    assert "stage=project created" not in result.output
    assert "stage=ASR polling" not in result.output
    assert "heartbeat=" not in result.output


def test_project_run_applies_local_lexicon_corrections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should apply accepted local lexicon rules before final outputs."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"
    lexicon_db = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts(
        [
            LexiconContext(
                canonical="iSee",
                wrong_text="IC",
                corrected_text="iSee",
                left_context="今天聊",
                right_context="产品",
                category="system",
                speaker_name="欧丁",
                project_id="p-test",
                sentence_id=1,
                source="test",
            )
        ],
        db_path=lexicon_db,
    )
    monkeypatch.setattr(
        "app.transcript_corrections.default_lexicon_db_path", lambda: lexicon_db
    )

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sentences(
            project_dir / "asr" / "sentences.json",
            [
                {
                    "begin_time_ms": 0,
                    "end_time_ms": 1000,
                    "text": "今天聊IC产品。",
                    "speaker_id": 0,
                    "sentence_id": 1,
                },
                {
                    "begin_time_ms": 1100,
                    "end_time_ms": 2000,
                    "text": "PIC保持不变。",
                    "speaker_id": 0,
                    "sentence_id": 2,
                },
            ],
        )
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 2)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--no-polish",
            "--no-summarize",
        ],
    )

    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)
    corrected = json.loads(
        (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    )
    auto_show = runner.invoke(
        app,
        [
            "project",
            "transcript",
            "show",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Local correction" in result.output
    assert "applied (1 sentence(s), 1 rule(s))" in result.output
    assert manifest.runtime["local_correction"]["status"] == "applied"
    assert (
        manifest.outputs["corrected_named_transcript"]
        == "exports/transcript_named_corrected.txt"
    )
    assert corrected["sentences"][0]["text"] == "今天聊iSee产品。"
    assert corrected["sentences"][1]["text"] == "PIC保持不变。"
    assert auto_show.exit_code == 0
    assert "今天聊iSee产品。" in auto_show.output
    assert "PIC保持不变。" in auto_show.output


def test_asr_polling_heartbeat_redacts_signed_url_query_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ASR polling heartbeats must not leak signed URL query tokens."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {
        "path": "audio/audio.wav",
        "format": "wav",
        "duration_seconds": 10,
    }
    save_manifest(project_dir, manifest)
    events = []

    class FakeTask:
        """Minimal DashScope task response."""

        output = {"task_id": "task-1234567890"}

    def fake_wait_transcription(*, settings, task, poll_callback=None):
        if poll_callback is not None:
            poll_callback(
                TranscriptionPollEvent(
                    status="RUNNING", elapsed_seconds=31.0, wait_seconds=5.0
                )
            )
        return object()

    monkeypatch.setattr(project_manager, "load_settings", lambda **_: _settings())
    monkeypatch.setattr(
        project_manager,
        "upload_file_to_oss",
        lambda *_, **__: (
            "https://oss.example.com/audio.wav?Signature=secret-token&Expires=1"
        ),
    )
    monkeypatch.setattr(project_manager, "record_oss_upload", lambda *_, **__: None)
    monkeypatch.setattr(project_manager, "submit_transcription", lambda **_: FakeTask())
    monkeypatch.setattr(project_manager, "wait_transcription", fake_wait_transcription)
    monkeypatch.setattr(
        project_manager, "download_transcription_json", lambda _: {"raw": True}
    )
    monkeypatch.setattr(
        project_manager,
        "parse_transcription_result",
        lambda _: TranscriptResult(
            "大家好。",
            [SentenceSegment(0, 1000, "大家好。", 0, 1)],
            [0],
        ),
    )
    monkeypatch.setattr(
        project_manager,
        "_resolve_project_asr_hotwords",
        lambda settings, options: AsrHotwordResolution(None, "disabled"),
    )
    monkeypatch.setattr(project_manager, "record_dashscope_wait", lambda **_: None)

    project_manager.transcribe_project(
        project_dir, _transcribe_options(), progress=events.append
    )

    heartbeat_events = [
        event
        for event in events
        if event.log_kind == "heartbeat" and event.stage == "ASR polling"
    ]
    assert heartbeat_events
    rendered = "\n".join(
        " ".join(f"{key}={value}" for key, value in event.log_fields)
        for event in events
        if event.log_kind
    )
    assert "secret-token" not in rendered
    assert "Signature=" not in rendered
    assert "dashscope_task_id=task-1234567890" in rendered


def test_transcribe_project_snapshots_lexicon_asr_hotwords(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ASR submit must record the lexicon hotwords into corrections/asr_hotwords.json."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {
        "path": "audio/audio.wav",
        "format": "wav",
        "duration_seconds": 10,
    }
    save_manifest(project_dir, manifest)

    lexicon_db = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts(
        [
            LexiconContext(
                canonical="iSee",
                wrong_text="艾赛",
                corrected_text="iSee",
                left_context="",
                right_context="系统",
                category="system",
                speaker_name="敬悦",
                project_id="p-demo",
                sentence_id=1,
                source="test",
            )
        ],
        db_path=lexicon_db,
    )

    class FakeTask:
        """Minimal DashScope task response."""

        output = {"task_id": "task-1"}

    submitted: dict = {}

    def fake_submit_transcription(**kwargs):
        submitted.update(kwargs)
        return FakeTask()

    monkeypatch.setattr(project_manager, "load_settings", lambda **_: _settings())
    monkeypatch.setattr(
        project_manager,
        "upload_file_to_oss",
        lambda *_, **__: "https://oss.example.com/audio.wav",
    )
    monkeypatch.setattr(project_manager, "record_oss_upload", lambda *_, **__: None)
    monkeypatch.setattr(
        project_manager, "submit_transcription", fake_submit_transcription
    )
    monkeypatch.setattr(project_manager, "wait_transcription", lambda **_: object())
    monkeypatch.setattr(
        project_manager, "download_transcription_json", lambda _: {"raw": True}
    )
    monkeypatch.setattr(
        project_manager,
        "parse_transcription_result",
        lambda _: TranscriptResult(
            "大家好。", [SentenceSegment(0, 1000, "大家好。", 0, 1)], [0]
        ),
    )
    monkeypatch.setattr(project_manager, "record_dashscope_wait", lambda **_: None)
    # Point auto resolution at the test lexicon and stub the remote sync so a
    # vocabulary id is returned without any DashScope call.
    monkeypatch.setattr("app.asr_hotwords.default_lexicon_db_path", lambda: lexicon_db)
    monkeypatch.setattr(
        "app.asr_hotwords.sync_asr_hotwords",
        lambda **_: SimpleNamespace(
            vocabulary_id="vocab-test", hotword_count=1, vocabulary_hash="hash"
        ),
    )

    project_manager.transcribe_project(project_dir, _transcribe_options())

    artifact = project_dir / "corrections" / "asr_hotwords.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["dashscope_vocabulary"] == [{"text": "iSee", "weight": 4}]
    assert submitted["vocabulary_id"] == "vocab-test"


def test_project_run_generates_default_transcript_polish_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should prepare a reviewable transcript polish proposal after ASR."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"
    calls = {}

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 3)

    def fake_summarize_project(
        project_dir, model=None, update_title=True, progress=None
    ):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(
            project_dir, "自动会议标题", summary_path, json_path, "qwen-test", False
        )

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    def fake_prepare_polish(
        project_dir,
        correction_model,
        polish_concurrency=None,
        polish_legacy=False,
        progress=None,
    ):
        calls["model"] = correction_model
        calls["polish_concurrency"] = polish_concurrency
        calls["polish_legacy"] = polish_legacy
        proposal_dir = project_dir / "tmp" / "corrections"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        review_path = proposal_dir / "review_polish_test.md"
        proposal_path = proposal_dir / "proposal_test.md"
        diff_path = proposal_dir / "proposal_test.diff"
        json_path = proposal_dir / "proposal_test.json"
        for path in (review_path, proposal_path, diff_path, json_path):
            path.write_text("proposal\n", encoding="utf-8")
        return CorrectionEditSummary(
            review_path=review_path,
            proposal_path=proposal_path,
            proposal_diff_path=diff_path,
            proposal_json_path=json_path,
            change_count=0,
            sample_change_count=0,
            proposed_change_count=1,
            learned_count=0,
            accepted=False,
            model=correction_model,
            model_error=None,
            understanding=[],
            corrected_sentences_path=None,
            corrected_transcript_path=None,
            corrected_named_transcript_path=None,
            corrected_srt_path=None,
            hotwords_path=None,
            applied_path=None,
            lexicon_db=tmp_path / "lexicon.sqlite",
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )
    monkeypatch.setattr(
        project_commands, "_prepare_run_transcript_polish", fake_prepare_polish
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--correction-model",
            "qwen-test",
            "--polish-concurrency",
            "3",
        ],
    )
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert calls["model"] == "qwen-test"
    assert calls["polish_concurrency"] == 3
    assert manifest.runtime["polish"]["status"] == "proposal_ready"
    assert manifest.runtime["polish"]["proposed_changes"] == 1
    assert (
        manifest.runtime["polish"]["proposal_diff"]
        == "tmp/corrections/proposal_test.diff"
    )
    assert "Transcript polish" in result.output
    assert "proposal ready (1 change(s))" in result.output
    assert "Transcript polish proposal" in result.output
    assert f"meeting-asr project correct diff {manifest.project_id}" in result.output
    assert f"meeting-asr project correct accept {manifest.project_id}" in result.output


def test_project_run_auto_accepts_polish_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should avoid review handoff when polish auto-accept is configured."""
    set_config_value("correction.polish_auto_accept", "true")
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 3)

    def fake_summarize_project(
        project_dir, model=None, update_title=True, progress=None
    ):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(
            project_dir, "自动会议标题", summary_path, json_path, "qwen-test", False
        )

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    def fake_prepare_polish(
        project_dir,
        correction_model,
        polish_concurrency=None,
        polish_legacy=False,
        progress=None,
    ):
        proposal_dir = project_dir / "tmp" / "corrections"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        review_path = proposal_dir / "review_polish_test.md"
        proposal_path = proposal_dir / "proposal_test.md"
        diff_path = proposal_dir / "proposal_test.diff"
        json_path = proposal_dir / "proposal_test.json"
        for path in (review_path, proposal_path, diff_path, json_path):
            path.write_text("proposal\n", encoding="utf-8")
        return CorrectionEditSummary(
            review_path=review_path,
            proposal_path=proposal_path,
            proposal_diff_path=diff_path,
            proposal_json_path=json_path,
            change_count=0,
            sample_change_count=0,
            proposed_change_count=2,
            learned_count=0,
            accepted=False,
            model="qwen-test",
            model_error=None,
            understanding=[],
            corrected_sentences_path=None,
            corrected_transcript_path=None,
            corrected_named_transcript_path=None,
            corrected_srt_path=None,
            hotwords_path=None,
            applied_path=None,
            lexicon_db=tmp_path / "lexicon.sqlite",
        )

    def fake_accept_polish(project_dir, summary):
        manifest = load_manifest(project_dir)
        corrected_path = project_dir / "exports" / "transcript_named_corrected.txt"
        corrected_srt = project_dir / "exports" / "subtitle_named_corrected.srt"
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_path.write_text("欧丁: 修正后的内容。\n", encoding="utf-8")
        corrected_srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n修正后的内容。\n", encoding="utf-8"
        )
        manifest.status = "corrected"
        manifest.outputs["corrected_named_transcript"] = (
            "exports/transcript_named_corrected.txt"
        )
        manifest.outputs["corrected_named_subtitle"] = (
            "exports/subtitle_named_corrected.srt"
        )
        save_manifest(project_dir, manifest)
        return CorrectionEditSummary(
            review_path=summary.review_path,
            proposal_path=summary.proposal_path,
            proposal_diff_path=summary.proposal_diff_path,
            proposal_json_path=summary.proposal_json_path,
            change_count=2,
            sample_change_count=0,
            proposed_change_count=2,
            learned_count=0,
            accepted=True,
            model=summary.model,
            model_error=None,
            understanding=[],
            corrected_sentences_path=project_dir / "asr" / "sentences_corrected.json",
            corrected_transcript_path=project_dir
            / "exports"
            / "transcript_corrected.txt",
            corrected_named_transcript_path=corrected_path,
            corrected_srt_path=corrected_srt,
            hotwords_path=project_dir / "corrections" / "asr_hotwords.json",
            applied_path=project_dir / "corrections" / "applied.json",
            lexicon_db=summary.lexicon_db,
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )
    monkeypatch.setattr(
        project_commands, "_prepare_run_transcript_polish", fake_prepare_polish
    )
    monkeypatch.setattr(
        project_commands, "_accept_run_transcript_polish", fake_accept_polish
    )

    result = runner.invoke(
        app, ["project", "run", str(source), "--projects-dir", str(projects_dir)]
    )
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0, result.output
    assert manifest.runtime["polish"]["status"] == "accepted"
    assert manifest.runtime["polish"]["accepted_changes"] == 2
    assert "accepted (2/2 change(s))" in result.output
    assert "Transcript polish proposal" not in result.output
    assert (
        f"meeting-asr project correct diff {manifest.project_id}" not in result.output
    )
    assert (
        f"meeting-asr project correct accept {manifest.project_id}" not in result.output
    )
    assert "exports/transcript_named_corrected.txt" in result.output


def test_project_run_polish_failure_prints_recovery_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project run should explain polish batch failures without aborting the run."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    projects_dir = tmp_path / "projects"

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        _write_sample_sentences(project_dir / "asr" / "sentences.json")
        return ProjectTranscribeSummary(project_dir, "task-1", "test", 1, 3)

    def fake_match_project_speakers(project_dir, **kwargs):
        return SpeakerMatchSummary(
            project_dir / "speakers" / "speaker_matches.json",
            "fake-provider",
            "fake-model",
            0.75,
            [SpeakerMatch(0, "Speaker A", "欧丁", 0.91, True, 2)],
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(
            dashscope_api_key="key",
            dashscope_base_url=None,
            dashscope_correction_model="qwen-test",
        ),
    )
    monkeypatch.setattr(
        "app.transcript_corrections.propose_transcript_polish",
        lambda **_: (_ for _ in ()).throw(TimeoutError("read timeout")),
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(source),
            "--projects-dir",
            str(projects_dir),
            "--no-summarize",
            "--correction-model",
            "qwen-test",
            "--no-progress",
            "--legacy-polish",
        ],
    )
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert "Transcript polish" in result.output
    assert f"project_id={manifest.project_id}" in result.output
    assert "stage=polish" in result.output
    assert "batch=1/1" in result.output
    assert "model=qwen-test" in result.output
    assert "timeout=120s" in result.output
    assert f"meeting-asr project show {manifest.project_id}" in result.output
    assert f"meeting-asr project review {manifest.project_id}" in result.output
    assert f"meeting-asr project correct polish {manifest.project_id}" in result.output


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
                SpeakerMatch(
                    1,
                    "Speaker B",
                    None,
                    0.12,
                    False,
                    1,
                    best_name="敬悦",
                    best_score=0.12,
                    threshold=0.75,
                ),
            ],
        )

    def fake_summarize_project(
        project_dir, model=None, update_title=True, progress=None
    ):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# 自动会议标题\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return ProjectMeetingSummary(
            project_dir, "自动会议标题", summary_path, json_path, "qwen-test", False
        )

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)
    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    monkeypatch.setattr(
        project_commands, "match_project_speakers", fake_match_project_speakers
    )

    result = runner.invoke(
        app, ["project", "run", str(source), "--projects-dir", str(projects_dir)]
    )
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert "Project automation needs review." in result.output
    assert "Voiceprint matches" in result.output
    assert "1/2 matched | below-threshold 1 | no-candidate 0" in result.output
    assert "Voiceprint threshold" in result.output
    assert "auto accept >= 0.750" in result.output
    assert "Voiceprint candidates" in result.output
    assert "Speaker B" in result.output
    assert "below-threshold" in result.output
    assert "best: 敬悦" in result.output
    assert "0.120" in result.output
    assert "partial" in result.output
    assert f"meeting-asr project review {manifest.project_id}" in result.output
    assert (
        "This opens the human review workflow for unresolved speakers." in result.output
    )
    assert (
        f"meeting-asr project speakers inspect {manifest.project_id} --sample-count 5"
        in result.output
    )
    assert (
        f"meeting-asr project speakers apply {manifest.project_id} --map 0=Name"
        in result.output
    )
    assert result.output.index(
        f"meeting-asr project review {manifest.project_id}"
    ) < result.output.index(
        f"meeting-asr project speakers apply {manifest.project_id} --map 0=Name"
    )
    assert f"meeting-asr voiceprint review {manifest.project_id}" in result.output
    assert "meeting-asr voiceprint embed" in result.output
    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert mapping == {"0": "欧丁"}
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
            ["目标牵引", "路径收敛", "飞轮闭环", "里程碑518", "团队对齐"],
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
    assert manifest.title_source == "llm"
    assert manifest.title_model == "qwen-test"
    assert manifest.outputs["meeting_summary"] == "exports/meeting_summary.md"
    assert manifest.outputs["meeting_summary_json"] == "exports/meeting_summary.json"
    assert manifest.asr["summary_model"] == "qwen-test"
    rendered = summary.summary_path.read_text(encoding="utf-8")
    assert "讨论 AI 转型目标" in rendered
    assert "待办" not in rendered


def test_summarize_project_prefixes_auto_title_with_meeting_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LLM-generated meeting titles should keep the meeting time prefix."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title=None,
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time="2026-05-02T10:00:00+08:00",
        hash_source=False,
    )
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    monkeypatch.setattr(
        "app.project_manager.generate_meeting_summary",
        lambda result, settings, model: MeetingSummary(
            "AI 转型研讨",
            "讨论 AI 转型目标和落地路径。",
            ["目标牵引", "路径收敛", "飞轮闭环", "里程碑518", "团队对齐"],
            "qwen-test",
        ),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings", lambda require_oss=False: object()
    )

    summarize_project(project_dir, model=None, update_title=True)
    manifest = load_manifest(project_dir)

    assert manifest.title == "2026-05-02 10:00 AI 转型研讨"
    assert manifest.title_source == "llm"


def test_summarize_project_replaces_existing_llm_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A rerun may improve an existing LLM-generated title without touching manual titles."""
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
    manifest = load_manifest(project_dir)
    manifest.title = "旧自动标题"
    manifest.title_source = "llm"
    manifest.title_model = "qwen-old"
    save_manifest(project_dir, manifest)
    monkeypatch.setattr(
        "app.project_manager.generate_meeting_summary",
        lambda result, settings, model: MeetingSummary(
            "新自动标题", "回忆提示。", ["关键词1", "关键词2"], "qwen-test"
        ),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings", lambda require_oss=False: object()
    )

    summary = summarize_project(project_dir, model=None, update_title=True)
    manifest = load_manifest(project_dir)

    assert summary.title_updated is True
    assert manifest.title == "新自动标题"
    assert manifest.title_source == "llm"
    assert manifest.title_model == "qwen-test"


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
        lambda result, settings, model: MeetingSummary(
            "自动标题", "摘要", [], "qwen-test"
        ),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings",
        lambda require_oss=False: object(),
    )

    summary = summarize_project(project_dir, model=None, update_title=True)

    assert summary.title_updated is False
    manifest = load_manifest(project_dir)
    assert manifest.title == "手工标题"
    assert manifest.title_source == "manual"
    assert manifest.title_model is None


def test_summarize_project_marks_legacy_custom_unknown_title_as_manual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy unknown titles that are not source names should be preserved as manual."""
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
    manifest = load_manifest(project_dir)
    manifest.title = "旧自定义标题"
    manifest.title_source = "unknown"
    manifest.title_model = None
    save_manifest(project_dir, manifest)
    monkeypatch.setattr(
        "app.project_manager.generate_meeting_summary",
        lambda result, settings, model: MeetingSummary(
            "自动标题", "摘要", [], "qwen-test"
        ),
    )
    monkeypatch.setattr(
        "app.project_manager.load_settings", lambda require_oss=False: object()
    )

    summary = summarize_project(project_dir, model=None, update_title=True)

    assert summary.title_updated is False
    manifest = load_manifest(project_dir)
    assert manifest.title == "旧自定义标题"
    assert manifest.title_source == "manual"
    assert manifest.title_model is None


def test_project_summarize_command_prints_absolute_memory_index_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project memory-index command should print directly openable artifact paths."""
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

    def fake_summarize_project(
        project_dir, model=None, update_title=True, progress=None
    ):
        summary_path = project_dir / "exports" / "meeting_summary.md"
        json_path = project_dir / "exports" / "meeting_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# Demo\n", encoding="utf-8")
        json_path.write_text('{"model":"qwen-test"}\n', encoding="utf-8")
        return ProjectMeetingSummary(
            project_dir, "Demo", summary_path, json_path, "qwen-test", False
        )

    monkeypatch.setattr(project_commands, "summarize_project", fake_summarize_project)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "project",
            "summarize",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert (
        f"Memory index: {(project_dir / 'exports' / 'meeting_summary.md').resolve()}"
        in result.output
    )
    assert (
        f"Memory index JSON: {(project_dir / 'exports' / 'meeting_summary.json').resolve()}"
        in result.output
    )


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
    manifest.meeting_keywords = ["旧关键字1", "旧关键字2"]
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
    assert manifest.meeting_keywords == []
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


def test_project_rerun_command_uses_existing_project_transcription_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The explicit rerun entrypoint should call the reusable project ASR path."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)
    captured: dict[str, object] = {}

    def fake_transcribe_project(project_dir, options, progress=None, **kwargs):
        captured["project_dir"] = project_dir
        captured["options"] = options
        captured["progress"] = progress
        return ProjectTranscribeSummary(project_dir, "task-rerun", "test", 6, 12)

    monkeypatch.setattr(project_commands, "transcribe_project", fake_transcribe_project)

    result = runner.invoke(
        app,
        [
            "project",
            "rerun",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--speaker-count",
            "6",
            "--no-progress",
        ],
    )

    options = captured["options"]
    assert result.exit_code == 0
    assert captured["project_dir"] == project_dir.resolve()
    assert isinstance(options, ProjectTranscribeOptions)
    assert options.speaker_count == 6
    assert options.audio_format == "flac"
    assert captured["progress"] is None
    assert "Project transcription completed." in result.output
    assert "task-rerun" in result.output


def test_resolve_project_ref_accepts_path_id_title_and_unique_partial(
    tmp_path: Path,
) -> None:
    """Project references should use stable ids, paths, titles, or unique title fragments."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(
        tmp_path, projects_dir=projects_dir, title="Project Ref Demo"
    )
    manifest = load_manifest(project_dir)

    assert resolve_project_ref(project_dir, projects_dir) == project_dir.resolve()
    assert (
        resolve_project_ref(manifest.project_id, projects_dir) == project_dir.resolve()
    )
    assert (
        resolve_project_ref("Project Ref Demo", projects_dir) == project_dir.resolve()
    )
    assert resolve_project_ref("Ref Demo", projects_dir) == project_dir.resolve()


def test_project_list_order_follows_meeting_time_desc(tmp_path: Path) -> None:
    """Project list should show later meetings first and untimed projects last."""
    projects_dir = tmp_path / "projects"
    early_source = tmp_path / "early.mp4"
    late_source = tmp_path / "late.mp4"
    untimed_source = tmp_path / "untimed.mp4"
    early_source.write_bytes(b"early video")
    late_source.write_bytes(b"late video")
    untimed_source.write_bytes(b"untimed video")
    early = projects_dir / "early"
    late = projects_dir / "late"
    untimed = projects_dir / "untimed"
    create_project(
        early_source,
        title="Early Meeting",
        projects_dir=projects_dir,
        project_dir=early,
        meeting_time="2026-05-02T10:00:00+08:00",
        hash_source=False,
    )
    create_project(
        late_source,
        title="Late Meeting",
        projects_dir=projects_dir,
        project_dir=late,
        meeting_time="2026-05-03T10:00:00+08:00",
        hash_source=False,
    )
    create_project(
        untimed_source,
        title="Untimed Project",
        projects_dir=projects_dir,
        project_dir=untimed,
        meeting_time=None,
        hash_source=False,
    )
    early_manifest = load_manifest(early)
    late_manifest = load_manifest(late)
    untimed_manifest = load_manifest(untimed)
    early_manifest.created_at = "2026-05-03T10:00:00+08:00"
    late_manifest.created_at = "2026-05-01T10:00:00+08:00"
    untimed_manifest.created_at = "2026-05-04T10:00:00+08:00"
    save_manifest(early, early_manifest)
    save_manifest(late, late_manifest)
    save_manifest(untimed, untimed_manifest)

    projects = list_projects(projects_dir).projects

    assert [project.title for project in projects] == [
        "2026-05-03 10:00 Late Meeting",
        "2026-05-02 10:00 Early Meeting",
        "Untimed Project",
    ]


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
    assert "meeting-asr project show PROJECT_ID" in result.output
    assert "Project ID" in result.output
    assert "Meeting (Local)" not in result.output
    assert "Updated (Local)" not in result.output
    assert project_id in result.output
    assert "No." not in result.output
    assert "State" in result.output
    assert "Demo" in result.output
    assert "Created" in result.output
    assert "transcribe 1" not in result.output


def test_project_list_command_keeps_time_in_title_only(tmp_path: Path) -> None:
    """Human project list should avoid redundant meeting/update columns."""
    projects_dir = tmp_path / "projects"
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    create_project(
        source,
        title="Timed Demo",
        projects_dir=projects_dir,
        project_dir=projects_dir / "timed-demo",
        meeting_time="2026-05-02T10:00:00+08:00",
        hash_source=False,
    )

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir)]
    )

    assert result.exit_code == 0
    assert "Meeting (Local)" not in result.output
    assert "Updated (Local)" not in result.output
    assert "2026-05-02 10:00 Timed Demo" in result.output


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
        meeting_time="2026-05-02T10:00:00+08:00",
        hash_source=False,
    )

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir), "--json"]
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["projects_dir"] == str(projects_dir.resolve())
    assert payload["count"] == 1
    assert payload["projects"][0]["title"] == "2026-05-02 10:00 Demo"
    assert payload["projects"][0]["meeting_time"] == "2026-05-02T10:00:00+08:00"
    assert payload["projects"][0]["project_dir"] == str(project_dir.resolve())
    assert payload["projects"][0]["status"] == "created"
    assert payload["projects"][0]["workflow"]["state"] == "Created"
    assert payload["projects"][0]["workflow"]["artifacts"] == []
    assert payload["projects"][0]["workflow"]["next_command_short"] == (
        f"transcribe {load_manifest(project_dir).project_id}"
    )
    # Fresh projects have no keywords yet but the field is always present
    # so scripts consuming --json can treat it as a stable schema.
    assert payload["projects"][0]["meeting_keywords"] == []


def test_project_list_json_includes_meeting_keywords(tmp_path: Path) -> None:
    """Project list --json must expose summary keywords once a project has them."""
    projects_dir = tmp_path / "projects"
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = projects_dir / "kw"
    create_project(
        source,
        title="Demo",
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time="2026-05-02T10:00:00+08:00",
        hash_source=False,
    )
    manifest = load_manifest(project_dir)
    manifest.meeting_keywords = ["飞轮POC本地跑通", "诊断准确率30%", "A3A6A8"]
    save_manifest(project_dir, manifest)

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir), "--json"]
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["projects"][0]["meeting_keywords"] == [
        "飞轮POC本地跑通",
        "诊断准确率30%",
        "A3A6A8",
    ]


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

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir)]
    )

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "Demo" in result.output
    assert load_manifest(project_dir).project_id in result.output
    assert "Created" in result.output
    assert "not-a-project" not in result.output


def test_project_list_command_handles_empty_projects_dir(tmp_path: Path) -> None:
    """Project list should treat a missing projects parent as empty."""
    projects_dir = tmp_path / "missing"

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir)]
    )

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert "No projects found." in result.output


def test_project_update_command_changes_title_and_meeting_time(tmp_path: Path) -> None:
    """Project update should change editable manifest metadata."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(
        tmp_path, projects_dir=projects_dir, title="Old Title"
    )
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
    assert "Title: 2026-05-02 10:00 New Title" in result.output
    assert manifest.title == "2026-05-02 10:00 New Title"
    assert manifest.title_source == "manual"
    assert manifest.title_model is None
    assert manifest.source.meeting_time == "2026-05-02T10:00:00+08:00"


def test_project_update_command_refreshes_title_time_prefix(tmp_path: Path) -> None:
    """Updating only meeting time should repair the title time prefix."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(
        tmp_path,
        projects_dir=projects_dir,
        title="2026-05-01 Old Meeting",
        meeting_time="2026-05-01T09:00:00+08:00",
    )
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "project",
            "update",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--meeting-time",
            "2026-05-02T10:30:00+08:00",
        ],
    )
    manifest = load_manifest(project_dir)

    assert result.exit_code == 0
    assert manifest.title == "2026-05-02 10:30 Old Meeting"
    assert manifest.source.meeting_time == "2026-05-02T10:30:00+08:00"


def test_project_update_requires_a_field(tmp_path: Path) -> None:
    """Project update without updates should fail with a clear message."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir, title="Demo")
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        ["project", "update", manifest.project_id, "--projects-dir", str(projects_dir)],
    )

    assert result.exit_code == 1
    assert "Nothing to update" in result.output


def test_project_delete_moves_project_to_trash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Project delete should move the project to Meeting-ASR trash by default."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(
        tmp_path, projects_dir=projects_dir, title="Delete Me"
    )
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "project",
            "delete",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--yes",
        ],
    )
    list_result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir)]
    )
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
    project_dir = _sample_project(
        tmp_path, projects_dir=projects_dir, title="Delete Me"
    )
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "project",
            "delete",
            manifest.project_id,
            "--projects-dir",
            str(projects_dir),
            "--permanent",
            "--yes",
        ],
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
    assert (
        f"meeting-asr project review {load_manifest(project_dirs[0]).project_id}"
        in second.output
    )


def test_project_create_explicit_dir_does_not_duplicate_existing_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit project dir must not bypass source-based reuse."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")
    projects_dir = data_home / "meeting-asr" / "projects"
    duplicate_dir = projects_dir / "duplicate"

    first = runner.invoke(app, ["project", "create", str(source), "--title", "Demo"])
    second = runner.invoke(
        app,
        [
            "project",
            "create",
            str(source),
            "--project-dir",
            str(duplicate_dir),
            "--title",
            "Duplicate",
        ],
    )

    project_dirs = [path for path in projects_dir.iterdir() if path.is_dir()]
    manifest = load_manifest(project_dirs[0])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert len(project_dirs) == 1
    assert not duplicate_dir.exists()
    assert "Project already exists; reusing it." in second.output
    assert manifest.title == "Duplicate"


def test_project_create_variant_gets_distinct_project_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Experiment variants should be explicit and have separate project ids."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")

    base = runner.invoke(app, ["project", "create", str(source), "--title", "Base"])
    variant = runner.invoke(
        app, ["project", "create", str(source), "--variant", "spk5", "--title", "SPK5"]
    )

    projects = list_projects(data_home / "meeting-asr" / "projects").projects
    manifests = [load_manifest(project.project_dir) for project in projects]
    ids = {manifest.project_id for manifest in manifests}
    variants = {manifest.source.variant for manifest in manifests}
    assert base.exit_code == 0
    assert variant.exit_code == 0
    assert len(projects) == 2
    assert len(ids) == 2
    assert variants == {None, "spk5"}
    assert any(project_id.endswith("-v-spk5") for project_id in ids)


def test_project_create_reuse_applies_explicit_manual_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit title on a reused source should not be silently ignored."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"same video")

    first = runner.invoke(app, ["project", "create", str(source)])
    second = runner.invoke(
        app, ["project", "create", str(source), "--title", "手工复命名"]
    )

    projects_dir = data_home / "meeting-asr" / "projects"
    project_dir = next(path for path in projects_dir.iterdir() if path.is_dir())
    manifest = load_manifest(project_dir)
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert manifest.title == "手工复命名"
    assert manifest.title_source == "manual"
    assert manifest.title_model is None


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
    project_dir = _sample_project(tmp_path, meeting_time="2026-05-02T10:00:00+08:00")

    result = runner.invoke(app, ["project", "status", str(project_dir)])

    assert result.exit_code == 0
    assert "Title: 2026-05-02 10:00 Demo" in result.output
    assert "Meeting time: 2026-05-02T10:00:00+08:00" in result.output
    assert "State: Created" in result.output
    assert "Next: transcribe" in result.output
    assert "Artifacts: -" in result.output
    assert "Source: source/meeting.mp4" in result.output


def test_project_status_command_prints_json(tmp_path: Path) -> None:
    """Project status should be script-friendly."""
    project_dir = _sample_project(tmp_path, meeting_time="2026-05-02T10:00:00+08:00")
    manifest = load_manifest(project_dir)

    result = runner.invoke(app, ["project", "status", str(project_dir), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["project"] == str(project_dir.resolve())
    assert payload["project_id"] == manifest.project_id
    assert payload["title"] == "2026-05-02 10:00 Demo"
    assert payload["meeting_time"] == "2026-05-02T10:00:00+08:00"
    assert payload["workflow"]["state"] == "Created"
    assert payload["source"] == "source/meeting.mp4"


def test_project_status_accepts_project_id(tmp_path: Path) -> None:
    """Project status should resolve project ids from the projects parent."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        ["project", "status", manifest.project_id, "--projects-dir", str(projects_dir)],
    )

    assert result.exit_code == 0
    assert f"Project: {project_dir.resolve()}" in result.output
    assert "Title: Demo" in result.output


def test_project_show_command_summarizes_outputs(tmp_path: Path) -> None:
    """Project show should be the human landing page after project list."""
    project_dir = _sample_project(tmp_path, meeting_time="2026-05-02T10:00:00+08:00")
    manifest = load_manifest(project_dir)
    exports_dir = project_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (exports_dir / "transcript_named.txt").write_text(
        "欧丁: 大家好\n", encoding="utf-8"
    )
    (exports_dir / "subtitle_named.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n欧丁: 大家好\n", encoding="utf-8"
    )
    (exports_dir / "meeting_summary.md").write_text(
        "# Demo\n\n## 摘要\n关键结论。\n", encoding="utf-8"
    )
    manifest.status = "named"
    manifest.outputs.update(
        {
            "meeting_summary": "exports/meeting_summary.md",
            "named_transcript": "exports/transcript_named.txt",
            "named_subtitle": "exports/subtitle_named.srt",
        }
    )
    manifest.asr.update(
        {"provider": "dashscope", "model": "fun-asr", "task_id": "task-1"}
    )
    manifest.audio["duration_seconds"] = 125
    manifest.speakers.update({"detected_ids": [0], "mapped": {"0": "欧丁"}})
    manifest.runtime = {
        "current_stage": "ASR polling",
        "stage_started_at": "2026-05-06T10:00:00+08:00",
        "last_heartbeat_at": "2026-05-06T10:00:30+08:00",
        "external_ids": {"dashscope_task_id": "task-1"},
        "polish": {
            "status": "proposal_ready",
            "updated_at": "2026-05-06T10:01:00+08:00",
            "model": "qwen-test",
            "proposed_changes": 2,
            "proposal_json": "tmp/corrections/proposal_test.json",
            "proposal_diff": "tmp/corrections/proposal_test.diff",
        },
    }
    save_manifest(project_dir, manifest)

    result = runner.invoke(app, ["project", "show", str(project_dir)])

    assert result.exit_code == 0
    assert manifest.project_id in result.output
    assert "Details" in result.output
    assert "2026-05-02T10:00:00+08:00" in result.output
    assert "Outputs" in result.output
    assert "Commands" in result.output
    assert "Memory Index" in result.output
    assert "关键结论" in result.output
    assert "Current stage" in result.output
    assert "ASR polling" in result.output
    assert "dashscope_task_id=task-1" in result.output
    assert "Transcript polish" in result.output
    assert (
        "proposal ready (2 change(s)); accept or inspect diff if needed"
        in result.output
    )
    assert f"meeting-asr project correct accept {manifest.project_id}" in result.output
    assert f"meeting-asr project correct diff {manifest.project_id}" in result.output
    assert "Final transcript" in result.output
    assert "Location" not in result.output
    assert "How to view" not in result.output
    assert (
        f"meeting-asr project transcript show {manifest.project_id} --kind auto"
        in result.output
    )
    assert (
        f"meeting-asr project transcript show {manifest.project_id} --kind srt"
        in result.output
    )
    assert f"meeting-asr project transcript show {manifest.project_id}" in result.output
    assert f"meeting-asr project transcript list {manifest.project_id}" in result.output


def test_project_list_plain_prints_stable_rows(tmp_path: Path) -> None:
    """Project list should offer stable plain output for scripts."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(
        tmp_path, projects_dir=projects_dir, title="Plain Demo"
    )
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app, ["project", "list", "--projects-dir", str(projects_dir), "--plain"]
    )

    assert result.exit_code == 0
    assert (
        result.output.splitlines()[0] == "project_id\tstate\tupdated\ttitle\tkeywords"
    )
    assert f"{manifest.project_id}\tCreated\t" in result.output
    assert "Plain Demo" in result.output
    assert "╭" not in result.output


def test_project_list_timestamp_uses_local_timezone() -> None:
    """Project list timestamps should convert aware manifest times to local display time."""
    china_timezone = timezone(timedelta(hours=8))

    rendered = _project_list_timestamp(
        "2026-05-03T00:51:31+00:00", timezone=china_timezone
    )

    assert rendered == "2026-05-03 08:51"


def test_project_show_accepts_project_id(tmp_path: Path) -> None:
    """Project show should resolve stable content-based project ids."""
    projects_dir = tmp_path / "projects"
    project_dir = _sample_project(tmp_path, projects_dir=projects_dir)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        ["project", "show", manifest.project_id, "--projects-dir", str(projects_dir)],
    )

    assert result.exit_code == 0
    assert "Demo" in result.output
    assert manifest.project_id in result.output


def test_project_show_explains_voiceprint_candidates(tmp_path: Path) -> None:
    """Project show should expose low-score voiceprint candidates instead of anonymous names."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    manifest = load_manifest(project_dir)
    manifest.speakers.update(
        {"detected_ids": [0, 1], "mapped": {"0": "欧丁", "1": "Speaker B"}}
    )
    save_manifest(project_dir, manifest)
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(
            {
                "threshold": 0.75,
                "matches": [
                    {
                        "speaker_id": 0,
                        "label": "Speaker A",
                        "name": "欧丁",
                        "score": 0.91,
                        "accepted": True,
                        "accepted_name": "欧丁",
                        "threshold": 0.75,
                    },
                    {
                        "speaker_id": 1,
                        "label": "Speaker B",
                        "name": None,
                        "score": 0.67,
                        "accepted": False,
                        "best_name": "敬悦",
                        "best_score": 0.67,
                        "accepted_name": None,
                        "threshold": 0.75,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["project", "show", str(project_dir)])

    assert result.exit_code == 0
    assert "Speakers" in result.output
    assert "2 detected; 欧丁" in result.output
    assert "2 detected; 欧丁, Speaker B" not in result.output
    assert "Voiceprint candidates (auto accept >= 0.750)" in result.output
    assert "Speaker B" in result.output
    assert "below-threshold" in result.output
    assert "best: 敬悦" in result.output
    assert "0.670" in result.output
    assert "Next" in result.output
    assert f"meeting-asr project review {manifest.project_id}" in result.output


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

    assert (
        resolve_project_source_path(project_dir, load_manifest(project_dir))
        == source.resolve()
    )


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

    mapping_path, transcript_path, srt_path = apply_project_speakers(
        project_dir, {0: "欧丁"}
    )

    assert mapping_path == project_dir / "speakers" / "speaker_map.json"
    assert transcript_path == project_dir / "exports" / "transcript_named.txt"
    assert srt_path == project_dir / "exports" / "subtitle_named.srt"
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")


def test_apply_project_speakers_refreshes_corrected_named_outputs(
    tmp_path: Path,
) -> None:
    """Speaker naming should keep corrected transcript outputs aligned with new names."""
    project_dir = _sample_project(tmp_path)
    sentences_path = project_dir / "asr" / "sentences.json"
    _write_sample_sentences(sentences_path)
    corrected = json.loads(sentences_path.read_text(encoding="utf-8"))
    corrected["sentences"][0]["text"] = "修正后的大家好。"
    corrected["full_text"] = "修正后的大家好。收到。再补一句。"
    (project_dir / "asr" / "sentences_corrected.json").write_text(
        json.dumps(corrected, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = load_manifest(project_dir)
    manifest.status = "corrected"
    save_manifest(project_dir, manifest)

    apply_project_speakers(project_dir, {0: "欧丁"})

    reloaded = load_manifest(project_dir)
    corrected_named = project_dir / "exports" / "transcript_named_corrected.txt"
    corrected_srt = project_dir / "exports" / "subtitle_named_corrected.srt"

    assert reloaded.status == "corrected"
    assert (
        reloaded.outputs["corrected_named_transcript"]
        == "exports/transcript_named_corrected.txt"
    )
    assert (
        reloaded.outputs["corrected_named_subtitle"]
        == "exports/subtitle_named_corrected.srt"
    )
    assert "修正后的大家好" in corrected_named.read_text(encoding="utf-8")
    assert "欧丁" in corrected_srt.read_text(encoding="utf-8")


def test_apply_project_speakers_keeps_person_map_in_sync(tmp_path: Path) -> None:
    """Applying names without person ids should not leave stale identity links."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    apply_project_speakers(project_dir, {0: "欧丁"}, person_mapping={0: 7})
    assert json.loads(
        (project_dir / "speakers" / "speaker_person_map.json").read_text(
            encoding="utf-8"
        )
    ) == {"0": 7}

    apply_project_speakers(
        project_dir,
        {0: "欧丁"},
        person_mapping={0: 7},
        person_public_mapping={0: "vpp-0000000000000007"},
    )
    assert json.loads(
        (project_dir / "speakers" / "speaker_person_map.json").read_text(
            encoding="utf-8"
        )
    ) == {"0": "vpp-0000000000000007"}

    apply_project_speakers(project_dir, {0: "新名字"})
    manifest = load_manifest(project_dir)

    assert not (project_dir / "speakers" / "speaker_person_map.json").exists()
    assert "person_map" not in manifest.speakers


def test_apply_project_speakers_merges_existing_names_by_default(
    tmp_path: Path,
) -> None:
    """Applying one confirmed name should not erase existing speaker names."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    apply_project_speakers(
        project_dir,
        {0: "欧丁", 1: "敬悦"},
        person_public_mapping={0: "vpp-0000000000000000", 1: "vpp-0000000000000001"},
    )
    apply_project_speakers(project_dir, {1: "米汤"})

    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    person_mapping = json.loads(
        (project_dir / "speakers" / "speaker_person_map.json").read_text(
            encoding="utf-8"
        )
    )

    assert mapping == {"0": "欧丁", "1": "米汤"}
    assert person_mapping == {"0": "vpp-0000000000000000"}


def test_apply_project_speakers_can_replace_existing_names(tmp_path: Path) -> None:
    """Callers that need destructive replacement must opt in explicitly."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    apply_project_speakers(project_dir, {0: "欧丁", 1: "敬悦"})
    apply_project_speakers(project_dir, {1: "米汤"}, replace_existing=True)

    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )

    assert mapping == {"1": "米汤"}


def test_apply_project_speakers_ignores_low_information_speaker(tmp_path: Path) -> None:
    """Existing normalized transcripts should still drop backchannel-only speakers."""
    project_dir = _sample_project(tmp_path)
    payload = {
        "full_text": "大家看供应商闭环。嗯对啊。",
        "detected_speakers": [0, 2],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "大家看供应商闭环。",
                "speaker_id": 0,
            },
            {
                "begin_time_ms": 1100,
                "end_time_ms": 1200,
                "text": "嗯。",
                "speaker_id": 2,
            },
            {
                "begin_time_ms": 1300,
                "end_time_ms": 1400,
                "text": "对。",
                "speaker_id": 2,
            },
            {
                "begin_time_ms": 1500,
                "end_time_ms": 1600,
                "text": "啊。",
                "speaker_id": 2,
            },
            {
                "begin_time_ms": 1700,
                "end_time_ms": 1800,
                "text": "这样吧。",
                "speaker_id": 2,
            },
            {
                "begin_time_ms": 1900,
                "end_time_ms": 2000,
                "text": "就是听听听听他。",
                "speaker_id": 2,
            },
        ],
    }
    sentences_path = project_dir / "asr" / "sentences.json"
    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    sentences_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    _, transcript_path, _ = apply_project_speakers(
        project_dir, {0: "欧丁", 2: "Speaker C"}
    )
    manifest = load_manifest(project_dir)
    transcript = transcript_path.read_text(encoding="utf-8")

    assert manifest.speakers["detected_ids"] == [0]
    assert manifest.speakers["mapped"] == {"0": "欧丁"}
    assert "Speaker C" not in transcript
    assert "大家看供应商闭环" in transcript


def test_apply_project_speakers_persists_explicit_ignored_speakers(
    tmp_path: Path,
) -> None:
    """Explicit ignore state should not be encoded as a fake speaker name."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    apply_project_speakers(
        project_dir, {0: "欧丁", 1: "Speaker B"}, ignored_speaker_ids=(1,)
    )

    manifest = load_manifest(project_dir)
    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    ignored = json.loads(
        (project_dir / "speakers" / "speaker_ignore.json").read_text(encoding="utf-8")
    )

    assert mapping == {"0": "欧丁"}
    assert ignored == {"ignored_speakers": [1]}
    assert manifest.speakers["mapped"] == {"0": "欧丁"}
    assert manifest.speakers["ignored"] == [1]


def test_project_speakers_inspect_shows_mapped_names(tmp_path: Path) -> None:
    """Speaker inspect should show human names after speaker apply."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {0: "欧丁", 1: "敬悦"})

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

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

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

    assert result.exit_code == 0
    assert "Speaker B (speaker_id=1)" in result.output
    assert (
        "Voiceprint match: name=敬悦 score=0.775 threshold=0.750 status=matched"
        in result.output
    )


def test_project_speakers_inspect_shows_below_threshold_candidates(
    tmp_path: Path,
) -> None:
    """Speaker inspect should explain low-score voiceprint candidates."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    manifest = load_manifest(project_dir)
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(
            {
                "provider": "local-speechbrain",
                "model": "speechbrain-spkrec-ecapa-voxceleb",
                "threshold": 0.75,
                "matches": [
                    {
                        "speaker_id": 0,
                        "label": "Speaker A",
                        "name": None,
                        "score": 0.6704,
                        "accepted": False,
                        "best_name": "墨泪",
                        "best_score": 0.6704,
                        "accepted_name": None,
                        "threshold": 0.75,
                        "status": "below-threshold",
                        "sample_count": 23,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

    assert result.exit_code == 0
    assert (
        "Voiceprint match: best=墨泪 score=0.670 threshold=0.750 status=below-threshold"
        in result.output
    )
    assert "unknown" not in result.output
    assert (
        f"Recommended next step: meeting-asr project speakers review {manifest.project_id}"
        in result.output
    )
    assert (
        f"meeting-asr project speakers inspect {manifest.project_id} --sample-count 5"
        in result.output
    )
    assert (
        f"meeting-asr project speakers apply {manifest.project_id} --map 0=Name"
        in result.output
    )
    assert f"meeting-asr voiceprint review {manifest.project_id}" in result.output


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

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

    assert result.exit_code == 0
    assert "Name: 敬悦" in result.output
    assert (
        "Voiceprint match: name=墨泪 score=0.801 threshold=0.750 status=matched CONFLICT"
        in result.output
    )


def test_speaker_match_summary_colors_review_states() -> None:
    """Voiceprint match summaries should use color to separate review states."""
    accepted = project_commands._speaker_match_summary(
        {"name": "敬悦", "score": 0.775052, "accepted": True}
    )
    review = project_commands._speaker_match_summary(
        {
            "name": None,
            "best_name": "墨泪",
            "best_score": 0.6704,
            "accepted": False,
            "threshold": 0.75,
        }
    )
    conflict = project_commands._speaker_match_summary(
        {"label": "Speaker B", "name": "墨泪", "score": 0.80123, "accepted": True},
        mapped_name="敬悦",
    )

    assert "\x1b[32m" in accepted
    assert "\x1b[33m" in review
    assert "best=墨泪" in review
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
    assert (
        "Speaker B speaker_id=1 status=conflict name=敬悦 match=matched:墨泪"
        in result.output
    )


def test_project_review_summary_shows_below_threshold_best_candidate(
    tmp_path: Path,
) -> None:
    """Project review summary should not collapse low-score candidates to unknown."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    manifest = load_manifest(project_dir)
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(
            {
                "threshold": 0.75,
                "matches": [
                    {
                        "speaker_id": 0,
                        "label": "Speaker A",
                        "name": None,
                        "score": 0.6704,
                        "accepted": False,
                        "best_name": "墨泪",
                        "best_score": 0.6704,
                        "accepted_name": None,
                        "threshold": 0.75,
                        "status": "below-threshold",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "Speaker A"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "project",
            "review",
            str(project_dir),
            "--summary",
            "--store-dir",
            str(tmp_path / "voiceprints"),
        ],
    )

    assert result.exit_code == 0
    assert (
        f"Recommended next step: meeting-asr project review {manifest.project_id}"
        in result.output
    )
    assert (
        "This opens the human review workflow for unresolved speakers." in result.output
    )
    assert (
        "Speaker A: below-threshold best=墨泪 score=0.670 threshold=0.750"
        in result.output
    )
    assert "Speaker A speaker_id=0 status=review name=Speaker A" in result.output
    assert "match=below-threshold:墨泪 score=0.670 threshold=0.750" in result.output
    assert "status=ignored" not in result.output
    assert "unknown" not in result.output
    assert (
        f"meeting-asr project speakers apply {manifest.project_id} --map 0=Name"
        in result.output
    )
    assert result.output.index(
        f"meeting-asr project review {manifest.project_id}"
    ) < result.output.index(
        f"meeting-asr project speakers apply {manifest.project_id} --map 0=Name"
    )
    assert f"meeting-asr voiceprint review {manifest.project_id}" in result.output


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

    result = runner.invoke(
        app, ["project", "review", "--summary", "--projects-dir", str(projects_dir)]
    )

    assert result.exit_code == 0
    assert f"Projects: {projects_dir.resolve()}" in result.output
    assert manifest.project_id in result.output


def test_speaker_review_help_separates_human_and_scripted_paths() -> None:
    """Speaker command help should point humans to review and scripts to apply --map."""
    project_review = runner.invoke(app, ["--lang", "en", "project", "review", "--help"])
    speaker_review = runner.invoke(
        app, ["--lang", "en", "project", "speakers", "review", "--help"]
    )
    apply_help = runner.invoke(
        app, ["--lang", "en", "project", "speakers", "apply", "--help"]
    )
    inspect_help = runner.invoke(
        app, ["--lang", "en", "project", "speakers", "inspect", "--help"]
    )

    assert project_review.exit_code == 0
    assert "recommended human review workflow" in project_review.output
    assert "unresolved speaker names" in project_review.output
    assert speaker_review.exit_code == 0
    assert "recommended interactive speaker identity review" in speaker_review.output
    assert apply_help.exit_code == 0
    assert "non-interactively" in apply_help.output
    assert "scripts or" in apply_help.output
    assert "already confirmed mappings" in apply_help.output
    assert inspect_help.exit_code == 0
    assert "diagnostic speaker samples" in inspect_help.output
    assert "read-only, does not apply names" in inspect_help.output


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
    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert "Name for Speaker A" in result.output
    assert "Name for Speaker B" in result.output
    assert mapping == {"0": "欧丁", "1": "敬悦"}
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")
    assert "meeting-asr project speakers preview" in result.output
    assert "meeting-asr voiceprint review PROJECT_ID" in result.output
    assert f"open {transcript_path.resolve()}" in result.output


def test_project_speakers_apply_map_writes_named_transcript(tmp_path: Path) -> None:
    """Scripted speaker mappings should still write named project outputs."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "apply",
            str(project_dir),
            "--map",
            "0=欧丁",
            "--map",
            "1=敬悦",
        ],
    )

    transcript_path = project_dir / "exports" / "transcript_named.txt"
    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert mapping == {"0": "欧丁", "1": "敬悦"}
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")
    assert "敬悦" in transcript_path.read_text(encoding="utf-8")


def test_project_speakers_apply_map_merges_saved_names(tmp_path: Path) -> None:
    """Scripted speaker apply should patch names instead of clearing the project map."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {0: "欧丁"})

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--map", "1=敬悦"],
    )

    transcript_path = project_dir / "exports" / "transcript_named.txt"
    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert mapping == {"0": "欧丁", "1": "敬悦"}
    assert "欧丁" in transcript_path.read_text(encoding="utf-8")
    assert "敬悦" in transcript_path.read_text(encoding="utf-8")


def test_project_speakers_apply_map_replace_clears_saved_names(tmp_path: Path) -> None:
    """Destructive scripted speaker apply should require the explicit replace flag."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    apply_project_speakers(project_dir, {0: "欧丁"})

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "apply",
            str(project_dir),
            "--map",
            "1=敬悦",
            "--replace",
        ],
    )

    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert mapping == {"1": "敬悦"}


def test_project_speakers_apply_can_show_more_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Speaker apply should let users ask for more evidence before naming."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    remembered: list[str] = []
    monkeypatch.setattr(
        "app.commands.project._remember_prompt_history", remembered.append
    )

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--sample-count", "1"],
        input="/more\n欧丁\n敬悦\n",
    )

    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
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
    audio_clip = (
        project_dir / "tmp" / "speaker_apply_preview" / "Speaker_A" / "preview.wav"
    )

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

    monkeypatch.setattr(
        "app.commands.project.build_audio_preview_command",
        fake_build_audio_preview_command,
    )
    monkeypatch.setattr(
        "app.commands.project._build_speaker_apply_audio_preview_clip",
        fake_build_audio_preview_clip,
    )
    monkeypatch.setattr(
        "app.commands.project._run_speaker_apply_preview_command",
        fake_run_preview_command,
    )
    monkeypatch.setattr(
        "app.commands.project._remember_prompt_history", remembered.append
    )

    result = runner.invoke(
        app,
        ["project", "speakers", "apply", str(project_dir), "--sample-count", "1"],
        input="/more\n/audio\n欧丁\n敬悦\n",
    )

    mapping = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert played == [["audio-player"]]
    assert remembered == ["/more", "/audio"]
    assert captured["audio"] == audio_clip
    assert captured["audio_start"] == 0.0
    assert captured["audio_duration"] == 0.0
    assert audio_segments == ["再补一句。"]
    assert (
        "Preview sample for Speaker A: [00:00:01.900 - 00:00:02.500] 再补一句。"
        in result.output
    )
    assert (
        "Starting audio preview for Speaker A with 1 displayed sample(s)."
        in result.output
    )
    assert (
        "Controls: Space/P pauses, Q/Esc stops early, Ctrl-C also stops."
        in result.output
    )
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

    assert (
        output
        == tmp_path / "tmp" / "speaker_apply_preview" / "Speaker_A" / "preview.wav"
    )
    assert [(start, duration) for _, start, duration in calls] == [
        (0.0, 4.0),
        (4.0, 6.0),
    ]
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
    monkeypatch.setattr(
        project_commands.subprocess, "Popen", lambda command, stdin=None: process
    )
    monkeypatch.setattr(project_commands.termios, "tcgetattr", lambda fd: ["old"])
    monkeypatch.setattr(
        project_commands.termios, "tcsetattr", lambda fd, when, settings: None
    )
    monkeypatch.setattr(project_commands.tty, "setcbreak", lambda fd: None)
    monkeypatch.setattr(
        project_commands.select,
        "select",
        lambda read, write, error, timeout: (read, [], []),
    )

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
    (project_dir / "exports" / "subtitle_named.srt").write_text(
        "named", encoding="utf-8"
    )
    captured: dict[str, Path] = {}

    def fake_build_preview_command(
        *, video: Path, subtitle: Path, start_seconds: float
    ) -> list[str]:
        captured["video"] = video
        captured["subtitle"] = subtitle
        return ["player", str(subtitle)]

    monkeypatch.setattr(
        "app.commands.project.build_preview_command", fake_build_preview_command
    )

    result = runner.invoke(
        app, ["project", "speakers", "preview", str(project_dir), "--dry-run"]
    )

    assert result.exit_code == 0
    assert captured["video"] == project_dir.resolve() / "source" / "meeting.mp4"
    assert (
        captured["subtitle"] == project_dir.resolve() / "exports" / "subtitle_named.srt"
    )
    assert "subtitle_named.srt" in result.output


def test_project_speakers_preview_prefers_corrected_named_subtitle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preview after vocabulary correction should use corrected named subtitles."""
    project_dir = _sample_project(tmp_path)
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "exports").mkdir(exist_ok=True)
    (project_dir / "exports" / "subtitle_named.srt").write_text(
        "named", encoding="utf-8"
    )
    (project_dir / "exports" / "subtitle_named_corrected.srt").write_text(
        "corrected", encoding="utf-8"
    )
    captured: dict[str, Path] = {}

    def fake_build_preview_command(
        *, video: Path, subtitle: Path, start_seconds: float
    ) -> list[str]:
        captured["subtitle"] = subtitle
        return ["player", str(subtitle)]

    monkeypatch.setattr(
        "app.commands.project.build_preview_command", fake_build_preview_command
    )

    result = runner.invoke(
        app, ["project", "speakers", "preview", str(project_dir), "--dry-run"]
    )

    assert result.exit_code == 0
    assert (
        captured["subtitle"]
        == project_dir.resolve() / "exports" / "subtitle_named_corrected.srt"
    )
    assert "subtitle_named_corrected.srt" in result.output


def test_project_speakers_preview_uses_asr_audio_for_audio_only_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Audio-only subtitle preview should use the ASR audio timeline."""
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"source")
    project_dir = tmp_path / "projects" / "audio-preview"
    create_project(
        source,
        title="Audio Preview",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    _write_sample_sentences(project_dir / "asr" / "sentences.json")
    (project_dir / "exports").mkdir(exist_ok=True)
    (project_dir / "exports" / "subtitle.srt").write_text("anonymous", encoding="utf-8")
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {
        "path": "audio/audio.flac",
        "format": "flac",
        "duration_seconds": 10.0,
    }
    save_manifest(project_dir, manifest)
    captured: dict[str, Path] = {}

    def fake_build_preview_command(
        *, video: Path, subtitle: Path, start_seconds: float
    ) -> list[str]:
        captured["video"] = video
        return ["player", str(video)]

    monkeypatch.setattr(
        "app.commands.project.build_preview_command", fake_build_preview_command
    )

    result = runner.invoke(
        app, ["project", "speakers", "preview", str(project_dir), "--dry-run"]
    )

    assert result.exit_code == 0
    assert captured["video"] == audio_path.resolve()
    assert "audio.flac" in result.output


def test_project_git_init_writes_safe_ignore_file(tmp_path: Path) -> None:
    """Optional Git tracking should ignore heavy generated artifacts."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    project_dir = _sample_project(tmp_path)

    gitignore_path = init_project_git(project_dir)

    content = gitignore_path.read_text(encoding="utf-8")
    assert "source/" in content
    assert "audio/" in content


def _sample_project(
    tmp_path: Path,
    *,
    projects_dir: Path | None = None,
    title: str = "Demo",
    meeting_time: str | None = None,
) -> Path:
    """Create a minimal project for tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = (projects_dir or tmp_path) / "project"
    create_project(
        source,
        title=title,
        projects_dir=projects_dir or tmp_path,
        project_dir=project_dir,
        meeting_time=meeting_time,
        hash_source=False,
    )
    return project_dir


def _write_sample_sentences(path: Path) -> None:
    """Write a normalized sentences.json fixture."""
    _write_sentences(
        path,
        [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "大家好。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 1200,
                "end_time_ms": 1800,
                "text": "收到。",
                "speaker_id": 1,
                "sentence_id": 2,
            },
            {
                "begin_time_ms": 1900,
                "end_time_ms": 2500,
                "text": "再补一句。",
                "speaker_id": 0,
                "sentence_id": 3,
            },
        ],
    )


def _write_sentences(path: Path, sentences: list[dict[str, object]]) -> None:
    """Write a normalized sentences.json fixture with custom text."""
    payload = {
        "full_text": "".join(str(sentence["text"]) for sentence in sentences),
        "detected_speakers": sorted(
            {
                int(sentence["speaker_id"])
                for sentence in sentences
                if sentence.get("speaker_id") is not None
            }
        ),
        "sentences": sentences,
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


def _transcribe_options() -> ProjectTranscribeOptions:
    """Build minimal transcription options for workflow tests."""
    return ProjectTranscribeOptions(
        speaker_count=None,
        language=None,
        model="fun-asr",
        oss_upload=True,
        file_url=None,
        generate_srt=True,
        timestamp_alignment=True,
        disfluency_removal=False,
        audio_format="wav",
    )


def _settings() -> Settings:
    """Build runtime settings for isolated project workflow tests."""
    return Settings(
        dashscope_api_key="dashscope-key",
        dashscope_base_url="https://dashscope.example.com",
        oss_access_key_id="oss-id",
        oss_access_key_secret="oss-secret",
        oss_bucket_name="meeting-bucket",
        oss_region="cn-test",
        oss_endpoint="https://oss.example.com",
    )


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


def test_parse_mapping_items_separates_names_and_public_ids() -> None:
    """@vpp-id becomes a person public-id binding; plain values stay display names."""
    names, publics = project_manager.parse_mapping_items(
        ["0=徐铤(彬川)", "2=@vpp-f61409c960abfe86"], {0, 1, 2}
    )
    assert names == {0: "徐铤(彬川)"}
    assert publics == {2: "vpp-f61409c960abfe86"}


def test_parse_mapping_items_rejects_invalid_public_id() -> None:
    """A malformed @vpp value is rejected instead of silently becoming a name."""
    with pytest.raises(typer.BadParameter):
        project_manager.parse_mapping_items(["0=@vpp-not-hex"], {0})
