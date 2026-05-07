"""Tests for voiceprint capture review and selective persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import Static
from typer.testing import CliRunner

from app.cli import app
from app.presentation.tui import voiceprint_capture
from app.presentation.tui.voiceprint_capture import (
    FOCUSED_PANE_CLASS,
    UNFOCUSED_PANE_CLASS,
    VoiceprintCaptureReviewApp,
    load_voiceprint_capture_review_session,
    render_voiceprint_capture_review_summary,
)
from app.project_manager import create_project, load_manifest
from app.voiceprint_people import create_voiceprint_person
from app.voiceprint_store import get_voiceprint_db_path, list_all_voiceprint_samples
from app.voiceprints import (
    VoiceprintCaptureSummary,
    VoiceprintClip,
    VoiceprintSpeaker,
    persist_voiceprint_capture_selection,
    plan_voiceprint_capture,
)

runner = CliRunner()


def test_voiceprint_capture_review_session_renders_candidates(tmp_path: Path) -> None:
    """Capture review should expose planned samples before anything is written."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    summary = _capture_summary(tmp_path)

    session = load_voiceprint_capture_review_session(summary=summary, source_path=source_path, page_size=1)
    rendered = render_voiceprint_capture_review_summary(session)

    assert session.project_id == "project-1"
    assert len(session.speakers) == 1
    assert len(session.speakers[0].clips) == 2
    assert "Candidate samples: 2" in rendered
    assert "Alice speaker=0 score=- samples=2" in rendered


def test_voiceprint_capture_review_tui_toggles_selection_and_saves(tmp_path: Path) -> None:
    """The review TUI should return only checked sample paths."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    session = load_voiceprint_capture_review_session(
        summary=_capture_summary(tmp_path),
        source_path=source_path,
        page_size=2,
    )
    app = VoiceprintCaptureReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(100, 20)) as pilot:
            speakers = app.query_one("#speakers", Static)
            samples = app.query_one("#samples", Static)

            assert "Selected" in app._overview_pane()
            assert "selected 2/2" in app._speaker_pane()
            assert speakers.has_class(FOCUSED_PANE_CLASS)
            assert samples.has_class(UNFOCUSED_PANE_CLASS)

            await pilot.press("right")
            await pilot.press("x")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.saved is True
    assert app.return_value.selected_clip_rel_paths == frozenset({"clips/project-1/speaker_0/clip_002.wav"})


def test_voiceprint_capture_review_tui_space_toggles_playback(monkeypatch, tmp_path: Path) -> None:
    """Space should play and stop the selected planned sample from source media."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    session = load_voiceprint_capture_review_session(summary=_capture_summary(tmp_path), source_path=source_path)
    process = _RunningFakeProcess()
    starts = 0

    monkeypatch.setattr(
        voiceprint_capture,
        "build_audio_preview_command",
        lambda **kwargs: ["fake-player", str(kwargs["media"])],
    )

    def fake_popen(*args, **kwargs) -> _RunningFakeProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(voiceprint_capture.subprocess, "Popen", fake_popen)

    async def scenario() -> None:
        async with VoiceprintCaptureReviewApp(session).run_test() as pilot:
            await pilot.press("space")

            assert starts == 1
            assert pilot.app.playback_process is process

            await pilot.press("space")

            assert starts == 1
            assert process.terminated is True
            assert pilot.app.playback_process is None

    asyncio.run(scenario())


def test_persist_voiceprint_capture_selection_writes_only_accepted_samples(monkeypatch, tmp_path: Path) -> None:
    """Selective persistence should write only human-approved WAV clips and database rows."""
    project_dir = _sample_project(tmp_path)
    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=2,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
    )

    def fake_extract_audio_clip(source, output, start_seconds, duration_seconds) -> Path:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(f"{start_seconds}:{duration_seconds}".encode())
        return Path(output)

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)

    summary = persist_voiceprint_capture_selection(
        project_dir,
        planned=planned,
        selected_clip_rel_paths={planned.speakers[0].clips[1].rel_path},
    )
    samples = list_all_voiceprint_samples(get_voiceprint_db_path(tmp_path / "voiceprints"))
    manifest = load_manifest(project_dir)

    assert summary.sample_count == 1
    assert len(samples) == 1
    assert samples[0].transcript_text == "第二段更长一点"
    assert samples[0].clip_path.exists()
    assert manifest.status == "voiceprinted"
    assert manifest.speakers["voiceprints"]["sample_count"] == 1


def test_plan_voiceprint_capture_resolves_existing_person_by_name(tmp_path: Path) -> None:
    """Capture planning should show a stable VPP id when a named person already exists."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    person = create_voiceprint_person("Alice", get_voiceprint_db_path(store_dir))

    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=2,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=store_dir,
    )

    assert planned.speakers[0].person_id == person.speaker_id
    assert planned.speakers[0].person_public_id == person.public_id


def test_voiceprint_capture_command_accepts_project_id_for_dry_run(tmp_path: Path) -> None:
    """Capture should accept the same project id references shown in next-step guidance."""
    project_dir = _sample_project(tmp_path)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            manifest.project_id,
            "--projects-dir",
            str(tmp_path / "projects"),
            "--store-dir",
            str(tmp_path / "voiceprints"),
            "--dry-run",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert "Planned voiceprint samples: 2" in result.output
    assert "Alice (speaker 0): 2 sample(s)" in result.output


def test_voiceprint_review_command_summarizes_project_and_library(tmp_path: Path) -> None:
    """Unified review should accept project ids and show both review scopes."""
    project_dir = _sample_project(tmp_path)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "review",
            manifest.project_id,
            "--projects-dir",
            str(tmp_path / "projects"),
            "--store-dir",
            str(tmp_path / "voiceprints"),
            "--summary",
        ],
    )

    assert result.exit_code == 0
    assert "Voiceprint review" in result.output
    assert f"Project candidates: {manifest.title}" in result.output
    assert f"Project ID: {manifest.project_id}" in result.output
    assert "Status: created" in result.output
    assert "Source: meeting.mp4" in result.output
    assert "Candidate speakers: 1 | samples: 2" in result.output
    assert "Global library:" in result.output


class _RunningFakeProcess:
    """Fake process that remains alive until terminated."""

    def __init__(self) -> None:
        """Initialize process state."""
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        """Return None while the fake process is running."""
        return 0 if self.terminated or self.killed else None

    def terminate(self) -> None:
        """Mark the process as terminated."""
        self.terminated = True

    def wait(self, timeout: int | None = None) -> int:
        """Pretend playback exits after termination."""
        return 0

    def kill(self) -> None:
        """Mark the process as killed."""
        self.killed = True


def _capture_summary(tmp_path: Path) -> VoiceprintCaptureSummary:
    """Build a planned capture summary fixture."""
    store_dir = tmp_path / "voiceprints"
    clip_dir = store_dir / "clips"
    clips = [
        VoiceprintClip(
            path=clip_dir / "project-1" / "speaker_0" / "clip_001.wav",
            rel_path="clips/project-1/speaker_0/clip_001.wav",
            source_begin_time_ms=1000,
            source_end_time_ms=3000,
            clip_begin_time_ms=1000,
            clip_end_time_ms=3000,
            text="first alice sample",
        ),
        VoiceprintClip(
            path=clip_dir / "project-1" / "speaker_0" / "clip_002.wav",
            rel_path="clips/project-1/speaker_0/clip_002.wav",
            source_begin_time_ms=4000,
            source_end_time_ms=6500,
            clip_begin_time_ms=4000,
            clip_end_time_ms=6500,
            text="second alice sample",
        ),
    ]
    return VoiceprintCaptureSummary(
        store_dir=store_dir,
        db_path=get_voiceprint_db_path(store_dir),
        clip_dir=clip_dir,
        speakers=[VoiceprintSpeaker(0, "Alice", None, None, clips)],
        dry_run=True,
    )


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project with named speaker transcript samples."""
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
    payload = {
        "full_text": "第一段。第二段更长一点",
        "detected_speakers": [0],
        "sentences": [
            _sentence(1, "第一段。", 1000, 1500),
            _sentence(2, "第二段更长一点", 2000, 4000),
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text('{"0": "Alice"}\n', encoding="utf-8")
    return project_dir


def _sentence(sentence_id: int, text: str, begin_ms: int, end_ms: int) -> dict:
    """Build one transcript sentence payload."""
    return {
        "begin_time_ms": begin_ms,
        "end_time_ms": end_ms,
        "text": text,
        "speaker_id": 0,
        "sentence_id": sentence_id,
    }
