"""Tests for user-facing project workflow inference."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.project_workflow import project_workflow_summary
from app.project_manager import apply_project_speakers, create_project, load_manifest, save_manifest


def test_created_project_points_to_transcription(tmp_path: Path) -> None:
    """A new project should show its current stage and detail next action."""
    project_dir = _sample_project(tmp_path)
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref=manifest.project_id)

    assert summary.state == "Created"
    assert summary.next_command_short == f"transcribe {manifest.project_id}"
    assert summary.outputs == ()


def test_prepared_project_stays_before_transcription(tmp_path: Path) -> None:
    """A project with audio but no ASR result should be in the prepared stage."""
    project_dir = _sample_project(tmp_path)
    (project_dir / "audio" / "audio.flac").write_bytes(b"fake audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {"path": "audio/audio.flac", "format": "flac"}
    save_manifest(project_dir, manifest)
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref="2")

    assert summary.state == "Prepared"
    assert summary.next_command_short == "transcribe 2"
    assert summary.outputs == ("audio",)


def test_transcribed_project_points_to_speaker_review(tmp_path: Path) -> None:
    """A transcribed project without named outputs should remain transcribed."""
    project_dir = _sample_project(tmp_path)
    _write_transcribed_outputs(project_dir)
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref="3")

    assert summary.state == "Transcribed"
    assert summary.next_command_short == "review 3"
    assert summary.outputs == ("plain-txt", "speaker-txt", "srt")


def test_named_project_is_completed(tmp_path: Path) -> None:
    """Named outputs should make the main project workflow complete."""
    project_dir = _sample_project(tmp_path)
    _write_transcribed_outputs(project_dir)
    apply_project_speakers(project_dir, {0: "敬悦"})
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref="4")

    assert summary.state == "Completed"
    assert summary.next_command_short == "correct edit 4"
    assert summary.outputs == ("named-txt", "named-srt")


def test_corrected_project_points_at_corrected_transcript(tmp_path: Path) -> None:
    """Corrected artifacts should be treated as the completed project state."""
    project_dir = _sample_project(tmp_path)
    _write_transcribed_outputs(project_dir)
    _write_corrected_outputs(project_dir)
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref="5")

    assert summary.state == "Corrected"
    assert summary.next_command_short == "transcript show 5 --kind corrected"
    assert summary.outputs == ("corrected-txt", "corrected-srt")


def test_missing_declared_output_marks_project_broken(tmp_path: Path) -> None:
    """Manifest output pointers should be verified against the filesystem."""
    project_dir = _sample_project(tmp_path)
    manifest = load_manifest(project_dir)
    manifest.outputs["named_transcript"] = "exports/missing.txt"
    save_manifest(project_dir, manifest)
    manifest = load_manifest(project_dir)

    summary = project_workflow_summary(project_dir, manifest, project_ref="5")

    assert summary.state == "Broken"
    assert summary.next_command_short == "status 5"
    assert summary.missing == ("named_transcript:exports/missing.txt",)


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for workflow inference tests."""
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
    return project_dir


def _write_transcribed_outputs(project_dir: Path) -> None:
    """Write the minimal ASR artifacts that transcription normally produces."""
    payload = {
        "full_text": "大家好。",
        "detected_speakers": [0],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "大家好。",
                "speaker_id": 0,
                "sentence_id": 1,
            }
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (project_dir / "exports" / "transcript.txt").write_text("大家好。\n", encoding="utf-8")
    (project_dir / "exports" / "transcript_speakers.txt").write_text("Speaker A: 大家好。\n", encoding="utf-8")
    (project_dir / "exports" / "subtitle.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n大家好。\n",
        encoding="utf-8",
    )
    manifest = load_manifest(project_dir)
    manifest.status = "transcribed"
    manifest.outputs.update(
        {
            "sentences": "asr/sentences.json",
            "plain_transcript": "exports/transcript.txt",
            "anonymous_transcript": "exports/transcript_speakers.txt",
            "subtitle": "exports/subtitle.srt",
        }
    )
    save_manifest(project_dir, manifest)


def _write_corrected_outputs(project_dir: Path) -> None:
    """Write corrected artifacts and record their manifest pointers."""
    transcript = project_dir / "exports" / "transcript_named_corrected.txt"
    subtitle = project_dir / "exports" / "subtitle_named_corrected.srt"
    transcript.write_text("敬悦: 大家好。\n", encoding="utf-8")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n敬悦: 大家好。\n", encoding="utf-8")
    manifest = load_manifest(project_dir)
    manifest.status = "corrected"
    manifest.outputs["corrected_named_transcript"] = "exports/transcript_named_corrected.txt"
    manifest.outputs["corrected_named_subtitle"] = "exports/subtitle_named_corrected.srt"
    save_manifest(project_dir, manifest)
