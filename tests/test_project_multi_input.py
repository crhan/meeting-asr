"""Tests for multi-input ``project run`` (concatenated single project)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app import project_manager
from app.cli import app
from app.infra import ffmpeg

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep these tests away from the developer's real XDG/project state."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def _media(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


# --- identity ---------------------------------------------------------------


def test_combined_identity_single_is_plain_file_hash() -> None:
    """A single input keeps its own SHA-256, preserving today's project id."""
    sha = hashlib.sha256(b"abc").hexdigest()
    assert project_manager._combined_identity_sha([sha]) == sha


def test_combined_identity_multi_is_deterministic_and_order_sensitive() -> None:
    """Multiple inputs hash the ordered per-file digests; order changes identity."""
    a = hashlib.sha256(b"a").hexdigest()
    b = hashlib.sha256(b"b").hexdigest()
    first = project_manager._combined_identity_sha([a, b])
    assert first == project_manager._combined_identity_sha([a, b])
    assert first != project_manager._combined_identity_sha([b, a])
    assert first != a and first != b


# --- create / reuse / collision --------------------------------------------


def test_multi_source_create_records_segments(tmp_path: Path) -> None:
    """Multi-input create stages every segment and records ordered provenance."""
    part_a = _media(tmp_path / "partA.mp4", b"AAAA")
    part_b = _media(tmp_path / "partB.mp4", b"BBBBBB")
    projects = tmp_path / "projects"

    summary = project_manager.create_or_reuse_project(
        part_a,
        title=None,
        projects_dir=projects,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
        extra_inputs=[part_b],
    )

    assert summary.created is True
    manifest = summary.manifest
    segments = manifest.audio["segments"]
    assert [seg["filename"] for seg in segments] == ["partA.mp4", "partB.mp4"]
    assert [seg["index"] for seg in segments] == [0, 1]
    assert segments[0]["original_path"] == str(part_a.resolve())
    # Nulled on purpose so single/multi runs never collide on reuse.
    assert manifest.source.original_path is None
    # Identity is the combined content hash of the ordered segments.
    assert manifest.source.sha256 is not None
    assert manifest.project_id == "p-" + manifest.source.sha256[:16]
    assert (summary.project_dir / "source" / "partA.mp4").exists()
    assert (summary.project_dir / "source" / "partB.mp4").exists()


def test_multi_source_reuses_same_ordered_inputs(tmp_path: Path) -> None:
    """Re-running the same ordered inputs reuses the project, not a duplicate."""
    part_a = _media(tmp_path / "a.mp4", b"AAAA")
    part_b = _media(tmp_path / "b.mp4", b"BBBB")
    projects = tmp_path / "projects"
    kwargs = dict(
        title=None,
        projects_dir=projects,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
        extra_inputs=[part_b],
    )

    first = project_manager.create_or_reuse_project(part_a, **kwargs)
    second = project_manager.create_or_reuse_project(part_a, **kwargs)

    assert first.created is True
    assert second.created is False
    assert first.project_dir == second.project_dir


def test_multi_source_identity_distinct_from_single_run(tmp_path: Path) -> None:
    """A single-file run of one segment must not reuse the concatenated project."""
    part_a = _media(tmp_path / "a.mp4", b"AAAA")
    part_b = _media(tmp_path / "b.mp4", b"BBBB")
    projects = tmp_path / "projects"

    multi = project_manager.create_or_reuse_project(
        part_a,
        title=None,
        projects_dir=projects,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
        extra_inputs=[part_b],
    )
    single = project_manager.create_or_reuse_project(
        part_a,
        title=None,
        projects_dir=projects,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
    )

    assert single.created is True
    assert single.project_dir != multi.project_dir


# --- concat helper ----------------------------------------------------------


def test_concat_audio_for_asr_returns_segment_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concat normalizes each input, probes per-part durations, and joins once."""
    part_a = _media(tmp_path / "a.mp4", b"a")
    part_b = _media(tmp_path / "b.mp4", b"b")
    out = tmp_path / "audio" / "audio.flac"

    def fake_extract(src: Path, dst: Path, *, audio_format: str) -> Path:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"part")
        return Path(dst)

    durations = iter([12.0, 7.5])
    ran: dict[str, list[str]] = {}

    def fake_run(command: list[str]) -> None:
        ran["command"] = command
        Path(command[-1]).write_bytes(b"concat")

    monkeypatch.setattr(ffmpeg, "extract_audio_for_asr", fake_extract)
    monkeypatch.setattr(ffmpeg, "probe_media_duration_seconds", lambda p: next(durations))
    monkeypatch.setattr(ffmpeg, "_run_ffmpeg", fake_run)

    result = ffmpeg.concat_audio_for_asr([part_a, part_b], out, audio_format="flac")

    assert result == [12.0, 7.5]
    assert out.exists()
    assert "concat" in ran["command"]


# --- multi audio preparation ------------------------------------------------


def test_prepare_project_audio_multi_builds_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concatenation enriches segments with offsets and prunes staged copies."""
    part_a = _media(tmp_path / "a.mp4", b"AAAA")
    part_b = _media(tmp_path / "b.mp4", b"BBBB")
    projects = tmp_path / "projects"
    summary = project_manager.create_or_reuse_project(
        part_a,
        title=None,
        projects_dir=projects,
        project_dir=None,
        meeting_time=None,
        hash_source=False,
        extra_inputs=[part_b],
    )
    project_dir = summary.project_dir

    def fake_concat(input_paths, output_path, *, audio_format):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-audio")
        return [10.0, 20.0]

    monkeypatch.setattr(project_manager, "concat_audio_for_asr", fake_concat)
    monkeypatch.setattr(project_manager, "probe_media_duration_seconds", lambda p: 30.0)

    audio_path = project_manager.prepare_project_audio_multi(
        project_dir, audio_format="flac"
    )

    assert audio_path.exists()
    manifest = project_manager.load_manifest(project_dir)
    assert manifest.audio["duration_seconds"] == 30.0
    segments = manifest.audio["segments"]
    assert segments[0]["offset_seconds"] == 0.0
    assert segments[0]["duration_seconds"] == 10.0
    assert segments[1]["offset_seconds"] == 10.0
    assert segments[1]["duration_seconds"] == 20.0
    # Staged video copies are pruned once the concatenated audio exists.
    assert not (project_dir / "source" / "a.mp4").exists()
    assert not (project_dir / "source" / "b.mp4").exists()


# --- CLI guardrails ---------------------------------------------------------


def test_run_rejects_file_url_with_multiple_inputs(tmp_path: Path) -> None:
    """--file-url is incompatible with multiple inputs (segments concat locally)."""
    part_a = _media(tmp_path / "a.wav", b"a")
    part_b = _media(tmp_path / "b.wav", b"b")

    result = runner.invoke(
        app,
        [
            "project",
            "run",
            str(part_a),
            str(part_b),
            "--file-url",
            "http://example.com/audio.flac",
        ],
    )

    assert result.exit_code != 0
    assert "file-url" in result.output.lower()
