"""Tests for voiceprint sample audio normalization."""

from __future__ import annotations

from pathlib import Path
import os

import app.voiceprint_audio as voiceprint_audio
from app.voiceprint_audio import (
    VOICEPRINT_AUDIO_PREPROCESS_VERSION,
    normalize_voiceprint_samples,
    normalized_voiceprint_sample_path,
)
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    store_voiceprint_samples,
)


def test_normalized_voiceprint_sample_path_is_versioned(tmp_path: Path) -> None:
    """Normalized clips should be derived from clip_rel_path under a versioned directory."""
    store_dir = _store(tmp_path)
    sample = list_all_voiceprint_samples(get_voiceprint_db_path(store_dir))[0]

    path = normalized_voiceprint_sample_path(sample, store_dir=store_dir)

    assert path == store_dir / "normalized" / VOICEPRINT_AUDIO_PREPROCESS_VERSION / sample.clip_rel_path


def test_normalize_voiceprint_samples_keeps_original_and_skips_existing(monkeypatch, tmp_path: Path) -> None:
    """Normalization should write derived files without modifying original sample clips."""
    store_dir = _store(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def fake_run_ffmpeg(command: list[str]) -> None:
        source = Path(command[command.index("-i") + 1])
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"normalized")
        calls.append((source, output))

    monkeypatch.setattr(voiceprint_audio, "_run_ffmpeg", fake_run_ffmpeg)

    first = normalize_voiceprint_samples(store_dir=store_dir, rebuild=False)
    second = normalize_voiceprint_samples(store_dir=store_dir, rebuild=False)

    sample = list_all_voiceprint_samples(get_voiceprint_db_path(store_dir))[0]
    assert first.processed_count == 1
    assert second.skipped_count == 1
    assert sample.clip_path.read_bytes() == b"original"
    assert calls == [(sample.clip_path, normalized_voiceprint_sample_path(sample, store_dir=store_dir))]


def test_normalize_voiceprint_samples_rebuilds_stale_derived_clip(monkeypatch, tmp_path: Path) -> None:
    """A derived clip older than its source should be rebuilt."""
    store_dir = _store(tmp_path)
    sample = list_all_voiceprint_samples(get_voiceprint_db_path(store_dir))[0]
    normalized_path = normalized_voiceprint_sample_path(sample, store_dir=store_dir)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_bytes(b"stale")
    os.utime(normalized_path, (1, 1))
    calls: list[Path] = []

    def fake_run_ffmpeg(command: list[str]) -> None:
        output = Path(command[-1])
        output.write_bytes(b"fresh")
        calls.append(output)

    monkeypatch.setattr(voiceprint_audio, "_run_ffmpeg", fake_run_ffmpeg)

    summary = normalize_voiceprint_samples(store_dir=store_dir, rebuild=False)

    assert summary.processed_count == 1
    assert calls == [normalized_path]
    assert normalized_path.read_bytes() == b"fresh"


def _store(tmp_path: Path) -> Path:
    """Build a voiceprint store with one sample."""
    store_dir = tmp_path / "voiceprints"
    clip_path = store_dir / "clips" / "project-1" / "speaker_0" / "clip_001.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"original")
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    store_voiceprint_samples(
        [
            StoredVoiceprintSample(
                speaker_name="Alice",
                project_id="project-1",
                project_path=tmp_path / "project-1",
                project_speaker_id=0,
                source_path=source_path,
                clip_path=clip_path,
                clip_rel_path=str(clip_path.relative_to(store_dir)),
                source_begin_time_ms=0,
                source_end_time_ms=1000,
                clip_begin_time_ms=0,
                clip_end_time_ms=1000,
                transcript_text="hello",
            )
        ],
        get_voiceprint_db_path(store_dir),
    )
    return store_dir
