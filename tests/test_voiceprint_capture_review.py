"""Tests for voiceprint capture review and selective persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
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
from app.project_manager import create_project, load_manifest, save_manifest
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_people import create_voiceprint_person
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples_with_rows,
    upsert_voiceprint_embedding,
)
from app.voiceprints import (
    VoiceprintCaptureSummary,
    VoiceprintClip,
    VoiceprintSpeaker,
    capture_voiceprints,
    persist_voiceprint_capture_selection,
    plan_voiceprint_capture,
)

runner = CliRunner()


def test_voiceprint_capture_review_session_renders_candidates(tmp_path: Path) -> None:
    """Capture review should expose planned samples before anything is written."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    summary = _capture_summary(tmp_path)

    session = load_voiceprint_capture_review_session(
        summary=summary, source_path=source_path, page_size=1
    )
    rendered = render_voiceprint_capture_review_summary(session)

    assert session.project_id == "project-1"
    assert len(session.speakers) == 1
    assert len(session.speakers[0].clips) == 2
    assert "Candidate samples: 2" in rendered
    assert "Alice speaker=0 score=- samples=2" in rendered


def test_voiceprint_capture_review_tui_toggles_selection_and_saves(
    tmp_path: Path,
) -> None:
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
    assert app.return_value.selected_clip_rel_paths == frozenset(
        {"clips/project-1/speaker_0/clip_002.wav"}
    )


def test_voiceprint_capture_review_tui_space_toggles_playback(
    monkeypatch, tmp_path: Path
) -> None:
    """Space should play and stop the selected planned sample from source media."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    session = load_voiceprint_capture_review_session(
        summary=_capture_summary(tmp_path), source_path=source_path
    )
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


def test_persist_voiceprint_capture_selection_writes_only_accepted_samples(
    monkeypatch, tmp_path: Path
) -> None:
    """Selective persistence should write only human-approved WAV clips and database rows."""
    project_dir = _sample_project(tmp_path)
    audio_path = project_dir / "audio" / "audio.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"asr audio")
    manifest = load_manifest(project_dir)
    manifest.audio = {
        "path": "audio/audio.flac",
        "format": "flac",
        "duration_seconds": 4.0,
    }
    save_manifest(project_dir, manifest)
    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=2,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
    )
    sources: list[Path] = []

    def fake_extract_audio_clip(
        source, output, start_seconds, duration_seconds
    ) -> Path:
        sources.append(Path(source))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(f"{start_seconds}:{duration_seconds}".encode())
        return Path(output)

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)

    summary = persist_voiceprint_capture_selection(
        project_dir,
        planned=planned,
        selected_clip_rel_paths={planned.speakers[0].clips[1].rel_path},
    )
    samples = list_all_voiceprint_samples(
        get_voiceprint_db_path(tmp_path / "voiceprints")
    )
    manifest = load_manifest(project_dir)

    assert summary.sample_count == 1
    assert len(samples) == 1
    assert samples[0].transcript_text == "第二段更长一点"
    assert samples[0].clip_path.exists()
    assert sources == [audio_path.resolve()]
    assert manifest.status == "voiceprinted"
    assert manifest.speakers["voiceprints"]["sample_count"] == 1


def test_review_multi_speaker_failure_rolls_back_entire_selection(
    monkeypatch, tmp_path: Path
) -> None:
    """Strict review persistence must not retain an earlier speaker on later failure."""
    project_dir = _sample_project(tmp_path)
    sentences_path = project_dir / "asr" / "sentences.json"
    sentences = [
        _sentence(1, "Alice 第一段完整的声纹采集表达。", 1000, 5000),
        _sentence(2, "Bob 第一段完整的声纹采集表达。", 6000, 10_000),
        _sentence(3, "Alice 第二段完整的声纹采集表达。", 11_000, 15_000),
        _sentence(4, "Bob 第二段完整的声纹采集表达。", 16_000, 20_000),
    ]
    sentences[1]["speaker_id"] = 1
    sentences[3]["speaker_id"] = 1
    payload = {
        "full_text": "".join(str(sentence["text"]) for sentence in sentences),
        "detected_speakers": [0, 1],
        "sentences": sentences,
    }
    sentences_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text(
        '{"0": "Alice", "1": "Bob"}\n', encoding="utf-8"
    )
    store_dir = tmp_path / "voiceprints"
    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=1,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=store_dir,
    )
    assert [speaker.speaker_id for speaker in planned.speakers] == [0, 1]
    first_clip = planned.speakers[0].clips[0].path
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"previous valid sample")
    original_manifest = load_manifest(project_dir)

    def fail_second_speaker(source, output, start_seconds, duration_seconds) -> Path:
        if "speaker_1" in str(output):
            raise RuntimeError("speaker 1 slicing failed")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"new speaker 0 sample")
        return Path(output)

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fail_second_speaker)

    with pytest.raises(RuntimeError, match="speaker 1 slicing failed"):
        persist_voiceprint_capture_selection(
            project_dir,
            planned=planned,
            selected_clip_rel_paths={
                clip.rel_path for speaker in planned.speakers for clip in speaker.clips
            },
        )

    assert first_clip.read_bytes() == b"previous valid sample"
    assert not planned.speakers[1].clips[0].path.exists()
    assert list_all_voiceprint_samples(get_voiceprint_db_path(store_dir)) == []
    current_manifest = load_manifest(project_dir)
    assert current_manifest.status == original_manifest.status
    assert "voiceprints" not in current_manifest.speakers


def test_plan_voiceprint_capture_resolves_existing_person_by_name(
    tmp_path: Path,
) -> None:
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


def test_plan_voiceprint_capture_prefers_diverse_informative_segments(
    tmp_path: Path,
) -> None:
    """Capture planning should avoid filler and near-duplicate neighboring samples."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        [
            _sentence(1, "嗯嗯嗯嗯嗯嗯嗯嗯", 0, 10_000),
            _sentence(2, "这是第一段适合做声纹的完整表达。", 20_000, 30_000),
            _sentence(3, "这是离上一段太近的重复表达。", 21_000, 31_000),
            _sentence(4, "这是第二段相隔较远的有效声纹样本。", 50_000, 60_000),
        ],
    )

    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
    )

    texts = [clip.text for clip in planned.speakers[0].clips]
    assert texts == [
        "这是第一段适合做声纹的完整表达。",
        "这是第二段相隔较远的有效声纹样本。",
    ]
    assert all(clip.selection_score > 0 for clip in planned.speakers[0].clips)
    assert "boundary=" in planned.speakers[0].clips[0].selection_reason


def test_plan_voiceprint_capture_exposes_candidate_pool_with_recommendations(
    tmp_path: Path,
) -> None:
    """Planning should expose more candidates while marking only balanced defaults."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        [
            _sentence(
                index,
                f"这是第 {index} 段适合做声纹的完整表达。",
                index * 20_000,
                index * 20_000 + 8_000,
            )
            for index in range(1, 16)
        ],
    )

    planned = plan_voiceprint_capture(
        project_dir,
        sample_count=3,
        max_seconds=12.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
    )
    session = load_voiceprint_capture_review_session(
        summary=planned, source_path=project_dir / "source" / "meeting.mp4"
    )

    assert len(planned.speakers[0].clips) == 12
    assert sum(1 for clip in planned.speakers[0].clips if clip.recommended) == 3
    assert sum(1 for clip in session.speakers[0].clips if clip.included) == 3
    recommended = [clip for clip in planned.speakers[0].clips if clip.recommended]
    assert (
        recommended[-1].source_begin_time_ms - recommended[0].source_begin_time_ms
        >= 100_000
    )


def test_capture_voiceprints_skips_definitely_bad_audio(
    monkeypatch, tmp_path: Path
) -> None:
    """Captured WAVs with clearly unusable audio should not enter the store."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        [
            _sentence(1, "这是第一段有效声纹样本。", 0, 10_000),
            _sentence(2, "这是第二段有效声纹样本。", 30_000, 40_000),
        ],
    )

    def fake_extract_audio_clip(
        source, output, start_seconds, duration_seconds
    ) -> Path:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        amplitude = 0 if "clip_001" in str(output) else 1200
        _write_wav(Path(output), amplitude)
        return Path(output)

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)

    summary = capture_voiceprints(
        project_dir,
        sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
        dry_run=False,
    )
    samples = list_all_voiceprint_samples(summary.db_path)

    assert summary.sample_count == 1
    assert len(samples) == 1
    assert summary.speakers[0].clips[0].audio_reason == "audio=ok"


def test_capture_voiceprints_prefers_embedding_central_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    """Final persistence should prefer candidates near the embedding centroid."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        [
            _sentence(
                index,
                f"这是第 {index} 段适合做声纹的完整表达。",
                index * 20_000,
                index * 20_000 + 8_000,
            )
            for index in range(1, 5)
        ],
    )

    def fake_extract_audio_clip(
        source, output, start_seconds, duration_seconds
    ) -> Path:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _write_wav(Path(output), 1200)
        return Path(output)

    def fake_embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
        return [0.0, 1.0] if "clip_004" in str(path) else [1.0, 0.0]

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)
    monkeypatch.setattr("app.voiceprints.embed_audio_file", fake_embed_audio_file)

    summary = capture_voiceprints(
        project_dir,
        sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.0,
        store_dir=tmp_path / "voiceprints",
        dry_run=False,
    )
    texts = [clip.text for clip in summary.speakers[0].clips]

    assert len(texts) == 2
    assert "这是第 4 段适合做声纹的完整表达。" not in texts


def test_capture_quarantines_clips_far_from_existing_person_centroid(
    monkeypatch, tmp_path: Path
) -> None:
    """New clips that do not sound like the known person enter quarantined."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    person = create_voiceprint_person("Alice", db_path)
    _seed_person_embeddings(store_dir, db_path, vector=[1.0, 0.0], count=3)

    def fake_extract_audio_clip(
        source, output, start_seconds, duration_seconds
    ) -> Path:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _write_wav(Path(output), 1200)
        return Path(output)

    # New capture clips embed orthogonally to the library centroid.
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)
    monkeypatch.setattr(
        "app.voiceprints.embed_audio_file",
        lambda path, *, provider: [0.0, 1.0],
    )

    summary = capture_voiceprints(
        project_dir,
        sample_count=2,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=store_dir,
        dry_run=False,
    )

    assert summary.speakers[0].person_id == person.speaker_id
    rows = list_voiceprint_samples_for_project(
        summary.project_id or _project_id_of(project_dir), db_path
    )
    assert rows
    assert {row.sample_status for row in rows} == {"quarantined"}


def test_capture_keeps_matching_clips_active_with_existing_centroid(
    monkeypatch, tmp_path: Path
) -> None:
    """Clips consistent with the person's centroid keep the active status."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    create_voiceprint_person("Alice", db_path)
    _seed_person_embeddings(store_dir, db_path, vector=[1.0, 0.0], count=3)

    def fake_extract_audio_clip(
        source, output, start_seconds, duration_seconds
    ) -> Path:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _write_wav(Path(output), 1200)
        return Path(output)

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", fake_extract_audio_clip)
    monkeypatch.setattr(
        "app.voiceprints.embed_audio_file",
        lambda path, *, provider: [1.0, 0.0],
    )

    summary = capture_voiceprints(
        project_dir,
        sample_count=2,
        max_seconds=10.0,
        padding_seconds=0.0,
        store_dir=store_dir,
        dry_run=False,
    )

    project_rows = list_voiceprint_samples_for_project(
        summary.project_id or _project_id_of(project_dir), db_path
    )
    assert project_rows
    assert {row.sample_status for row in project_rows} == {"active"}


def _project_id_of(project_dir: Path) -> str:
    """Read the project id from the manifest."""
    return load_manifest(project_dir).project_id


def _seed_person_embeddings(
    store_dir: Path, db_path: Path, *, vector: list[float], count: int
) -> None:
    """Store embedded library samples for Alice under the default model."""
    _provider, model = resolve_voiceprint_embedding_options(provider=None, model=None)
    source = store_dir / "seed-source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"seed")
    samples = []
    for index in range(count):
        clip_path = store_dir / "clips" / "seed" / f"seed_{index}.wav"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"seed-{index}".encode())
        samples.append(
            StoredVoiceprintSample(
                speaker_name="Alice",
                project_id="p-seed",
                project_path=store_dir,
                project_speaker_id=0,
                source_path=source,
                clip_path=clip_path,
                clip_rel_path=str(clip_path.relative_to(store_dir)),
                source_begin_time_ms=index * 1000,
                source_end_time_ms=index * 1000 + 900,
                clip_begin_time_ms=0,
                clip_end_time_ms=900,
                transcript_text=f"seed {index}",
            )
        )
    _db, rows = store_voiceprint_samples_with_rows(samples, db_path)
    for row in rows:
        upsert_voiceprint_embedding(row.sample_id, model, vector, db_path)


def test_voiceprint_capture_command_accepts_project_id_for_dry_run(
    tmp_path: Path,
) -> None:
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


def test_voiceprint_review_command_summarizes_project_and_library(
    tmp_path: Path,
) -> None:
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
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        '{"0": "Alice"}\n', encoding="utf-8"
    )
    return project_dir


def _sample_project_with_sentences(tmp_path: Path, sentences: list[dict]) -> Path:
    """Create a minimal project with caller-provided transcript sentences."""
    project_dir = _sample_project(tmp_path)
    payload = {
        "full_text": "".join(str(sentence["text"]) for sentence in sentences),
        "detected_speakers": [0],
        "sentences": sentences,
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
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


def _write_wav(path: Path, amplitude: int) -> None:
    """Write a tiny mono PCM WAV fixture."""
    import wave

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(amplitude.to_bytes(2, "little", signed=True) * 1600)
