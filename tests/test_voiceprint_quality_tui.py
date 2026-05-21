"""Tests for voiceprint quality review TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from app.presentation.tui import voiceprint_quality as quality_tui
from app.presentation.tui.voiceprint_quality import (
    VoiceprintQualityApp,
    VoiceprintQualityHelpScreen,
)
from app.voiceprint_quality import analyze_voiceprint_quality
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_embeddings,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)


def test_voiceprint_quality_tui_stages_quarantine_and_saves(tmp_path: Path) -> None:
    """The quality TUI should let users quarantine a suspicious sample after review."""
    report = analyze_voiceprint_quality(
        store_dir=_quality_store(tmp_path), speaker="Alice"
    )

    async def scenario() -> None:
        async with VoiceprintQualityApp(report).run_test(size=(120, 24)) as pilot:
            app = pilot.app

            assert "suspicious 1" in app._overview()
            assert "critical" in app._samples_pane()

            await pilot.press("right")
            await pilot.press("x")
            await pilot.press("s")
            await pilot.pause()

            decision = pilot.app.return_value
            assert decision.saved is True
            assert list(decision.statuses.values()) == ["quarantined"]

    asyncio.run(scenario())


def test_voiceprint_quality_tui_save_refreshes_scores_in_place(tmp_path: Path) -> None:
    """Saving in review mode should keep the TUI open and recompute quality scores."""
    store_dir = _quality_store(tmp_path)
    report = analyze_voiceprint_quality(store_dir=store_dir, speaker="Alice")

    async def scenario() -> None:
        async with VoiceprintQualityApp(
            report, store_dir=store_dir, speaker="Alice"
        ).run_test(size=(120, 24)) as pilot:
            app = pilot.app

            assert app.report.suspicious_count == 1

            await pilot.press("right")
            await pilot.press("x")
            await pilot.press("s")
            await pilot.pause()

            assert app.report.suspicious_count == 0
            assert "已保存 1 个变更" in str(
                app.query_one("#status", Static).render()
            ) or "Saved 1 change" in str(app.query_one("#status", Static).render())

            await pilot.press("q")

    asyncio.run(scenario())


def test_voiceprint_quality_tui_marks_verified_active(tmp_path: Path) -> None:
    """The quality TUI should mark human-confirmed outliers without excluding them."""
    store_dir = _quality_store(tmp_path)
    report = analyze_voiceprint_quality(store_dir=store_dir, speaker="Alice")

    async def scenario() -> None:
        async with VoiceprintQualityApp(
            report, store_dir=store_dir, speaker="Alice"
        ).run_test(size=(120, 24)) as pilot:
            app = pilot.app

            assert app.report.suspicious_count == 1

            await pilot.press("right")
            await pilot.press("v")
            await pilot.press("s")
            await pilot.pause()

            assert app.report.suspicious_count == 0
            assert (
                len(
                    list_voiceprint_embeddings(
                        LOCAL_SPEECHBRAIN_MODEL, get_voiceprint_db_path(store_dir)
                    )
                )
                == 4
            )

    asyncio.run(scenario())


def test_voiceprint_quality_tui_space_toggles_playback(
    monkeypatch, tmp_path: Path
) -> None:
    """Space should play and stop the selected suspicious WAV sample."""
    store_dir = _quality_store(tmp_path)
    report = analyze_voiceprint_quality(store_dir=store_dir, speaker="Alice")
    process = _RunningFakeProcess()
    starts = 0

    monkeypatch.setattr(
        quality_tui,
        "build_voiceprint_play_command",
        lambda path: ["fake-player", str(path)],
    )

    def fake_popen(*args, **kwargs) -> _RunningFakeProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(quality_tui.subprocess, "Popen", fake_popen)

    async def scenario() -> None:
        async with VoiceprintQualityApp(
            report, store_dir=store_dir
        ).run_test() as pilot:
            await pilot.press("space")

            assert starts == 1
            assert pilot.app.playback_process is process

            await pilot.press("space")

            assert starts == 1
            assert process.terminated is True
            assert pilot.app.playback_process is None

    asyncio.run(scenario())


def test_voiceprint_quality_tui_question_mark_shows_help(tmp_path: Path) -> None:
    """The ? key should open and close shortcut help."""
    report = analyze_voiceprint_quality(
        store_dir=_quality_store(tmp_path), speaker="Alice"
    )

    async def scenario() -> None:
        async with VoiceprintQualityApp(report).run_test() as pilot:
            await pilot.press("?")
            await pilot.pause()

            screen = pilot.app.screen
            text = str(screen.query_one("#quality-help", Static).render())

            assert isinstance(screen, VoiceprintQualityHelpScreen)
            assert "Voiceprint Quality Review" in text

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, VoiceprintQualityHelpScreen)

    asyncio.run(scenario())


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


def _quality_store(tmp_path: Path) -> Path:
    """Create a voiceprint store with one obvious outlier."""
    store_dir = tmp_path / "quality-voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    samples = [
        _stored_sample(store_dir, source_path, "Alice", index=index)
        for index in range(1, 5)
    ]
    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples_for_project("project-quality", db_path)
    vectors = ([1.0, 0.0], [0.98, 0.02], [0.99, 0.01], [0.0, 1.0])
    for row, vector in zip(rows, vectors, strict=True):
        upsert_voiceprint_embedding(
            row.sample_id, LOCAL_SPEECHBRAIN_MODEL, vector, db_path
        )
    return store_dir


def _stored_sample(
    store_dir: Path,
    source_path: Path,
    speaker_name: str,
    *,
    index: int,
) -> StoredVoiceprintSample:
    """Build one stored sample fixture."""
    clip_path = (
        store_dir / "clips" / "project-quality" / "speaker_0" / f"clip_{index:03d}.wav"
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(f"{speaker_name}-{index}".encode())
    return StoredVoiceprintSample(
        speaker_name=speaker_name,
        project_id="project-quality",
        project_path=store_dir / "project-quality",
        project_speaker_id=0,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=index * 1000,
        source_end_time_ms=index * 1000 + 500,
        clip_begin_time_ms=0,
        clip_end_time_ms=500,
        transcript_text=f"sample {index}",
    )
