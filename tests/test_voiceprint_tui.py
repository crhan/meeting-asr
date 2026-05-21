"""Tests for the voiceprint library TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from app import voiceprint_tui
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)
from app.voiceprint_tui import (
    FOCUSED_PANE_CLASS,
    UNFOCUSED_PANE_CLASS,
    VoiceprintHelpScreen,
    VoiceprintLibraryApp,
    load_voiceprint_library_session,
    render_voiceprint_library_summary,
)


def test_load_voiceprint_library_session_groups_samples_without_losing_counts(
    tmp_path: Path,
) -> None:
    """The browser session should expose speaker, sample, and embedding coverage."""
    store_dir = _voiceprint_store(tmp_path)

    session = load_voiceprint_library_session(store_dir=store_dir, page_size=1)
    summary = render_voiceprint_library_summary(session)

    assert session.db_path == get_voiceprint_db_path(store_dir)
    assert [speaker.name for speaker in session.speakers] == ["Alice", "Bob"]
    assert [len(speaker.samples) for speaker in session.speakers] == [2, 1]
    assert [speaker.embedded_sample_count for speaker in session.speakers] == [1, 0]
    assert "Speakers: 2 | Samples: 3 | Embedded: 1/3" in summary
    assert "Alice id=" in summary


def test_voiceprint_library_tui_browses_speakers_and_samples(tmp_path: Path) -> None:
    """The TUI should start in browse mode and use one focus model."""
    session = load_voiceprint_library_session(
        store_dir=_voiceprint_store(tmp_path), page_size=1
    )

    async def scenario() -> None:
        async with VoiceprintLibraryApp(session).run_test(size=(100, 20)) as pilot:
            app = pilot.app
            speakers = app.query_one("#speakers", Static)
            samples = app.query_one("#samples", Static)

            assert "speakers 2" in app._overview_pane()
            assert "Alice" in app._speaker_pane()
            assert "hello from alice" in app._sample_pane()
            assert speakers.has_class(FOCUSED_PANE_CLASS)
            assert samples.has_class(UNFOCUSED_PANE_CLASS)

            await pilot.press("right")
            await pilot.press("down")

            assert app.focused_column == "samples"
            assert app._speaker().selected_sample_index == 1
            assert speakers.has_class(UNFOCUSED_PANE_CLASS)
            assert samples.has_class(FOCUSED_PANE_CLASS)

    asyncio.run(scenario())


def test_voiceprint_library_tui_question_mark_shows_help(tmp_path: Path) -> None:
    """The ? key should open and close a shortcut help modal."""
    session = load_voiceprint_library_session(store_dir=_voiceprint_store(tmp_path))

    async def scenario() -> None:
        async with VoiceprintLibraryApp(session).run_test() as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_screen = pilot.app.screen
            help_text = str(help_screen.query_one("#voiceprint-help", Static).render())

            assert isinstance(help_screen, VoiceprintHelpScreen)
            assert "Voiceprint Library Shortcuts" in help_text
            assert "delete-sample" in help_text

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, VoiceprintHelpScreen)

    asyncio.run(scenario())


def test_voiceprint_library_tui_space_toggles_playback(
    monkeypatch, tmp_path: Path
) -> None:
    """Space should play and stop the selected voiceprint WAV sample."""
    session = load_voiceprint_library_session(store_dir=_voiceprint_store(tmp_path))
    process = _RunningFakeProcess()
    starts = 0

    monkeypatch.setattr(
        voiceprint_tui,
        "build_voiceprint_play_command",
        lambda path: ["fake-player", str(path)],
    )

    def fake_popen(*args, **kwargs) -> _RunningFakeProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(voiceprint_tui.subprocess, "Popen", fake_popen)

    async def scenario() -> None:
        async with VoiceprintLibraryApp(session).run_test() as pilot:
            await pilot.press("space")

            assert starts == 1
            assert pilot.app.playback_process is process

            await pilot.press("space")

            assert starts == 1
            assert process.terminated is True
            assert pilot.app.playback_process is None

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


def _voiceprint_store(tmp_path: Path) -> Path:
    """Create a small voiceprint store fixture."""
    store_dir = tmp_path / "voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    samples = [
        _sample(
            store_dir,
            source_path,
            "Alice",
            speaker_id=0,
            index=1,
            text="hello from alice",
        ),
        _sample(
            store_dir,
            source_path,
            "Alice",
            speaker_id=0,
            index=2,
            text="second alice sample",
        ),
        _sample(
            store_dir, source_path, "Bob", speaker_id=1, index=1, text="hello from bob"
        ),
    ]
    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))
    first_sample_id = list_all_voiceprint_samples(db_path)[0].sample_id
    upsert_voiceprint_embedding(
        first_sample_id, LOCAL_SPEECHBRAIN_MODEL, [0.1, 0.2], db_path
    )
    return store_dir


def _sample(
    store_dir: Path,
    source_path: Path,
    speaker_name: str,
    *,
    speaker_id: int,
    index: int,
    text: str,
) -> StoredVoiceprintSample:
    """Build one stored voiceprint sample fixture."""
    clip_path = (
        store_dir
        / "clips"
        / "project-1"
        / f"speaker_{speaker_id}"
        / f"clip_{index:03d}.wav"
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(f"{speaker_name}-{index}".encode())
    return StoredVoiceprintSample(
        speaker_name=speaker_name,
        project_id="project-1",
        project_path=store_dir / "project-1",
        project_speaker_id=speaker_id,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=index * 1000,
        source_end_time_ms=index * 1000 + 500,
        clip_begin_time_ms=0,
        clip_end_time_ms=500,
        transcript_text=text,
    )
