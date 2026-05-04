"""Tests for the unified voiceprint review TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

from app.presentation.tui.voiceprint import load_voiceprint_library_session
from app.presentation.tui.voiceprint_capture import load_voiceprint_capture_review_session
from app.presentation.tui.voiceprint_review import VoiceprintReviewApp, VoiceprintReviewSession
from app.voiceprint_store import StoredVoiceprintSample, get_voiceprint_db_path, store_voiceprint_samples
from app.voiceprints import VoiceprintCaptureSummary, VoiceprintClip, VoiceprintSpeaker


def test_voiceprint_review_tui_switches_project_and_library_views(tmp_path: Path) -> None:
    """Unified review should switch between project candidates and global library."""
    session = _review_session(tmp_path)
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.mode == "project"
            assert "Project candidates" in app._overview_pane()
            assert "selected 2/2" in app._speaker_pane()
            assert "project sample one" in app._sample_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert app.mode == "library"
            assert "Global library" in app._overview_pane()
            assert "Global voiceprint people" in app._speaker_pane()
            assert "library sample one" in app._sample_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert app.mode == "project"

    asyncio.run(scenario())


def test_voiceprint_review_tui_saves_only_selected_project_samples(tmp_path: Path) -> None:
    """Saving from the unified TUI should return checked project clip paths only."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            speakers = app.query_one("#speakers", Static)
            samples = app.query_one("#samples", Static)

            assert speakers.has_class("focused-pane")
            assert samples.has_class("unfocused-pane")

            await pilot.press("right")
            await pilot.press("x")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.saved is True
    assert app.return_value.selected_clip_rel_paths == frozenset({"clips/project-1/speaker_0/clip_002.wav"})


def test_voiceprint_review_without_project_starts_in_library_mode(tmp_path: Path) -> None:
    """Without a project, review should behave as the global library browser."""
    session = VoiceprintReviewSession(capture=None, library=load_voiceprint_library_session(store_dir=_store(tmp_path)))
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.mode == "library"
            assert "Project unavailable" in app._overview_pane()

            await pilot.press("p")
            await pilot.pause()

            status = str(app.query_one("#status", Static).render())
            assert "project candidates are unavailable" in status

    asyncio.run(scenario())


def _review_session(tmp_path: Path) -> VoiceprintReviewSession:
    """Build a unified review session fixture."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    capture = load_voiceprint_capture_review_session(
        summary=_capture_summary(tmp_path),
        source_path=source_path,
        page_size=2,
    )
    library = load_voiceprint_library_session(store_dir=_store(tmp_path), page_size=1)
    return VoiceprintReviewSession(capture=capture, library=library)


def _capture_summary(tmp_path: Path) -> VoiceprintCaptureSummary:
    """Build planned project capture samples."""
    store_dir = tmp_path / "voiceprints"
    clip_dir = store_dir / "clips"
    clips = [
        VoiceprintClip(
            path=clip_dir / "project-1" / "speaker_0" / "clip_001.wav",
            rel_path="clips/project-1/speaker_0/clip_001.wav",
            source_begin_time_ms=1000,
            source_end_time_ms=2000,
            clip_begin_time_ms=1000,
            clip_end_time_ms=2000,
            text="project sample one",
        ),
        VoiceprintClip(
            path=clip_dir / "project-1" / "speaker_0" / "clip_002.wav",
            rel_path="clips/project-1/speaker_0/clip_002.wav",
            source_begin_time_ms=3000,
            source_end_time_ms=4000,
            clip_begin_time_ms=3000,
            clip_end_time_ms=4000,
            text="project sample two",
        ),
    ]
    return VoiceprintCaptureSummary(
        store_dir=store_dir,
        db_path=get_voiceprint_db_path(store_dir),
        clip_dir=clip_dir,
        speakers=[VoiceprintSpeaker(0, "Alice", None, None, clips)],
        dry_run=True,
    )


def _store(tmp_path: Path) -> Path:
    """Create a small global voiceprint store fixture."""
    store_dir = tmp_path / "voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    store_voiceprint_samples(
        [
            _sample(store_dir, source_path, "Alice", speaker_id=0, index=1, text="library sample one"),
            _sample(store_dir, source_path, "Alice", speaker_id=0, index=2, text="library sample two"),
        ],
        get_voiceprint_db_path(store_dir),
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
    clip_path = store_dir / "clips" / "project-1" / f"speaker_{speaker_id}" / f"clip_{index:03d}.wav"
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
