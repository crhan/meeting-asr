"""Tests for the speaker review TUI behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Input, Static

from app import speaker_tui
from app.models import SentenceSegment
from app.speaker_tui import ReviewSpeaker, SpeakerReviewApp, SpeakerReviewSession


def test_speaker_review_tui_starts_in_browse_mode() -> None:
    """The TUI should not start with a hidden focused name prompt."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            field = pilot.app.query_one("#name-input", Input)
            identity = pilot.app.query_one("#identity", Static)
            main = pilot.app.query_one("#main")
            assert len(list(main.children)) == 2
            assert field.display is False
            assert field.disabled is True
            assert identity.display is False
            assert pilot.app.focused is None

            await pilot.press("/")

            assert field.display is True
            assert field.disabled is False
            assert identity.display is True
            assert pilot.app.focused is field

            await pilot.press("escape")

            assert field.display is False
            assert field.disabled is True
            assert identity.display is False
            assert pilot.app.focused is None

    asyncio.run(scenario())


def test_speaker_review_tui_plays_selected_sample(monkeypatch) -> None:
    """Space should play the currently selected sample, not the whole speaker batch."""
    captured: dict[str, float] = {}

    def fake_command(
        *,
        media: Path,
        start_seconds: float,
        duration_seconds: float | None,
    ) -> list[str]:
        captured["start_seconds"] = start_seconds
        captured["duration_seconds"] = duration_seconds or 0.0
        return ["fake-player"]

    monkeypatch.setattr(speaker_tui, "build_audio_preview_command", fake_command)
    monkeypatch.setattr(
        speaker_tui.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            await pilot.press("right")
            await pilot.press("down")
            await pilot.press("space")

    asyncio.run(scenario())

    assert captured["start_seconds"] == 1.5
    assert captured["duration_seconds"] == 2.0


def test_speaker_review_tui_uses_focused_columns_for_movement() -> None:
    """Arrow keys and HJKL should act on the currently focused column."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(two_speakers=True)).run_test() as pilot:
            app = pilot.app

            await pilot.press("down")

            assert app.selected_speaker_index == 1
            assert app._speaker().selected_sample_index == 0

            await pilot.press("right")
            await pilot.press("down")

            assert app.selected_speaker_index == 1
            assert app._speaker().selected_sample_index == 1

            await pilot.press("h")
            await pilot.press("k")

            assert app.selected_speaker_index == 0

    asyncio.run(scenario())


def test_speaker_review_tui_pages_samples() -> None:
    """Sample pagination should replace the old growing more-samples list."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(page_size=1)).run_test() as pilot:
            app = pilot.app
            speaker = app._speaker()

            assert [segment.text for segment in app._visible_segments(speaker)[1]] == ["第一句"]

            await pilot.press("]")

            assert speaker.selected_sample_index == 1
            assert [segment.text for segment in app._visible_segments(speaker)[1]] == ["第二句"]

    asyncio.run(scenario())


class _FakeProcess:
    """Minimal fake process for playback tests."""

    def poll(self) -> int:
        """Return an already-finished status."""
        return 0

    def terminate(self) -> None:
        """Pretend to terminate playback."""

    def wait(self, timeout: int | None = None) -> int:
        """Pretend playback has exited."""
        return 0

    def kill(self) -> None:
        """Pretend to kill playback."""


def _session(
    *,
    page_size: int | None = None,
    two_speakers: bool = False,
) -> SpeakerReviewSession:
    """Build a minimal review session."""
    segments = [
        SentenceSegment(
            begin_time_ms=0,
            end_time_ms=1000,
            text="第一句",
            speaker_id=0,
            sentence_id=1,
        ),
        SentenceSegment(
            begin_time_ms=2000,
            end_time_ms=2500,
            text="第二句",
            speaker_id=0,
            sentence_id=2,
        ),
    ]
    speakers = [ReviewSpeaker(0, "Speaker A", segments, "Speaker A", None)]
    if two_speakers:
        speakers.append(ReviewSpeaker(1, "Speaker B", segments, "Speaker B", None))
    return SpeakerReviewSession(
        project_dir=Path("."),
        source_media=Path("source.mp4"),
        speakers=speakers,
        people_names=["欧丁"],
        page_size=page_size,
    )
