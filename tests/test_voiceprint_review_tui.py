"""Tests for the unified voiceprint review TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Static

import app.presentation.tui.voiceprint_review as voiceprint_review
from app.presentation.cli.i18n import configure_cli_language
from app.presentation.tui.voiceprint_review_text import quality_reason_text
from app.presentation.tui.voiceprint import load_voiceprint_library_session
from app.presentation.tui.voiceprint_capture import (
    load_voiceprint_capture_review_session,
)
from app.presentation.tui.voiceprint_review import (
    VoiceprintReviewApp,
    VoiceprintReviewHelpScreen,
    VoiceprintReviewSession,
)
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_quality import analyze_voiceprint_quality
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_embeddings,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)
from app.voiceprints import VoiceprintCaptureSummary, VoiceprintClip, VoiceprintSpeaker


def test_voiceprint_review_tui_switches_project_and_library_views(
    tmp_path: Path,
) -> None:
    """Unified review should switch between project candidates and global library."""
    session = _review_session(tmp_path)
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.mode == "project"
            assert "VOICEPRINT REVIEW" in app._overview_pane()
            assert "view=[bold cyan]Project candidates" in app._overview_pane()
            assert "Voiceprint Demo" in app._overview_pane()
            assert "project-1" in app._overview_pane()
            assert "status=named" in app._overview_pane()
            assert "Source" in app._overview_pane()
            assert "meeting.mp4" in app._overview_pane()
            assert "verify samples" in app._overview_pane()
            assert "selected 2/2" in app._speaker_pane()
            assert "project sample one" in app._sample_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert app.mode == "library"
            assert "view=[bold cyan]Global library" in app._overview_pane()
            assert "Global library" in app._overview_pane()
            assert "Global voiceprint people" in app._speaker_pane()
            assert "library sample one" in app._sample_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert app.mode == "quality"
            assert "view=[bold cyan]Quality review" in app._overview_pane()
            assert "Voiceprint quality" in app._speaker_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert app.mode == "project"

    asyncio.run(scenario())


def test_voiceprint_review_tui_uses_chinese_language(tmp_path: Path) -> None:
    """Unified voiceprint review should localize visible project guidance."""
    try:
        configure_cli_language("zh")
        app = VoiceprintReviewApp(_review_session(tmp_path))

        assert "视图=[bold cyan]项目候选样本" in app._overview_pane()
        assert "[b]项目[/b]" in app._overview_pane()
        assert "[b]目标[/b]" in app._overview_pane()
        assert "全局声纹人员" in app._library_speaker_pane()
    finally:
        configure_cli_language("en")


def test_voiceprint_review_tui_saves_only_selected_project_samples(
    tmp_path: Path,
) -> None:
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
    assert app.return_value.selected_clip_rel_paths == frozenset(
        {"clips/project-1/speaker_0/clip_002.wav"}
    )


def test_voiceprint_review_tui_excludes_current_speaker_samples(tmp_path: Path) -> None:
    """The dedicated exclude shortcut should clear all samples for the current speaker."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert "selected 2/2" in app._speaker_pane()

            await pilot.press("d")
            await pilot.pause()

            assert "selected 0/2" in app._speaker_pane()
            assert app._selected_clip_rel_paths() == set()

    asyncio.run(scenario())


def test_voiceprint_review_defaults_high_score_matches_unchecked_and_shows_score(
    tmp_path: Path,
) -> None:
    """High-score project matches should not be captured again by default."""
    app = VoiceprintReviewApp(
        _review_session(
            tmp_path,
            match_candidates={
                0: {
                    "accepted": False,
                    "name": None,
                    "best_name": "Alice",
                    "best_score": 0.76,
                    "threshold": 0.85,
                }
            },
        )
    )

    assert "score=0.760" in app._speaker_pane()
    assert "selected 0/2" in app._speaker_pane()
    assert "[green]score=0.760" in app._speaker_pane()
    assert "[dim]selected 0/2" in app._speaker_pane()
    assert "score 0.760" in app._overview_pane()
    assert app._selected_clip_rel_paths() == set()


def test_voiceprint_review_tui_uses_colored_rows_and_checkmarks(tmp_path: Path) -> None:
    """Project rows should expose color markup for focus and sample state."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    speaker_pane = app._speaker_pane()
    sample_pane = app._sample_pane()

    assert "[reverse]" in speaker_pane
    assert "[bold yellow]>" in speaker_pane
    assert "[green]selected 2/2" in speaker_pane
    assert "[green]x[/]" in sample_pane


def test_voiceprint_review_project_playback_shows_progress(
    monkeypatch, tmp_path: Path
) -> None:
    """Playing a project sample should show a visible state and progress."""
    fake_process = _FakePlaybackProcess()
    monkeypatch.setattr(
        voiceprint_review, "_start_player", lambda command: fake_process
    )
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("right")
            await pilot.press("space")
            await pilot.pause()

            status = str(app.query_one("#status", Static).render())
            assert "Playing project sample" in status
            assert "0:00/0:01" in status
            assert "PLAY" in app._sample_pane()

            await pilot.press("space")
            await pilot.pause()

            assert fake_process.terminated is True
            assert "Stopped sample playback" in str(
                app.query_one("#status", Static).render()
            )

    asyncio.run(scenario())


def test_voiceprint_review_refuses_save_from_global_library(tmp_path: Path) -> None:
    """Save should be explicit to project candidates, not whichever view is open."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("tab")
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()

            assert app.return_value is None
            assert app.mode == "library"
            assert "Switch to Project candidates" in str(
                app.query_one("#status", Static).render()
            )

    asyncio.run(scenario())


def test_voiceprint_review_quality_mode_saves_and_refreshes(tmp_path: Path) -> None:
    """Unified review should update sample quality status without leaving the TUI."""
    store_dir = _quality_store(tmp_path)
    session = VoiceprintReviewSession(
        capture=None,
        library=load_voiceprint_library_session(store_dir=store_dir),
        quality=analyze_voiceprint_quality(store_dir=store_dir),
        store_dir=store_dir,
        initial_mode="quality",
    )
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.mode == "quality"
            assert app.session.quality.suspicious_count == 1

            await pilot.press("right")
            await pilot.press("x")
            await pilot.press("s")
            await pilot.pause()

            assert app.return_value is None
            assert app.session.quality.suspicious_count == 0
            assert "Saved 1 quality change" in str(
                app.query_one("#status", Static).render()
            )

    asyncio.run(scenario())


def test_voiceprint_review_quality_mode_marks_verified_active(tmp_path: Path) -> None:
    """Unified review should let users confirm a true sample without removing it from matching."""
    store_dir = _quality_store(tmp_path)
    db_path = get_voiceprint_db_path(store_dir)
    session = VoiceprintReviewSession(
        capture=None,
        library=load_voiceprint_library_session(store_dir=store_dir),
        quality=analyze_voiceprint_quality(store_dir=store_dir),
        store_dir=store_dir,
        initial_mode="quality",
    )
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.session.quality.suspicious_count == 1

            await pilot.press("right")
            await pilot.press("v")
            await pilot.press("s")
            await pilot.pause()

            # Verify-active confirms identity but KEEPS the sample in matching
            # (unlike quarantine, which drops it). The low-score outlier therefore
            # stays flagged as suspicious — confirming identity must not mask a poor
            # embedding score. The sample is kept, not removed (4 embeddings), and
            # its lifecycle status is now verified-active.
            assert app.session.quality.suspicious_count == 1
            assert (
                len(list_voiceprint_embeddings(LOCAL_SPEECHBRAIN_MODEL, db_path)) == 4
            )
            statuses = [
                row.sample_status
                for row in list_voiceprint_samples_for_project("project-1", db_path)
            ]
            assert statuses.count("verified-active") == 1

    asyncio.run(scenario())


def test_voiceprint_review_quality_rows_do_not_escape_markup(tmp_path: Path) -> None:
    """Quality rows should render Rich markup instead of showing markup tags as text."""
    store_dir = _quality_store(tmp_path)
    session = VoiceprintReviewSession(
        capture=None,
        library=load_voiceprint_library_session(store_dir=store_dir),
        quality=analyze_voiceprint_quality(store_dir=store_dir),
        store_dir=store_dir,
        initial_mode="quality",
    )
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)):
            pane = app._sample_pane()

            assert "\\[dim]" not in pane
            assert "\\[cyan]" not in pane
            assert "[cyan]#1[/]" in pane
            assert "vps-" in pane

    asyncio.run(scenario())


def test_voiceprint_quality_reason_is_localized() -> None:
    """Quality reason text should be human-facing in Chinese."""
    try:
        configure_cli_language("zh")
        assert (
            quality_reason_text("statistical outlier")
            == "统计离群：这段样本和此人的其他声纹样本差异明显"
        )
        assert (
            quality_reason_text("cluster-consistent")
            == "声纹一致：这段样本和此人的声纹簇匹配"
        )
        assert quality_reason_text("score<0.60") == "分数低于阈值（0.60）"
        assert (
            quality_reason_text("human verified active")
            == "人工确认：这段样本保留参与匹配，不再作为质量风险"
        )
    finally:
        configure_cli_language("en")


def test_voiceprint_review_without_project_starts_in_library_mode(
    tmp_path: Path,
) -> None:
    """Without a project, review should behave as the global library browser."""
    store_dir = _store(tmp_path)
    session = VoiceprintReviewSession(
        capture=None,
        library=load_voiceprint_library_session(store_dir=store_dir),
        quality=analyze_voiceprint_quality(store_dir=store_dir),
        store_dir=store_dir,
    )
    app = VoiceprintReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            assert app.mode == "library"
            assert "Tab -> Quality review" in app._overview_pane()

            await pilot.press("p")
            await pilot.pause()

            status = str(app.query_one("#status", Static).render())
            assert "project candidates are unavailable" in status

    asyncio.run(scenario())


def test_voiceprint_review_question_mark_shows_help(tmp_path: Path) -> None:
    """The shared voiceprint TUI should show help from the standalone app."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("?")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewHelpScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, VoiceprintReviewHelpScreen)

    asyncio.run(scenario())


def test_voiceprint_review_quit_returns_unsaved_decision(tmp_path: Path) -> None:
    """Quit should return a non-saving decision from the standalone app."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("q")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.saved is False


def test_voiceprint_review_escape_returns_unsaved_decision(tmp_path: Path) -> None:
    """Esc should behave like back/quit in the unified voiceprint review."""
    app = VoiceprintReviewApp(_review_session(tmp_path))

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("escape")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.saved is False


def _review_session(
    tmp_path: Path,
    *,
    match_candidates: dict[int, object] | None = None,
) -> VoiceprintReviewSession:
    """Build a unified review session fixture."""
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    capture = load_voiceprint_capture_review_session(
        summary=_capture_summary(tmp_path),
        source_path=source_path,
        page_size=2,
        project_title="Voiceprint Demo",
        project_status="named",
        source_name="meeting.mp4",
        meeting_time="2026-05-05T09:00:00+08:00",
        match_candidates=match_candidates,
    )
    store_dir = _store(tmp_path)
    library = load_voiceprint_library_session(store_dir=store_dir, page_size=1)
    quality = analyze_voiceprint_quality(store_dir=store_dir)
    return VoiceprintReviewSession(
        capture=capture, library=library, quality=quality, store_dir=store_dir
    )


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
            _sample(
                store_dir,
                source_path,
                "Alice",
                speaker_id=0,
                index=1,
                text="library sample one",
            ),
            _sample(
                store_dir,
                source_path,
                "Alice",
                speaker_id=0,
                index=2,
                text="library sample two",
            ),
        ],
        get_voiceprint_db_path(store_dir),
    )
    return store_dir


def _quality_store(tmp_path: Path) -> Path:
    """Create a voiceprint store with one suspicious embedded sample."""
    store_dir = tmp_path / "quality-voiceprints"
    source_path = tmp_path / "quality-meeting.mp4"
    source_path.write_bytes(b"source")
    samples = [
        _sample(
            store_dir,
            source_path,
            "Alice",
            speaker_id=0,
            index=index,
            text=f"sample {index}",
        )
        for index in range(1, 5)
    ]
    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples_for_project("project-1", db_path)
    vectors = ([1.0, 0.0], [0.98, 0.02], [0.99, 0.01], [0.0, 1.0])
    for row, vector in zip(rows, vectors, strict=True):
        upsert_voiceprint_embedding(
            row.sample_id, LOCAL_SPEECHBRAIN_MODEL, vector, db_path
        )
    return store_dir


class _FakePlaybackProcess:
    """Small subprocess.Popen test double for playback controls."""

    def __init__(self) -> None:
        """Create a running fake process."""
        self.terminated = False

    def poll(self) -> int | None:
        """Return None while the fake process is playing."""
        return 0 if self.terminated else None

    def terminate(self) -> None:
        """Mark the fake process as terminated."""
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        """Return a successful exit code."""
        self.terminated = True
        return 0

    def kill(self) -> None:
        """Mark the fake process as killed."""
        self.terminated = True


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
