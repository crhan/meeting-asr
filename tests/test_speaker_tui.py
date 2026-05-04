"""Tests for the speaker review TUI behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import Input, Static

from app import speaker_tui
from app.correction_types import CorrectionEditSummary
from app.models import SentenceSegment
from app.project_manager import create_project, project_paths
from app.speaker_tui import (
    CorrectionQueuedScreen,
    FOCUSED_PANE_CLASS,
    IdentityEditScreen,
    KnownPerson,
    ReviewSpeaker,
    SpeakerMatchCandidate,
    SentenceCorrectionScreen,
    SpeakerReviewApp,
    SpeakerReviewDecision,
    SpeakerReviewOverview,
    SpeakerReviewSession,
    ShortcutHelpScreen,
    UNFOCUSED_PANE_CLASS,
    VoiceprintReviewProgress,
    load_speaker_review_session,
)
from app.presentation.tui.speaker_save import (
    CorrectionProposalDiffScreen,
    SpeakerReviewSaveOutcome,
    SpeakerReviewSaveScreen,
    _styled_diff_text,
)
from app.presentation.tui.speaker_matches import SpeakerMatchPerson
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)


def test_speaker_review_tui_starts_in_browse_mode() -> None:
    """The TUI should open identity editing as a modal."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            main = pilot.app.query_one("#main")
            assert len(list(main.children)) == 2
            assert pilot.app.focused is None

            await pilot.press("/")
            await pilot.pause()

            assert isinstance(pilot.app.screen, IdentityEditScreen)
            field = pilot.app.screen.query_one("#identity-search", Input)
            assert pilot.app.focused is field

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, IdentityEditScreen)
            assert pilot.app.focused is None

    asyncio.run(scenario())


def test_speaker_review_tui_shows_project_workflow_status() -> None:
    """The top overview should expose project, workflow, match, and risk state."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(with_status=True)).run_test() as pilot:
            overview = pilot.app._overview_pane()

            assert "[b]Project[/b]  Demo" in overview
            assert "00:00:02.500" in overview
            assert "2 speakers" in overview
            assert "1 Match=[green]done" in overview
            assert "2 Names=[green]saved 2/2" in overview
            assert "ignored 0" in overview
            assert "3 Capture=[yellow]todo 1" in overview
            assert "4 Embed=[yellow]todo 1" in overview
            assert "exports/transcript_named.txt" in overview
            assert "exports/subtitle_named.srt" in overview
            assert "conflict 1 | mismatch 0" in overview
            assert "score avg 0.875, best 0.950" in overview

    asyncio.run(scenario())


def test_speaker_review_tui_question_mark_shows_shortcut_help() -> None:
    """The ? key should open and close a shortcut help modal."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            await pilot.press("?")
            await pilot.pause()

            help_screen = pilot.app.screen
            help_text = str(help_screen.query_one("#shortcut-help", Static).render())

            assert isinstance(help_screen, ShortcutHelpScreen)
            assert "Speaker Review Shortcuts" in help_text
            assert "Top status" in help_text
            assert "Next/Done" in help_text
            assert "Output" in help_text
            assert "h/l or left/right" in help_text
            assert "space" in help_text

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, ShortcutHelpScreen)

    asyncio.run(scenario())


def test_speaker_review_tui_highlights_focused_pane() -> None:
    """The focused column should be visible at the pane level, not only in the title."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            speakers = pilot.app.query_one("#speakers", Static)
            samples = pilot.app.query_one("#samples", Static)

            assert speakers.has_class(FOCUSED_PANE_CLASS)
            assert samples.has_class(UNFOCUSED_PANE_CLASS)
            assert "FOCUS" in pilot.app._speaker_pane()

            await pilot.press("right")

            assert speakers.has_class(UNFOCUSED_PANE_CLASS)
            assert samples.has_class(FOCUSED_PANE_CLASS)
            assert "FOCUS" in pilot.app._sample_pane()

    asyncio.run(scenario())


def test_speaker_review_tui_accepts_match_updates_status_and_saves() -> None:
    """Pilot-driven key flow should update review state and return a save result."""
    app = SpeakerReviewApp(_session(with_status=True))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            assert "conflict 1 | mismatch 0" in app._overview_pane()

            await pilot.press("a")

            assert app._speaker().current_name == "欧丁"
            assert "conflict 0 | mismatch 0" in app._overview_pane()
            assert "press `s` to write the updated speaker map" in app._overview_pane()

            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value == SpeakerReviewDecision(
        saved=True,
        mapping={0: "欧丁", 1: "欧丁"},
    )


def test_speaker_review_tui_can_ignore_anonymous_speaker() -> None:
    """Ignoring a speaker should be visible and persist as the anonymous label."""
    app = SpeakerReviewApp(_session())

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("i")

            assert app._speaker().ignored is True
            assert "selected Speaker A: ignored" in app._overview_pane()
            assert "ignored 1" in app._overview_pane()
            assert "match=- ignored" in app._speaker_pane()

            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value == SpeakerReviewDecision(
        saved=True,
        mapping={0: "Speaker A"},
    )


def test_project_review_tui_edits_transcript_text_inline() -> None:
    """Project review should edit selected transcript text without leaving the TUI."""
    app = SpeakerReviewApp(_session(allow_correction=True))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            assert isinstance(app.screen, SentenceCorrectionScreen)

            field = app.screen.query_one("#correction-input", Input)
            field.value = "第一句修正"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, CorrectionQueuedScreen)
            feedback = str(app.screen.query_one("#queued-body", Static).render())
            assert "staged" in feedback
            assert "Press s" in feedback
            assert app.return_value is None
            assert app._speaker().segments[0].text == "第一句修正"
            assert "run correction" in str(app.query_one("#status", Static).render())
            assert "edited" in app._sample_pane()

            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.saved is True
    assert app.return_value.action == "correct-inline"
    assert app.return_value.mapping == {0: "Speaker A"}
    assert app.return_value.correction_edit is not None
    assert app.return_value.correction_edit.original_text == "第一句"
    assert app.return_value.correction_edit.corrected_text == "第一句修正"
    assert len(app.return_value.correction_edits) == 1


def test_project_review_tui_keeps_multiple_inline_text_edits() -> None:
    """Multiple TUI text edits should be saved together, not overwritten."""
    app = SpeakerReviewApp(_session(allow_correction=True))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#correction-input", Input).value = "第一句修正"
            await pilot.press("enter")
            await pilot.press("enter")

            await pilot.press("right")
            await pilot.press("down")
            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#correction-input", Input).value = "第二句修正"
            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.action == "correct-inline"
    assert [edit.original_text for edit in app.return_value.correction_edits] == ["第一句", "第二句"]
    assert [edit.corrected_text for edit in app.return_value.correction_edits] == ["第一句修正", "第二句修正"]


def test_project_review_tui_save_handler_keeps_tui_open() -> None:
    """Project review save should run inside a modal instead of exiting the TUI."""
    seen: list[SpeakerReviewDecision] = []

    def save_handler(decision: SpeakerReviewDecision) -> SpeakerReviewSaveOutcome:
        seen.append(decision)
        return SpeakerReviewSaveOutcome(Path("speaker_map.json"), Path("transcript.txt"), Path("subtitle.srt"))

    app = SpeakerReviewApp(_session(allow_correction=True), save_handler=save_handler)

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, SpeakerReviewSaveScreen)
            assert app.return_value is None
            assert seen and seen[0].saved is True
            assert "Project review saved" in str(app.screen.query_one("#save-title", Static).render())

            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, SpeakerReviewSaveScreen)

    asyncio.run(scenario())


def test_project_review_tui_accepts_pending_correction_in_modal(tmp_path: Path) -> None:
    """The save modal should handle proposal acceptance without leaving the TUI."""
    accepted_paths: list[Path | None] = []
    diff_path = tmp_path / "proposal.diff"
    diff_path.write_text("- AS\n+ IaaS\n", encoding="utf-8")

    def save_handler(decision: SpeakerReviewDecision) -> SpeakerReviewSaveOutcome:
        assert len(decision.correction_edits) == 1
        return SpeakerReviewSaveOutcome(
            Path("speaker_map.json"),
            Path("transcript.txt"),
            Path("subtitle.srt"),
            _correction_summary(accepted=False, diff_path=diff_path),
        )

    def accept_handler(proposal_path: Path | None) -> SpeakerReviewSaveOutcome:
        accepted_paths.append(proposal_path)
        return SpeakerReviewSaveOutcome(None, None, None, _correction_summary(accepted=True))

    app = SpeakerReviewApp(
        _session(allow_correction=True),
        save_handler=save_handler,
        accept_handler=accept_handler,
    )

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#correction-input", Input).value = "第一句修正"
            await pilot.press("enter")
            await pilot.press("s")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, SpeakerReviewSaveScreen)
            assert "needs review" in str(app.screen.query_one("#save-title", Static).render())

            await pilot.press("d")
            await pilot.pause()

            assert isinstance(app.screen, CorrectionProposalDiffScreen)
            diff_render = app.screen.query_one("#diff-content", Static).render()
            assert "+ IaaS" in diff_render.plain
            styled_diff = _styled_diff_text("- AS\n+ IaaS\n")
            assert any(str(span.style) == "bold green" for span in styled_diff.spans)
            assert any(str(span.style) == "bold red" for span in styled_diff.spans)

            await pilot.press("enter")
            await pilot.pause()

            await pilot.press("a")
            await pilot.pause()
            await pilot.pause()

            assert accepted_paths == [Path("proposal.json")]
            assert not app.correction_edits
            assert "correction accepted" in str(app.screen.query_one("#save-title", Static).render())

    asyncio.run(scenario())


def test_correction_diff_viewer_highlights_changed_tokens_only() -> None:
    """Diff rendering should highlight changed tokens, not just whole lines."""
    styled_diff = _styled_diff_text("- 我们看一下IC系统。\n+ 我们看一下isee系统。\n")

    assert styled_diff.plain == "- 我们看一下IC系统。\n+ 我们看一下isee系统。\n"
    assert _style_for_text(styled_diff, "IC") == "bold red"
    assert _style_for_text(styled_diff, "isee") == "bold green"
    assert _style_for_text(styled_diff, "我们看一下") == "dim red"


def test_transcript_correction_input_uses_readline_cursor_keys() -> None:
    """Ctrl-F and Ctrl-B should move the cursor, not delete transcript text."""
    app = SpeakerReviewApp(_session(allow_correction=True))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            field = app.screen.query_one("#correction-input", Input)
            field.value = "abcdef"
            field.cursor_position = 0

            await pilot.press("ctrl+f")
            await pilot.pause()

            assert field.value == "abcdef"
            assert field.cursor_position == 1

            await pilot.press("ctrl+b")
            await pilot.pause()

            assert field.value == "abcdef"
            assert field.cursor_position == 0

    asyncio.run(scenario())


def test_identity_input_uses_readline_cursor_keys() -> None:
    """Identity edit should share non-destructive cursor keys."""
    app = SpeakerReviewApp(_session(people=(KnownPerson(42, "欧丁", "vpp-0000000000000042"),)))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            field = app.screen.query_one("#identity-search", Input)
            field.value = "欧丁"
            field.cursor_position = 0

            await pilot.press("ctrl+f")
            await pilot.press("ctrl+b")

            assert field.value == "欧丁"
            assert field.cursor_position == 0

    asyncio.run(scenario())


def test_speaker_review_tui_binds_existing_person_by_name() -> None:
    """Typing an existing person name should bind the stable person id."""
    app = SpeakerReviewApp(_session(people=(KnownPerson(42, "欧丁", "vpp-0000000000000042"),)))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            field = app.screen.query_one("#identity-search", Input)
            field.value = "欧丁"
            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value == SpeakerReviewDecision(
        saved=True,
        mapping={0: "欧丁"},
        person_mapping={0: 42},
        person_public_mapping={0: "vpp-0000000000000042"},
    )


def test_speaker_review_tui_shows_filterable_people_selector() -> None:
    """Name edit should visibly filter and select stable voiceprint people."""
    app = SpeakerReviewApp(
        _session(
            people=(
                KnownPerson(42, "欧丁", "vpp-0000000000000042"),
                KnownPerson(7, "敬悦", "vpp-0000000000000007"),
            )
        )
    )

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            identity = str(app.screen.query_one("#identity-list", Static).render())
            assert "People" in identity
            assert "欧丁" in identity
            assert "vpp-0000000000000042" in identity

            field = app.screen.query_one("#identity-search", Input)
            field.value = "敬"
            await pilot.pause()

            identity = str(app.screen.query_one("#identity-list", Static).render())
            assert "敬悦" in identity
            assert "vpp-0000000000000007" in identity

            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value == SpeakerReviewDecision(
        saved=True,
        mapping={0: "敬悦"},
        person_mapping={0: 7},
        person_public_mapping={0: "vpp-0000000000000007"},
    )


def test_speaker_review_identity_modal_sorts_people_by_score() -> None:
    """Identity modal should rank scored voiceprint candidates first."""
    match = SpeakerMatchCandidate(
        "墨泪",
        0.67,
        False,
        best_name="墨泪",
        best_score=0.67,
        best_person_id=10,
        candidates=(
            SpeakerMatchPerson(10, "墨泪", 0.67),
            SpeakerMatchPerson(34, "丰禾", 0.91),
            SpeakerMatchPerson(37, "华璟", 0.72),
        ),
    )
    app = SpeakerReviewApp(
        _session(
            people=(
                KnownPerson(10, "墨泪", "vpp-0000000000000010"),
                KnownPerson(37, "华璟", "vpp-0000000000000037"),
                KnownPerson(34, "丰禾", "vpp-0000000000000034"),
            )
        )
    )
    app.session.speakers[0].match = match

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()

            identity = str(app.screen.query_one("#identity-list", Static).render())

            assert identity.index("丰禾") < identity.index("华璟") < identity.index("墨泪")
            assert "score 0.910" in identity
            assert "score 0.720" in identity
            assert "score 0.670" in identity

    asyncio.run(scenario())


def test_speaker_review_tui_arrow_selects_known_person() -> None:
    """Up and down in name edit should move the highlighted person list row."""
    app = SpeakerReviewApp(
        _session(
            people=(
                KnownPerson(42, "欧丁", "vpp-0000000000000042"),
                KnownPerson(7, "敬悦", "vpp-0000000000000007"),
            )
        )
    )
    app.session.speakers[0].match = SpeakerMatchCandidate(
        "欧丁",
        0.95,
        True,
        best_name="欧丁",
        best_score=0.95,
        best_person_id=42,
        candidates=(SpeakerMatchPerson(42, "欧丁", 0.95), SpeakerMatchPerson(7, "敬悦", 0.80)),
    )

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value == SpeakerReviewDecision(
        saved=True,
        mapping={0: "敬悦"},
        person_mapping={0: 7},
        person_public_mapping={0: "vpp-0000000000000007"},
    )


def test_speaker_review_tui_requires_explicit_new_person(tmp_path: Path) -> None:
    """A new person must be created through the explicit +Name TUI flow."""
    store_dir = tmp_path / "voiceprints"
    app = SpeakerReviewApp(_session(store_dir=store_dir))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            field = app.screen.query_one("#identity-search", Input)
            field.value = "新同学"
            await pilot.press("enter")

            assert app._speaker().current_name == "Speaker A"
            assert "Unknown person" in str(app.screen.query_one("#identity-status", Static).render())

            field.value = "+新同学"
            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    person_id = next(iter(app.return_value.person_mapping.values()))
    assert app.return_value.mapping == {0: "新同学"}
    assert person_id > 0


def test_speaker_only_tui_does_not_launch_transcript_correction() -> None:
    """Speaker-only review should keep correction at the project review layer."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            await pilot.press("c")

            assert pilot.app.return_value is None
            assert "project review" in str(pilot.app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_speaker_review_tui_recomputes_page_size_after_resize() -> None:
    """The Pilot should verify responsive pagination instead of fixed row counts."""
    app = SpeakerReviewApp(_session(many_samples=True))

    async def scenario() -> None:
        async with app.run_test(size=(80, 18)) as pilot:
            small_page_size = app._sample_page_size()

            await pilot.resize_terminal(80, 30)
            await pilot.pause()

            large_page_size = app._sample_page_size()
            visible_segments = app._visible_segments(app._speaker())[1]

            assert large_page_size > small_page_size
            assert len(visible_segments) == min(app._speaker().segment_count, large_page_size)

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


def test_speaker_review_tui_space_stops_running_sample(monkeypatch) -> None:
    """Pressing Space while a sample is playing should stop playback."""
    process = _RunningFakeProcess()
    starts = 0

    monkeypatch.setattr(
        speaker_tui,
        "build_audio_preview_command",
        lambda **kwargs: ["fake-player"],
    )

    def fake_popen(*args, **kwargs) -> _RunningFakeProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(speaker_tui.subprocess, "Popen", fake_popen)

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            await pilot.press("space")

            assert starts == 1
            assert pilot.app.playback_process is process

            await pilot.press("space")

            assert starts == 1
            assert process.terminated is True
            assert pilot.app.playback_process is None

    asyncio.run(scenario())


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


def test_load_speaker_review_session_builds_project_overview_from_disk(tmp_path: Path) -> None:
    """Session loading should combine project files, match files, and voiceprint DB state."""
    project_dir, store_dir = _project_with_voiceprint_state(tmp_path)

    session = load_speaker_review_session(project_dir, store_dir=store_dir)
    overview = session.overview

    assert overview.project_id == "20260429-tui-test"
    assert overview.title == "TUI Test"
    assert overview.duration_ms == 3500
    assert overview.match_file_exists is True
    assert overview.saved_names_by_speaker == {0: "欧丁", 1: "Speaker B"}
    assert overview.voiceprint.captured_names_by_speaker == {0: frozenset({"欧丁"})}
    assert len(overview.voiceprint.captured_sample_ids) == 1
    assert overview.voiceprint.embedded_sample_ids == overview.voiceprint.captured_sample_ids
    assert session.people_names == ["欧丁"]
    assert session.people[0].person_id == session.speakers[0].person_id
    assert [speaker.current_name for speaker in session.speakers] == ["欧丁", "Speaker B"]
    assert [speaker.ignored for speaker in session.speakers] == [False, True]


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


def _session(
    *,
    page_size: int | None = None,
    two_speakers: bool = False,
    with_status: bool = False,
    many_samples: bool = False,
    allow_correction: bool = False,
    people: tuple[KnownPerson, ...] = (),
    store_dir: Path | None = None,
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
    if many_samples:
        for sentence_id in range(3, 13):
            segments.append(
                SentenceSegment(
                    begin_time_ms=sentence_id * 1000,
                    end_time_ms=sentence_id * 1000 + 500,
                    text=f"第 {sentence_id} 句",
                    speaker_id=0,
                    sentence_id=sentence_id,
                )
            )
    match = SpeakerMatchCandidate("欧丁", 0.95, True) if with_status else None
    current_name = "别人" if with_status else "Speaker A"
    speakers = [ReviewSpeaker(0, "Speaker A", segments, current_name, match)]
    if two_speakers:
        speakers.append(ReviewSpeaker(1, "Speaker B", segments, "Speaker B", None))
    if with_status:
        speakers.append(
            ReviewSpeaker(
                1,
                "Speaker B",
                segments,
                "欧丁",
                SpeakerMatchCandidate("欧丁", 0.8, True),
            )
        )
    return SpeakerReviewSession(
        project_dir=Path("."),
        source_media=Path("source.mp4"),
        overview=_overview(with_status=with_status),
        speakers=speakers,
        people_names=[person.name for person in people],
        page_size=page_size,
        allow_correction=allow_correction,
        people=people,
        store_dir=store_dir,
    )


def _overview(*, with_status: bool) -> SpeakerReviewOverview:
    """Build a minimal project overview."""
    saved_names = {0: "别人", 1: "欧丁"} if with_status else {}
    voiceprint = VoiceprintReviewProgress(
        captured_names_by_speaker={1: frozenset({"欧丁"})} if with_status else {},
        captured_sample_ids=frozenset({101, 102}) if with_status else frozenset(),
        embed_model="test-model",
        embedded_sample_ids=frozenset({102}) if with_status else frozenset(),
    )
    return SpeakerReviewOverview(
        project_id="project-1",
        title="Demo",
        project_status="named",
        source_name="source.mp4",
        duration_ms=2500,
        match_file_exists=with_status,
        saved_names_by_speaker=saved_names,
        voiceprint=voiceprint,
    )


def _correction_summary(*, accepted: bool, diff_path: Path | None = None) -> CorrectionEditSummary:
    """Build a minimal correction summary for save modal tests."""
    return CorrectionEditSummary(
        review_path=Path("review.md"),
        proposal_path=Path("proposal.md"),
        proposal_diff_path=diff_path or Path("proposal.diff"),
        proposal_json_path=Path("proposal.json"),
        change_count=1 if accepted else 0,
        sample_change_count=1,
        proposed_change_count=1,
        learned_count=1 if accepted else 0,
        accepted=accepted,
        model="test-model",
        model_error=None,
        understanding=[],
        corrected_sentences_path=Path("sentences_corrected.json") if accepted else None,
        corrected_transcript_path=Path("transcript_corrected.txt") if accepted else None,
        corrected_named_transcript_path=Path("transcript_named_corrected.txt") if accepted else None,
        corrected_srt_path=Path("subtitle_named_corrected.srt") if accepted else None,
        hotwords_path=Path("asr_hotwords.json") if accepted else None,
        applied_path=Path("applied.json") if accepted else None,
        lexicon_db=Path("lexicon.sqlite"),
    )


def _style_for_text(text: object, needle: str) -> str | None:
    """Return the first Rich style that covers a substring."""
    plain = text.plain
    start = plain.index(needle)
    end = start + len(needle)
    for span in text.spans:
        if span.start <= start and span.end >= end:
            return str(span.style)
    return None


def _project_with_voiceprint_state(tmp_path: Path) -> tuple[Path, Path]:
    """Build a project fixture with match, manual map, capture, and embedding state."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"media")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="TUI Test",
        projects_dir=None,
        project_dir=project_dir,
        meeting_time="2026-04-29T10:00:00+08:00",
        hash_source=False,
    )
    _force_project_identity(project_dir)
    _write_project_review_files(project_dir)
    store_dir = tmp_path / "voiceprints"
    sample_id, person_id = _store_project_voiceprint(project_dir, store_dir)
    (project_dir / "speakers" / "speaker_person_map.json").write_text(
        json.dumps({"0": person_id}, ensure_ascii=False),
        encoding="utf-8",
    )
    upsert_voiceprint_embedding(sample_id, LOCAL_SPEECHBRAIN_MODEL, [0.1, 0.2], get_voiceprint_db_path(store_dir))
    return project_dir, store_dir


def _force_project_identity(project_dir: Path) -> None:
    """Make the project id stable for status assertions."""
    manifest_path = project_dir / "project.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["project_id"] = "20260429-tui-test"
    payload["title"] = "TUI Test"
    payload["status"] = "named"
    payload["source"]["filename"] = "meeting.mp4"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_project_review_files(project_dir: Path) -> None:
    """Write transcript, map, and match fixtures for session loading."""
    paths = project_paths(project_dir)
    paths.asr_dir.mkdir(parents=True, exist_ok=True)
    paths.speakers_dir.mkdir(parents=True, exist_ok=True)
    transcript = {
        "full_text": "你好。收到。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1500,
                "text": "你好，我是欧丁。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2500,
                "end_time_ms": 3500,
                "text": "收到。",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }
    paths.asr_dir.joinpath("sentences.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    paths.speakers_dir.joinpath("speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "Speaker B"}, ensure_ascii=False),
        encoding="utf-8",
    )
    matches = {
        "matches": [
            {"speaker_id": 0, "name": "欧丁", "score": 0.91, "accepted": True},
            {"speaker_id": 1, "name": "unknown", "score": 0.0, "accepted": False},
        ]
    }
    paths.speakers_dir.joinpath("speaker_matches.json").write_text(
        json.dumps(matches, ensure_ascii=False),
        encoding="utf-8",
    )


def _store_project_voiceprint(project_dir: Path, store_dir: Path) -> tuple[int, int]:
    """Store one voiceprint sample for the project fixture and return sample and person ids."""
    clip_path = store_dir / "clips" / "clip_001.wav"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"wav")
    sample = StoredVoiceprintSample(
        speaker_name="欧丁",
        project_id="20260429-tui-test",
        project_path=project_dir,
        project_speaker_id=0,
        source_path=project_dir / "source" / "meeting.mp4",
        clip_path=clip_path,
        clip_rel_path="clips/clip_001.wav",
        source_begin_time_ms=0,
        source_end_time_ms=1500,
        clip_begin_time_ms=0,
        clip_end_time_ms=1500,
        transcript_text="你好，我是欧丁。",
    )
    db_path = store_voiceprint_samples([sample], get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples_for_project("20260429-tui-test", db_path)
    return rows[0].sample_id, rows[0].speaker_id
