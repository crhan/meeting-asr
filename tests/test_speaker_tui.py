"""Tests for the speaker review TUI behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path

from textual.widgets import Input, Static, TextArea

from app import speaker_tui
from app.presentation.tui import voiceprint_review_workflow
from app.correction_types import CorrectionEditSummary
from app.core.project_models import ProjectListItem
from app.models import SentenceSegment
from app.presentation.cli.i18n import configure_cli_language
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
    SpeakerRematchProcessingScreen,
    SpeakerRematchResult,
    ShortcutHelpScreen,
    UNFOCUSED_PANE_CLASS,
    VoiceprintReviewProgress,
    load_speaker_review_session,
)
from app.presentation.tui.diff_render import styled_unified_diff
from app.presentation.tui.project import ProjectPickerScreen, ProjectPickerSession
from app.presentation.tui.speaker_save import (
    CorrectionProposalDiffScreen,
    SpeakerReviewSaveOutcome,
    SpeakerReviewSaveScreen,
    _summary_lines,
    speaker_ignore_changes,
    speaker_name_changes,
)
from app.presentation.tui.voiceprint_capture import load_voiceprint_capture_review_session
from app.presentation.tui.voiceprint import VoiceprintLibrarySession
from app.presentation.tui.voiceprint_review import (
    VoiceprintReviewHelpScreen,
    VoiceprintReviewResultScreen,
    VoiceprintReviewScreen,
    VoiceprintReviewSession,
)
from app.presentation.tui.speaker_matches import SpeakerMatchPerson
from app.voiceprint_evaluation import VoiceprintEvaluationSummary, VoiceprintProjectEvaluation, VoiceprintScoreChange
from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL, VoiceprintEmbedSummary
from app.voiceprint_quality import analyze_voiceprint_quality
from app.speaker_labeling import (
    SentenceReassignmentSpec,
    apply_sentence_reassignments,
)
from app.speaker_matching import SpeakerMatch, SpeakerMatchSummary
from app.presentation.tui.speaker_timeline import (
    SpeakerPickScreen,
)
from app.presentation.tui.speaker_save import (
    SentenceReassignmentChange,
    sentence_reassignment_changes,
)
from app.presentation.tui.speaker_models import SentenceReassignment
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)
from app.voiceprints import VoiceprintCaptureSummary, VoiceprintClip, VoiceprintSpeaker


def test_speaker_review_tui_starts_in_browse_mode() -> None:
    """The TUI should open identity editing as a modal."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            main = pilot.app.query_one("#main")
            assert len(list(main.children)) == 2
            assert pilot.app.focused is None
            assert "v capture" in str(pilot.app.query_one("#status", Static).render())

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

            assert "PROJECT REVIEW" in overview
            assert "v: capture voiceprints" in overview
            assert "b: embed" in overview
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


def test_speaker_review_tui_uses_chinese_language() -> None:
    """Speaker review overview should localize visible workflow guidance."""
    try:
        configure_cli_language("zh")
        app = SpeakerReviewApp(_session(with_status=True))
        overview = app._overview_pane()

        assert "p: 切项目" in overview
        assert "[b]项目[/b]" in overview
        assert "[b]步骤[/b]" in overview
        assert "[b]自动[/b]" in overview
        assert "分数平均" in overview
    finally:
        configure_cli_language("en")


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
        ignored_speaker_ids=(0,),
    )


def test_speaker_review_save_diff_separates_ignore_from_name_change() -> None:
    """Ignoring a speaker is a review-state change, not a speaker rename."""
    speaker = ReviewSpeaker(0, "Speaker A", [], "Speaker A", None, ignored=True)

    assert speaker_name_changes([speaker], {}) == ()
    ignore_changes = speaker_ignore_changes([speaker], frozenset())

    assert len(ignore_changes) == 1
    assert ignore_changes[0].label == "Speaker A"
    assert ignore_changes[0].before is False
    assert ignore_changes[0].after is True


def test_project_review_tui_edits_transcript_text_inline() -> None:
    """Project review should edit selected transcript text without leaving the TUI."""
    app = SpeakerReviewApp(_session(allow_correction=True))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("e")
            await pilot.pause()

            assert isinstance(app.screen, SentenceCorrectionScreen)

            field = app.screen.query_one("#correction-input", TextArea)
            field.text = "第一句修正"
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, CorrectionQueuedScreen)
            feedback = app.screen.query_one("#queued-diff", Static).render()
            assert "staged" in feedback.plain
            assert "Before: 第一句" in feedback.plain
            assert "After:  第一句修正" in feedback.plain
            assert any("green" in str(span.style) and "bold" in str(span.style) for span in feedback.spans)
            assert any("red" in str(span.style) and "bold" in str(span.style) for span in feedback.spans)
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
            app.screen.query_one("#correction-input", TextArea).text = "第一句修正"
            await pilot.press("enter")
            await pilot.press("enter")

            await pilot.press("right")
            await pilot.press("down")
            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#correction-input", TextArea).text = "第二句修正"
            await pilot.press("enter")
            await pilot.press("s")

    asyncio.run(scenario())

    assert app.return_value is not None
    assert app.return_value.action == "correct-inline"
    assert [edit.original_text for edit in app.return_value.correction_edits] == ["第一句", "第二句"]
    assert [edit.corrected_text for edit in app.return_value.correction_edits] == ["第一句修正", "第二句修正"]


def test_project_review_tui_save_handler_keeps_tui_open(monkeypatch, tmp_path: Path) -> None:
    """Project review save should run inside a modal instead of exiting the TUI."""
    seen: list[SpeakerReviewDecision] = []
    library = VoiceprintLibrarySession(db_path=tmp_path / "voiceprints.sqlite", speakers=[])

    monkeypatch.setattr(
        speaker_tui,
        "load_voiceprint_review_session",
        lambda **kwargs: (
            VoiceprintReviewSession(
                capture=None,
                library=library,
                quality=analyze_voiceprint_quality(store_dir=tmp_path / "voiceprints"),
                store_dir=tmp_path / "voiceprints",
                return_hint=kwargs["return_hint"],
            ),
            None,
        ),
    )

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
            save_body = str(app.screen.query_one("#save-body", Static).render())
            assert "Speaker A" in save_body
            assert "<not saved> -> Speaker A" in save_body
            assert "Speaker outputs" not in save_body
            assert "Speaker 产物" not in save_body
            assert "speaker_map.json" not in save_body
            assert "v capture voiceprints" in str(app.screen.query_one("#save-actions", Static).render())

            await pilot.press("v")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, VoiceprintReviewScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, VoiceprintReviewScreen)

    asyncio.run(scenario())


def test_project_review_tui_embeds_captured_voiceprints(monkeypatch, tmp_path: Path) -> None:
    """Project Review should run voiceprint embedding without leaving the TUI."""
    store_dir = tmp_path / "voiceprints"
    embedded: list[Path | None] = []

    def fake_embed_voiceprint_samples(**kwargs) -> VoiceprintEmbedSummary:
        embedded.append(kwargs["store_dir"])
        return VoiceprintEmbedSummary(
            db_path=tmp_path / "voiceprints.sqlite",
            provider="local-speechbrain",
            model="test-model",
            embedded_count=2,
            skipped_count=1,
        )

    monkeypatch.setattr(speaker_tui, "embed_voiceprint_samples", fake_embed_voiceprint_samples)
    app = SpeakerReviewApp(_session(with_status=True, store_dir=store_dir))

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("b")
            await pilot.pause()
            await pilot.pause()

            assert embedded == [store_dir]
            assert "embedded 2, skipped 1" in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_project_review_tui_accepts_pending_correction_in_modal(tmp_path: Path) -> None:
    """The save modal should handle proposal acceptance without leaving the TUI."""
    accepted_paths: list[Path | None] = []
    diff_path = tmp_path / "proposal.diff"
    proposal_path = tmp_path / "proposal.json"
    diff_path.write_text("- AS\n+ IaaS\n", encoding="utf-8")
    proposal_path.write_text(
        json.dumps({"proposed_changes": [_proposal_change(1, "AS", "IaaS")]}),
        encoding="utf-8",
    )

    def save_handler(decision: SpeakerReviewDecision) -> SpeakerReviewSaveOutcome:
        assert len(decision.correction_edits) == 1
        return SpeakerReviewSaveOutcome(
            Path("speaker_map.json"),
            Path("transcript.txt"),
            Path("subtitle.srt"),
            _correction_summary(accepted=False, diff_path=diff_path, proposal_path=proposal_path),
        )

    def accept_handler(
        proposal_path: Path | None,
        selected_indices: tuple[int, ...] | None,
    ) -> SpeakerReviewSaveOutcome:
        accepted_paths.append(proposal_path)
        assert selected_indices == (0,)
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
            app.screen.query_one("#correction-input", TextArea).text = "第一句修正"
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
            assert "[x] Change 1/1" in diff_render.plain
            styled_diff = styled_unified_diff("- AS\n+ IaaS\n")
            assert any(str(span.style) == "bold green" for span in styled_diff.spans)
            assert any(str(span.style) == "bold red" for span in styled_diff.spans)

            await pilot.press("escape")
            await pilot.pause()

            await pilot.press("a")
            await pilot.pause()
            await pilot.pause()

            assert accepted_paths == [proposal_path]
            assert not app.correction_edits
            assert "correction accepted" in str(app.screen.query_one("#save-title", Static).render())

    asyncio.run(scenario())


def test_project_review_tui_can_exclude_one_proposed_change(tmp_path: Path) -> None:
    """Proposal review should pass only selected changes to acceptance."""
    accepted_indices: list[tuple[int, ...] | None] = []
    diff_path = tmp_path / "proposal.diff"
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(_proposal_payload()), encoding="utf-8")
    diff_path.write_text("- IC\n+ isee\n- AS\n+ IaaS\n", encoding="utf-8")

    def save_handler(decision: SpeakerReviewDecision) -> SpeakerReviewSaveOutcome:
        return SpeakerReviewSaveOutcome(
            Path("speaker_map.json"),
            Path("transcript.txt"),
            Path("subtitle.srt"),
            _correction_summary(accepted=False, diff_path=diff_path, proposal_path=proposal_path),
        )

    def accept_handler(
        proposal_path: Path | None,
        selected_indices: tuple[int, ...] | None,
    ) -> SpeakerReviewSaveOutcome:
        accepted_indices.append(selected_indices)
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
            app.screen.query_one("#correction-input", TextArea).text = "第一句修正"
            await pilot.press("enter")
            await pilot.press("s")
            await pilot.pause()
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("x")
            await pilot.press("a")
            await pilot.pause()
            await pilot.pause()

            assert accepted_indices == [(0,)]

    asyncio.run(scenario())


def test_correction_diff_viewer_uses_standard_vertical_keys(tmp_path: Path) -> None:
    """Proposal diff selection should use up/down and j/k like the rest of the TUI."""
    proposal_path = tmp_path / "proposal.json"
    diff_path = tmp_path / "proposal.diff"
    proposal_path.write_text(json.dumps(_proposal_payload()), encoding="utf-8")
    diff_path.write_text("- IC\n+ isee\n- AS\n+ IaaS\n", encoding="utf-8")
    screen = CorrectionProposalDiffScreen(
        diff_path=diff_path,
        proposal_path=proposal_path,
        selected_indices=None,
    )

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test() as pilot:
            pilot.app.push_screen(screen)
            await pilot.pause()

            assert screen.current_change_index == 0
            assert "up/down" in str(screen.query_one("#diff-actions", Static).render())
            assert "Esc returns" in str(screen.query_one("#diff-actions", Static).render())
            assert "Enter returns" not in str(screen.query_one("#diff-actions", Static).render())
            assert "n/p" not in str(screen.query_one("#diff-actions", Static).render())

            await pilot.press("down")
            await pilot.pause()

            assert screen.current_change_index == 1

            await pilot.press("k")
            await pilot.pause()

            assert screen.current_change_index == 0

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(pilot.app.screen, CorrectionProposalDiffScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, CorrectionProposalDiffScreen)

    asyncio.run(scenario())


def test_correction_diff_viewer_highlights_changed_tokens_only() -> None:
    """Diff rendering should highlight changed tokens, not just whole lines."""
    styled_diff = styled_unified_diff("- 我们看一下IC系统。\n+ 我们看一下isee系统。\n")

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

            field = app.screen.query_one("#correction-input", TextArea)
            field.text = "abcdef"
            field.cursor_location = (0, 0)

            await pilot.press("ctrl+f")
            await pilot.pause()

            assert field.text == "abcdef"
            assert field.cursor_location == (0, 1)

            await pilot.press("ctrl+b")
            await pilot.pause()

            assert field.text == "abcdef"
            assert field.cursor_location == (0, 0)

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


def test_project_review_tui_opens_embedded_voiceprint_review(monkeypatch, tmp_path: Path) -> None:
    """Project review should open the shared voiceprint screen without leaving the TUI."""
    library = VoiceprintLibrarySession(db_path=tmp_path / "voiceprints.sqlite", speakers=[])

    monkeypatch.setattr(
        speaker_tui,
        "load_voiceprint_review_session",
        lambda **kwargs: (
            VoiceprintReviewSession(
                capture=None,
                library=library,
                quality=analyze_voiceprint_quality(store_dir=tmp_path / "voiceprints"),
                store_dir=tmp_path / "voiceprints",
                return_hint=kwargs["return_hint"],
            ),
            None,
        ),
    )

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(with_status=True)).run_test(size=(120, 24)) as pilot:
            await pilot.press("v")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewScreen)
            assert "return to Project Review" in pilot.app.screen._overview_pane()

            await pilot.press("?")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewHelpScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, VoiceprintReviewScreen)

    asyncio.run(scenario())


def test_project_review_voiceprint_screen_saves_embeds_and_evaluates(monkeypatch, tmp_path: Path) -> None:
    """Saving in embedded Voiceprint Review should keep capture, embed, and eval in that screen."""
    store_dir = tmp_path / "voiceprints"
    session = _voiceprint_review_session(tmp_path, return_hint="return to Project Review")
    planned = _voiceprint_capture_plan(store_dir)
    completed: list[int] = []
    calls: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def fake_capture(project_dir, **kwargs) -> VoiceprintCaptureSummary:
        calls.append("capture")
        assert kwargs["planned"] is planned
        started.set()
        release.wait(timeout=2)
        return VoiceprintCaptureSummary(
            store_dir=store_dir,
            db_path=get_voiceprint_db_path(store_dir),
            clip_dir=store_dir / "clips",
            speakers=planned.speakers if planned is not None else [],
            dry_run=False,
        )

    def fake_embed(**kwargs) -> VoiceprintEmbedSummary:
        calls.append("embed")
        return VoiceprintEmbedSummary(get_voiceprint_db_path(store_dir), "local-speechbrain", "test-model", 2, 1)

    def fake_evaluate(project_dir, **kwargs) -> VoiceprintEvaluationSummary:
        calls.append("evaluate")
        return _evaluation_summary(tmp_path)

    monkeypatch.setattr(
        speaker_tui,
        "load_voiceprint_review_session",
        lambda **kwargs: (session, planned),
    )
    monkeypatch.setattr(voiceprint_review_workflow, "persist_voiceprint_capture_selection", fake_capture)
    monkeypatch.setattr(voiceprint_review_workflow, "embed_voiceprint_samples", fake_embed)
    monkeypatch.setattr(voiceprint_review_workflow, "evaluate_voiceprint_embedding", fake_evaluate)

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(with_status=True, store_dir=store_dir)).run_test(size=(120, 24)) as pilot:
            await pilot.press("v")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewScreen)

            await pilot.press("s")
            for _ in range(10):
                await pilot.pause()
                if started.is_set():
                    break

            assert isinstance(pilot.app.screen, voiceprint_review_workflow.VoiceprintReviewProcessingScreen)
            release.set()

            for _ in range(10):
                await pilot.pause()
                if isinstance(pilot.app.screen, VoiceprintReviewResultScreen):
                    break

            assert calls == ["capture", "embed", "evaluate"]
            assert isinstance(pilot.app.screen, VoiceprintReviewResultScreen)

            await pilot.press("a")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewScreen)
            assert "historical risks 0" in str(pilot.app.screen.query_one("#status", Static).render())
            completed.append(1)

    asyncio.run(scenario())
    assert completed == [1]


def test_project_review_tui_voiceprint_tab_shows_current_view(monkeypatch, tmp_path: Path) -> None:
    """Embedded voiceprint review should clearly show the active sub-view."""
    session = _voiceprint_review_session(tmp_path, return_hint="return to Project Review")

    monkeypatch.setattr(
        speaker_tui,
        "load_voiceprint_review_session",
        lambda **kwargs: (session, None),
    )

    async def scenario() -> None:
        async with SpeakerReviewApp(_session(with_status=True)).run_test(size=(120, 24)) as pilot:
            await pilot.press("v")
            await pilot.pause()

            assert isinstance(pilot.app.screen, VoiceprintReviewScreen)
            assert "view=[bold cyan]Project candidates" in pilot.app.screen._overview_pane()
            assert "Tab -> Global library" in pilot.app.screen._overview_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert "view=[bold cyan]Global library" in pilot.app.screen._overview_pane()
            assert "Tab -> Quality review" in pilot.app.screen._overview_pane()

            await pilot.press("tab")
            await pilot.pause()

            assert "view=[bold cyan]Quality review" in pilot.app.screen._overview_pane()
            assert "Tab -> Project candidates" in pilot.app.screen._overview_pane()

    asyncio.run(scenario())


def test_project_review_tui_requires_saved_names_before_voiceprint() -> None:
    """Voiceprint capture from project review should not use stale unsaved speaker names."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_session()).run_test(size=(120, 24)) as pilot:
            await pilot.press("v")
            await pilot.pause()

            assert not isinstance(pilot.app.screen, VoiceprintReviewScreen)
            assert "Save speaker names" in str(pilot.app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_project_review_tui_switches_projects_and_saves_active_project(monkeypatch, tmp_path: Path) -> None:
    """Project switching should reload review state and bind save to the active project."""
    current_dir = (tmp_path / "projects" / "p-current").resolve()
    target_dir = (tmp_path / "projects" / "p-target").resolve()
    current_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    current_session = _session_for_project(current_dir, project_id="p-current", title="Current")
    target_session = _session_for_project(target_dir, project_id="p-target", title="Target")
    picker_session = ProjectPickerSession(
        projects_dir=current_dir.parent,
        projects=[
            _project_list_item(current_dir, "p-current", "Current"),
            _project_list_item(target_dir, "p-target", "Target"),
        ],
    )
    saved_projects: list[Path] = []

    monkeypatch.setattr(speaker_tui, "load_project_picker_session", lambda projects_dir: picker_session)
    monkeypatch.setattr(
        speaker_tui,
        "load_speaker_review_session",
        lambda project_dir, **kwargs: target_session if Path(project_dir).resolve() == target_dir else current_session,
    )

    def save_active_project(project_dir: Path, decision: SpeakerReviewDecision) -> SpeakerReviewSaveOutcome:
        saved_projects.append(project_dir)
        assert decision.project_dir == target_dir
        return SpeakerReviewSaveOutcome(Path("speaker_map.json"), Path("transcript.txt"), Path("subtitle.srt"))

    app = SpeakerReviewApp(current_session, project_save_handler=save_active_project)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("p")
            await pilot.pause()

            assert isinstance(app.screen, ProjectPickerScreen)

            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, ProjectPickerScreen)
            assert app.session.project_dir == target_dir
            assert "p-target" in app._overview_pane()

            await pilot.press("s")
            await pilot.pause()
            await pilot.pause()

            assert isinstance(app.screen, SpeakerReviewSaveScreen)

    asyncio.run(scenario())

    assert saved_projects == [target_dir]


def test_project_review_tui_blocks_project_switch_with_unsaved_changes() -> None:
    """Project switching should not silently discard edited speaker names or text."""
    app = SpeakerReviewApp(_session())

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            app.session.speakers[0].current_name = "欧丁"
            await pilot.press("p")
            await pilot.pause()

            assert not isinstance(app.screen, ProjectPickerScreen)
            assert "Save current project changes" in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_project_review_tui_rematches_speakers_and_refreshes(monkeypatch, tmp_path: Path) -> None:
    """Project Review should rerun voiceprint matching and reload visible matches."""
    current_dir = (tmp_path / "projects" / "p-current").resolve()
    current_dir.mkdir(parents=True)
    current_session = _session_for_project(current_dir, project_id="p-current", title="Current")
    target_session = replace(
        current_session,
        speakers=[
            ReviewSpeaker(
                0,
                "Speaker A",
                current_session.speakers[0].segments,
                "墨泪",
                SpeakerMatchCandidate("墨泪", 0.91, True, best_name="墨泪", best_score=0.91),
            ),
            current_session.speakers[1],
        ],
        overview=replace(current_session.overview, match_file_exists=True),
    )
    summary = SpeakerMatchSummary(
        current_dir / "speakers" / "speaker_matches.json",
        "local-speechbrain",
        "test-model",
        0.75,
        [
            SpeakerMatch(0, "Speaker A", "墨泪", 0.91, True, 2, best_name="墨泪", best_score=0.91, accepted_name="墨泪", threshold=0.75),
            SpeakerMatch(1, "Speaker B", None, 0.64, False, 2, best_name="欧丁", best_score=0.64, threshold=0.75),
        ],
    )
    unblock = threading.Event()
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_rematch(project_dir: Path, **kwargs) -> SpeakerRematchResult:
        calls.append((project_dir, kwargs))
        assert unblock.wait(timeout=5)
        return SpeakerRematchResult(summary, target_session)

    monkeypatch.setattr(speaker_tui, "run_speaker_rematch", fake_rematch)
    app = SpeakerReviewApp(current_session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("m")
            await pilot.pause()

            assert isinstance(app.screen, SpeakerRematchProcessingScreen)

            unblock.set()
            for _ in range(20):
                await pilot.pause()
                if not isinstance(app.screen, SpeakerRematchProcessingScreen):
                    break

            assert app.session == target_session
            assert app._speaker().current_name == "墨泪"
            assert "Rematch complete: matched 1/2" in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())

    assert calls == [
        (
            current_dir,
            {
                "store_dir": current_session.store_dir,
                "page_size": current_session.page_size,
                "allow_correction": current_session.allow_correction,
            },
        )
    ]


def test_project_review_tui_rematch_allows_unpersisted_initial_matches(monkeypatch, tmp_path: Path) -> None:
    """Derived accepted matches should not be treated as unsaved human edits."""
    project_dir = (tmp_path / "projects" / "p-current").resolve()
    project_dir.mkdir(parents=True)
    session = _session_for_project(project_dir, project_id="p-current", title="Current")
    session.speakers[0].current_name = "墨泪"
    session.speakers[0].match = SpeakerMatchCandidate("墨泪", 0.91, True, best_name="墨泪", best_score=0.91)
    session = replace(session, overview=replace(session.overview, saved_names_by_speaker={}))
    summary = SpeakerMatchSummary(
        project_dir / "speakers" / "speaker_matches.json",
        "local-speechbrain",
        "test-model",
        0.75,
        [SpeakerMatch(0, "Speaker A", "墨泪", 0.91, True, 2, best_name="墨泪", best_score=0.91, accepted_name="墨泪", threshold=0.75)],
    )
    monkeypatch.setattr(speaker_tui, "run_speaker_rematch", lambda *args, **kwargs: SpeakerRematchResult(summary, session))
    app = SpeakerReviewApp(session)

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            await pilot.press("m")
            await pilot.pause()
            await pilot.pause()

            assert "Rematch complete" in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_project_review_tui_blocks_rematch_with_unsaved_changes(monkeypatch) -> None:
    """Voiceprint rematch should not discard unsaved human review edits."""
    monkeypatch.setattr(speaker_tui, "run_speaker_rematch", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rematch")))
    app = SpeakerReviewApp(_session())

    async def scenario() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            app.session.speakers[0].current_name = "欧丁"
            await pilot.press("m")
            await pilot.pause()

            assert not isinstance(app.screen, SpeakerRematchProcessingScreen)
            assert "Save current review changes" in str(app.query_one("#status", Static).render())

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


def test_load_speaker_review_session_prefers_corrected_transcript(tmp_path: Path) -> None:
    """Project review should reopen the corrected transcript after correction acceptance."""
    project_dir, store_dir = _project_with_voiceprint_state(tmp_path)
    corrected = json.loads((project_dir / "asr" / "sentences.json").read_text(encoding="utf-8"))
    corrected["sentences"][0]["text"] = "你好，我是修正后的欧丁。"
    corrected["full_text"] = "你好，我是修正后的欧丁。收到。"
    (project_dir / "asr" / "sentences_corrected.json").write_text(
        json.dumps(corrected, ensure_ascii=False),
        encoding="utf-8",
    )

    session = load_speaker_review_session(project_dir, store_dir=store_dir)

    assert session.speakers[0].segments[0].text == "你好，我是修正后的欧丁。"


def test_load_speaker_review_session_keeps_named_low_information_speakers(tmp_path: Path) -> None:
    """Review must show a real attendee even when their track is mostly backchannels."""
    project_dir, store_dir = _project_with_voiceprint_state(tmp_path)
    transcript_path = project_dir / "asr" / "sentences.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript["detected_speakers"].append(2)
    transcript["sentences"].append(
        {
            "begin_time_ms": 3600,
            "end_time_ms": 3900,
            "text": "嗯。",
            "speaker_id": 2,
            "sentence_id": 3,
        }
    )
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "Speaker B", "2": "岳周"}, ensure_ascii=False),
        encoding="utf-8",
    )

    session = load_speaker_review_session(project_dir, store_dir=store_dir)

    assert [speaker.label for speaker in session.speakers] == ["Speaker A", "Speaker B", "Speaker C"]
    assert [speaker.current_name for speaker in session.speakers] == ["欧丁", "Speaker B", "岳周"]


def test_save_summary_labels_empty_correction_as_no_changes() -> None:
    """A zero-change inline edit should not be shown as a ready proposal."""
    lines = _summary_lines(_correction_summary(accepted=False, no_proposal=True))

    assert "- State: no transcript changes" in lines
    assert all("proposal ready" not in line for line in lines)


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


def _session_for_project(project_dir: Path, *, project_id: str, title: str) -> SpeakerReviewSession:
    """Build a saved review session bound to a specific project."""
    overview = replace(_overview(with_status=True), project_id=project_id, title=title)
    return replace(
        _session(with_status=True),
        project_dir=project_dir,
        projects_dir=project_dir.parent,
        overview=overview,
    )


def _project_list_item(project_dir: Path, project_id: str, title: str) -> ProjectListItem:
    """Build one project picker row."""
    return ProjectListItem(
        project_dir=project_dir,
        project_id=project_id,
        title=title,
        meeting_time=None,
        status="named",
        created_at="2026-05-04T10:00:00+08:00",
        updated_at="2026-05-04T10:00:00+08:00",
    )


def _voiceprint_review_session(tmp_path: Path, *, return_hint: str) -> VoiceprintReviewSession:
    """Build a minimal embeddable voiceprint review session."""
    store_dir = tmp_path / "voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    capture = load_voiceprint_capture_review_session(
        summary=_voiceprint_capture_plan(store_dir),
        source_path=source_path,
    )
    return VoiceprintReviewSession(
        capture=capture,
        library=VoiceprintLibrarySession(db_path=get_voiceprint_db_path(store_dir), speakers=[]),
        quality=analyze_voiceprint_quality(store_dir=store_dir),
        store_dir=store_dir,
        return_hint=return_hint,
    )


def _voiceprint_capture_plan(store_dir: Path) -> VoiceprintCaptureSummary:
    """Build a planned voiceprint capture summary."""
    clip = VoiceprintClip(
        path=store_dir / "clips" / "project-1" / "speaker_0" / "clip_001.wav",
        rel_path="clips/project-1/speaker_0/clip_001.wav",
        source_begin_time_ms=1000,
        source_end_time_ms=2000,
        clip_begin_time_ms=1000,
        clip_end_time_ms=2000,
        text="voiceprint candidate",
    )
    return VoiceprintCaptureSummary(
        store_dir=store_dir,
        db_path=get_voiceprint_db_path(store_dir),
        clip_dir=store_dir / "clips",
        speakers=[VoiceprintSpeaker(0, "欧丁", None, None, [clip])],
        dry_run=True,
    )


def _evaluation_summary(tmp_path: Path) -> VoiceprintEvaluationSummary:
    """Build a small voiceprint evaluation fixture."""
    current = VoiceprintProjectEvaluation(
        tmp_path / "projects" / "project-1",
        "project-1",
        "Demo",
        True,
        (
            VoiceprintScoreChange(0, "Speaker A", "欧丁", 0.61, "欧丁", 0.78, 0.17, "improved"),
        ),
    )
    return VoiceprintEvaluationSummary(current, ())


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


def _correction_summary(
    *,
    accepted: bool,
    diff_path: Path | None = None,
    proposal_path: Path | None = None,
    no_proposal: bool = False,
) -> CorrectionEditSummary:
    """Build a minimal correction summary for save modal tests."""
    return CorrectionEditSummary(
        review_path=Path("review.md"),
        proposal_path=None if no_proposal else Path("proposal.md"),
        proposal_diff_path=None if no_proposal else diff_path or Path("proposal.diff"),
        proposal_json_path=None if no_proposal else proposal_path or Path("proposal.json"),
        change_count=1 if accepted else 0,
        sample_change_count=0 if no_proposal else 1,
        proposed_change_count=0 if no_proposal else 1,
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


def _proposal_payload() -> dict:
    """Build a minimal correction proposal payload with two changes."""
    return {
        "proposed_changes": [
            _proposal_change(1, "我们看一下IC系统。", "我们看一下isee系统。"),
            _proposal_change(2, "AS服务需要修正。", "IaaS服务需要修正。"),
        ]
    }


def _proposal_change(sentence_id: int, before: str, after: str) -> dict:
    """Build one proposed change payload."""
    return {
        "sentence_id": sentence_id,
        "speaker_name": "敬悦",
        "original_text": before,
        "corrected_text": after,
    }


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


def _timeline_session() -> SpeakerReviewSession:
    """Build a fresh review session with two speakers and distinct segments."""
    speaker_a_segments = [
        SentenceSegment(
            begin_time_ms=0,
            end_time_ms=1000,
            text="第一句来自惧留孙的发言",
            speaker_id=0,
            sentence_id=1,
        ),
        SentenceSegment(
            begin_time_ms=4000,
            end_time_ms=4500,
            text="第三句也是惧留孙",
            speaker_id=0,
            sentence_id=3,
        ),
    ]
    speaker_b_segments = [
        SentenceSegment(
            begin_time_ms=2000,
            end_time_ms=2500,
            text="第二句其实是欧丁讲的",
            speaker_id=1,
            sentence_id=2,
        ),
    ]
    speakers = [
        ReviewSpeaker(0, "Speaker A", speaker_a_segments, "惧留孙", None),
        ReviewSpeaker(1, "Speaker B", speaker_b_segments, "欧丁", None),
    ]
    return SpeakerReviewSession(
        project_dir=Path("."),
        source_media=Path("source.mp4"),
        overview=_overview(with_status=False),
        speakers=speakers,
        people_names=[],
        page_size=8,
    )


def test_timeline_view_toggle_renders_chronological_order() -> None:
    """Pressing t should swap to the timeline pane and back."""

    async def scenario() -> None:
        async with SpeakerReviewApp(_timeline_session()).run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()

            assert pilot.app.view_mode == "timeline"
            timeline_text = pilot.app.query_one("#timeline", Static).render().plain
            # Sentences should appear in chronological order (id 1, 2, 3).
            first = timeline_text.index("第一句来自惧留孙")
            second = timeline_text.index("第二句其实是欧丁")
            third = timeline_text.index("第三句也是惧留孙")
            assert first < second < third

            # Speaker pane is hidden in timeline view.
            assert pilot.app.query_one("#main").display is False
            assert pilot.app.query_one("#timeline-main").display is True

            await pilot.press("t")
            await pilot.pause()

            assert pilot.app.view_mode == "speakers"
            assert pilot.app.query_one("#main").display is True
            assert pilot.app.query_one("#timeline-main").display is False

    asyncio.run(scenario())


def test_timeline_reassign_moves_segment_and_records_change() -> None:
    """Reassigning a sentence in timeline view should move it between speakers."""

    app = SpeakerReviewApp(_timeline_session())

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            # Move cursor to the second timeline row (sentence_id=2, currently Speaker B).
            await pilot.press("j")
            await pilot.press("r")
            await pilot.pause()

            assert isinstance(pilot.app.screen, SpeakerPickScreen)

            # Picker preselects a non-current speaker; press enter to accept.
            await pilot.press("enter")
            await pilot.pause()

            speaker_a, speaker_b = pilot.app.session.speakers
            assert any(seg.sentence_id == 2 for seg in speaker_a.segments)
            assert all(seg.sentence_id != 2 for seg in speaker_b.segments)
            moved = next(seg for seg in speaker_a.segments if seg.sentence_id == 2)
            assert moved.speaker_id == 0

            decision = pilot.app._decision()
            assert len(decision.sentence_reassignments) == 1
            change = decision.sentence_reassignments[0]
            assert change.sentence_id == 2
            assert change.original_speaker_id == 1
            assert change.new_speaker_id == 0

    asyncio.run(scenario())


def test_speaker_view_reassigns_sample_to_new_speaker() -> None:
    """Grouped Project Review samples should reassign directly without timeline mode."""

    app = SpeakerReviewApp(_timeline_session())

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("r")
            await pilot.pause()

            assert isinstance(pilot.app.screen, SpeakerPickScreen)

            # Existing non-current speaker is preselected; move to the new speaker row.
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            speakers = pilot.app.session.speakers
            assert [speaker.speaker_id for speaker in speakers] == [0, 1, 2]
            assert speakers[2].label == "Speaker C"
            assert speakers[2].current_name == "Speaker C"
            assert any(seg.sentence_id == 1 for seg in speakers[2].segments)
            assert all(seg.sentence_id != 1 for seg in speakers[0].segments)
            assert pilot.app.view_mode == "speakers"
            assert "reassigned" in pilot.app._sample_pane()

            decision = pilot.app._decision()
            assert len(decision.sentence_reassignments) == 1
            change = decision.sentence_reassignments[0]
            assert change.sentence_id == 1
            assert change.original_speaker_id == 0
            assert change.new_speaker_id == 2

    asyncio.run(scenario())


def test_apply_sentence_reassignments_rewrites_persisted_files(tmp_path: Path) -> None:
    """The persistence helper updates raw and corrected sentence files in place."""
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    raw = {
        "full_text": "第一句。第二句。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "第一句",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 2500,
                "text": "第二句",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }
    (asr_dir / "sentences.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    corrected = {
        "full_text": "第一句修正。第二句修正。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "第一句修正",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 2500,
                "text": "第二句修正",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }
    (asr_dir / "sentences_corrected.json").write_text(
        json.dumps(corrected, ensure_ascii=False),
        encoding="utf-8",
    )

    written = apply_sentence_reassignments(
        asr_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=2,
                begin_time_ms=2000,
                end_time_ms=2500,
                new_speaker_id=0,
            )
        ],
    )

    assert {path.name for path in written} == {"sentences.json", "sentences_corrected.json"}
    raw_after = json.loads((asr_dir / "sentences.json").read_text(encoding="utf-8"))
    corrected_after = json.loads((asr_dir / "sentences_corrected.json").read_text(encoding="utf-8"))
    assert raw_after["sentences"][1]["speaker_id"] == 0
    assert corrected_after["sentences"][1]["speaker_id"] == 0
    # detected_speakers should drop the now-unused speaker.
    assert raw_after["detected_speakers"] == [0]


def test_apply_sentence_reassignments_falls_back_to_timing_match(tmp_path: Path) -> None:
    """Sentences without sentence_id should still match by timing."""
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    payload = {
        "sentences": [
            {
                "begin_time_ms": 1000,
                "end_time_ms": 2000,
                "text": "无 id 的句子",
                "speaker_id": 1,
            }
        ]
    }
    (asr_dir / "sentences.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    apply_sentence_reassignments(
        asr_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=None,
                begin_time_ms=1000,
                end_time_ms=2000,
                new_speaker_id=2,
            )
        ],
    )

    after = json.loads((asr_dir / "sentences.json").read_text(encoding="utf-8"))
    assert after["sentences"][0]["speaker_id"] == 2


def test_sentence_reassignment_changes_uses_speaker_names() -> None:
    """The save-modal helper should look up speaker names by current state."""
    speakers = [
        ReviewSpeaker(0, "Speaker A", [
            SentenceSegment(
                begin_time_ms=2000,
                end_time_ms=2500,
                text="第二句",
                speaker_id=0,
                sentence_id=2,
            ),
        ], "惧留孙", None),
        ReviewSpeaker(1, "Speaker B", [], "欧丁", None),
    ]
    reassignments = [
        SentenceReassignment(
            sentence_id=2,
            begin_time_ms=2000,
            end_time_ms=2500,
            original_speaker_id=1,
            new_speaker_id=0,
        )
    ]

    rows = sentence_reassignment_changes(speakers, reassignments)

    assert len(rows) == 1
    assert isinstance(rows[0], SentenceReassignmentChange)
    assert rows[0].before_label == "Speaker B"
    assert rows[0].before_name == "欧丁"
    assert rows[0].after_label == "Speaker A"
    assert rows[0].after_name == "惧留孙"
    assert "第二句" in rows[0].text


def test_timeline_unsaved_changes_block_project_switch() -> None:
    """A pending reassignment must block project switching just like rename edits."""

    app = SpeakerReviewApp(_timeline_session())

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("j")
            await pilot.press("r")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert pilot.app._has_unsaved_review_changes() is True

    asyncio.run(scenario())
