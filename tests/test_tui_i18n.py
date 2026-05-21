"""Tests for TUI language selection."""

from __future__ import annotations

from app.presentation.cli.i18n import configure_cli_language
from app.presentation.tui.speaker_help import browse_status as speaker_browse_status
from app.presentation.tui.speaker_help import shortcut_help as speaker_shortcut_help
from app.presentation.tui.voiceprint_review import (
    help_text as voiceprint_review_help_text,
)
from app.presentation.tui.voiceprint_review import (
    status_text as voiceprint_review_status_text,
)


def test_tui_text_follows_explicit_cli_language() -> None:
    """TUI text should follow the same configured language as CLI help."""
    try:
        configure_cli_language("zh")

        assert "浏览" in speaker_browse_status()
        assert "Speaker Review 快捷键" in speaker_shortcut_help()
        assert "声纹" in voiceprint_review_status_text()
        assert "Voiceprint Review 快捷键" in voiceprint_review_help_text()

        configure_cli_language("en")

        assert "Browse" in speaker_browse_status()
        assert "Speaker Review Shortcuts" in speaker_shortcut_help()
        assert "Voiceprint:" in voiceprint_review_status_text()
        assert "Voiceprint Review Shortcuts" in voiceprint_review_help_text()
    finally:
        configure_cli_language("en")


def test_tui_text_follows_locale_and_env_override(monkeypatch) -> None:
    """TUI language should use locale by default and MEETING_ASR_LANG as override."""
    try:
        monkeypatch.delenv("MEETING_ASR_LANG", raising=False)
        monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
        configure_cli_language(None)

        assert "浏览" in speaker_browse_status()

        monkeypatch.setenv("MEETING_ASR_LANG", "en")
        configure_cli_language(None)

        assert "Browse" in speaker_browse_status()
    finally:
        configure_cli_language("en")
