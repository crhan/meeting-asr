"""Runtime i18n helpers for Textual UI strings."""

from __future__ import annotations

from app.presentation.cli.i18n import current_cli_language


def tui_lang() -> str:
    """Return the active TUI language.

    Returns:
        ``en`` or ``zh``, matching the CLI language resolver.
    """
    return current_cli_language()


def tr(en: str, zh: str) -> str:
    """Choose an English or Chinese UI string at render time.

    Args:
        en: English text.
        zh: Chinese text.

    Returns:
        Text for the current UI language.
    """
    return zh if tui_lang() == "zh" else en
