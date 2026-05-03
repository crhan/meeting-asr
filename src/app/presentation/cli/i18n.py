"""Small CLI i18n helpers for Meeting-ASR-owned help text."""

from __future__ import annotations

import os

SUPPORTED_LANGUAGES = ("en", "zh")
_CURRENT_LANGUAGE = "en"


def configure_cli_language(value: str | None = None) -> str:
    """
    Configure process-wide CLI language.

    Args:
        value: Explicit language value. Supports ``auto``, ``en``, and ``zh``.

    Returns:
        Resolved language code.
    """
    global _CURRENT_LANGUAGE
    _CURRENT_LANGUAGE = resolve_cli_language(value)
    return _CURRENT_LANGUAGE


def current_cli_language() -> str:
    """
    Return the configured CLI language.

    Returns:
        ``en`` or ``zh``.
    """
    return _CURRENT_LANGUAGE


def resolve_cli_language(value: str | None = None) -> str:
    """
    Resolve a user-facing language value.

    Args:
        value: Explicit language value, environment fallback, or ``auto``.

    Returns:
        ``en`` or ``zh``.
    """
    raw = value or os.environ.get("MEETING_ASR_LANG") or "auto"
    normalized = raw.strip().lower().replace("_", "-")
    if normalized == "auto":
        normalized = _locale_language()
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    raise ValueError("CLI language must be one of: auto, en, zh.")


def _locale_language() -> str:
    """Resolve language from standard locale environment variables."""
    for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(key, "").strip().lower().replace("_", "-")
        if value.startswith("zh"):
            return "zh"
    return "en"
