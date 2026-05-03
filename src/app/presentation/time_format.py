"""Human timestamp formatting helpers."""

from __future__ import annotations

from datetime import datetime, tzinfo


def format_local_minute(value: str, *, timezone: tzinfo | None = None) -> str:
    """
    Format an ISO timestamp in the user's local timezone.

    Args:
        value: ISO timestamp, usually from a project manifest.
        timezone: Optional target timezone for tests.

    Returns:
        Compact local ``YYYY-MM-DD HH:MM`` timestamp.
    """
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return value[:16].replace("T", " ")
    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M")
    local = parsed.astimezone(timezone) if timezone is not None else parsed.astimezone()
    return local.strftime("%Y-%m-%d %H:%M")


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO datetime, including trailing-Z UTC values."""
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
