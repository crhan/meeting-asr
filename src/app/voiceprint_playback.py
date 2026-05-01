"""Playback helpers for stored voiceprint clips."""

from __future__ import annotations

import shutil
from pathlib import Path


def build_voiceprint_play_command(path: Path) -> list[str]:
    """
    Build a local playback command for one voiceprint clip.

    Args:
        path: Voiceprint WAV path.

    Returns:
        Command suitable for ``subprocess``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Voiceprint clip does not exist: {path}")
    player = shutil.which("afplay")
    if player:
        return [player, str(path)]
    return [shutil.which("open") or "open", str(path)]
