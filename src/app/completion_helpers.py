"""Reusable shell completion value providers."""

from __future__ import annotations

from app.config import CONFIG_KEYS
from app.ffmpeg_utils import SUPPORTED_AUDIO_FORMATS

ASR_MODELS = ("fun-asr",)
OSS_UPLOAD_MODES = ("auto", "true", "false")


def complete_audio_format(incomplete: str) -> list[str]:
    """
    Complete supported local audio output formats.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching audio format names.
    """
    return _matching(sorted(SUPPORTED_AUDIO_FORMATS), incomplete)


def complete_config_key(incomplete: str) -> list[str]:
    """
    Complete supported config key spellings.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching public, internal, and environment config names.
    """
    values: list[str] = []
    for config_key in CONFIG_KEYS:
        values.extend([config_key.name, config_key.field_name, config_key.env_name])
    return _matching(values, incomplete)


def complete_model(incomplete: str) -> list[str]:
    """
    Complete supported ASR model names.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching model names.
    """
    return _matching(ASR_MODELS, incomplete)


def complete_oss_upload_mode(incomplete: str) -> list[str]:
    """
    Complete OSS upload mode values.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching upload modes.
    """
    return _matching(OSS_UPLOAD_MODES, incomplete)


def _matching(values: list[str] | tuple[str, ...], incomplete: str) -> list[str]:
    """
    Filter completion candidates case-insensitively.

    Args:
        values: Candidate values.
        incomplete: Current shell token.

    Returns:
        Candidates that start with the token.
    """
    lowered = incomplete.lower()
    return [value for value in values if value.lower().startswith(lowered)]
