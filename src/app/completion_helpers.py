"""Reusable shell completion value providers."""

from __future__ import annotations

from app.config import CONFIG_KEYS
from app.infra.ffmpeg import SUPPORTED_AUDIO_FORMATS
from app.voiceprint_embedding import (
    LOCAL_SPEECHBRAIN_MODEL,
)

ASR_MODELS = ("fun-asr",)
ASR_HOTWORD_MODES = ("auto", "off")
OSS_UPLOAD_MODES = ("auto", "true", "false")
VOICEPRINT_MODELS = (LOCAL_SPEECHBRAIN_MODEL,)
CLI_LANGUAGES = ("auto", "en", "zh")


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


def complete_asr_hotwords(incomplete: str) -> list[str]:
    """
    Complete ASR hotword mode values.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching hotword mode names.
    """
    return _matching(ASR_HOTWORD_MODES, incomplete)


def complete_oss_upload_mode(incomplete: str) -> list[str]:
    """
    Complete OSS upload mode values.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching upload modes.
    """
    return _matching(OSS_UPLOAD_MODES, incomplete)


def complete_voiceprint_model(incomplete: str) -> list[str]:
    """
    Complete supported voiceprint model storage keys.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching voiceprint model keys.
    """
    return _matching(VOICEPRINT_MODELS, incomplete)


def complete_cli_language(incomplete: str) -> list[str]:
    """
    Complete supported CLI help languages.

    Args:
        incomplete: Current shell token.

    Returns:
        Matching language values.
    """
    return _matching(CLI_LANGUAGES, incomplete)


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
