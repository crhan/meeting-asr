"""FFmpeg helpers for ASR-ready audio extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path

SUPPORTED_AUDIO_FORMATS = {"wav", "flac"}


def extract_audio_for_asr(input_path: str | Path, output_path: str | Path, *, audio_format: str = "flac") -> Path:
    """
    Extract mono 16kHz s16 audio from a local media file.

    Args:
        input_path: Local video or audio file.
        output_path: Output audio path.
        audio_format: ``wav`` or ``flac``.

    Returns:
        Output path.
    """
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    normalized_format = audio_format.strip().lower()
    if normalized_format not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise ValueError(f"Unsupported audio format: {audio_format}. Supported formats: {supported}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output),
    ]
    _run_ffmpeg(command)
    return output


def extract_audio_to_wav(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Extract mono 16kHz s16 WAV from a local media file.

    Args:
        input_path: Local video or audio file.
        output_path: Output WAV path.

    Returns:
        Output path.
    """
    return extract_audio_for_asr(input_path, output_path, audio_format="wav")


def _run_ffmpeg(command: list[str]) -> None:
    """Run ffmpeg and surface readable errors."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found in PATH. Install ffmpeg first.") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}: {stderr}")
