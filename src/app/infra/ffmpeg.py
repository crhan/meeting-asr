"""FFmpeg helpers for ASR-ready audio extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path

SUPPORTED_AUDIO_FORMATS = {"wav", "flac"}


def extract_audio_for_asr(
    input_path: str | Path, output_path: str | Path, *, audio_format: str = "flac"
) -> Path:
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
        raise ValueError(
            f"Unsupported audio format: {audio_format}. Supported formats: {supported}."
        )
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


def extract_audio_clip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """
    Extract one mono 16kHz s16 WAV clip from local media.

    Args:
        input_path: Local video or audio file.
        output_path: Output WAV path.
        start_seconds: Clip start time in seconds.
        duration_seconds: Clip duration in seconds.

    Returns:
        Output path.
    """
    _validate_audio_clip_times(start_seconds, duration_seconds)
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(_audio_clip_command(source, output, start_seconds, duration_seconds))
    return output


def probe_media_duration_seconds(path: str | Path) -> float:
    """
    Probe media duration with ffprobe.

    Args:
        path: Local media file.

    Returns:
        Duration in seconds.
    """
    media = Path(path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"Media file does not exist: {media}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found in PATH. Install ffmpeg first."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"ffprobe failed with exit code {completed.returncode}: {stderr}"
        )
    duration_text = completed.stdout.strip()
    try:
        duration = float(duration_text)
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid duration: {duration_text}"
        ) from exc
    if duration <= 0:
        raise RuntimeError(f"ffprobe returned a non-positive duration: {duration}")
    return duration


def _validate_audio_clip_times(start_seconds: float, duration_seconds: float) -> None:
    """
    Validate clip timing.

    Args:
        start_seconds: Clip start time in seconds.
        duration_seconds: Clip duration in seconds.
    """
    if start_seconds < 0:
        raise ValueError("start_seconds must be >= 0.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0.")


def _audio_clip_command(
    source: Path, output: Path, start_seconds: float, duration_seconds: float
) -> list[str]:
    """
    Build an ffmpeg command for one reference clip.

    Args:
        source: Source media path.
        output: Output WAV path.
        start_seconds: Clip start time in seconds.
        duration_seconds: Clip duration in seconds.

    Returns:
        ffmpeg command.
    """
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output),
    ]


def _run_ffmpeg(command: list[str]) -> None:
    """Run ffmpeg and surface readable errors."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg first."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"ffmpeg failed with exit code {completed.returncode}: {stderr}"
        )
