"""Audio normalization for voiceprint samples."""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_store import get_voiceprint_db_path, list_all_voiceprint_samples

VOICEPRINT_AUDIO_PREPROCESS_VERSION = "audio-norm-v2"
VOICEPRINT_NORMALIZED_DIR = "normalized"
EMBEDDING_SILENCE_THRESHOLD = 0.01
EMBEDDING_SILENCE_PADDING_MS = 80
MIN_EMBEDDING_VOICED_SECONDS = 0.75


@dataclass(frozen=True, slots=True)
class VoiceprintNormalizeSummary:
    """Summary for normalizing stored voiceprint sample audio."""

    store_dir: Path
    normalized_dir: Path
    processed_count: int
    skipped_count: int


@dataclass(frozen=True, slots=True)
class EmbeddingAudioStats:
    """Audio stats for one voice embedding input clip."""

    duration_seconds: float
    voiced_duration_seconds: float
    rms: float
    silence_ratio: float
    clipping_ratio: float


def normalize_voiceprint_samples(*, store_dir: Path | None, rebuild: bool) -> VoiceprintNormalizeSummary:
    """
    Normalize all stored voiceprint samples into a deterministic derived directory.

    Args:
        store_dir: Optional voiceprint store directory.
        rebuild: Recreate normalized clips even when they already exist.

    Returns:
        Normalization summary.
    """
    resolved_store_dir = _resolve_store_dir(store_dir)
    samples = list_all_voiceprint_samples(get_voiceprint_db_path(store_dir))
    processed_count = 0
    skipped_count = 0
    for sample in samples:
        normalized_path = normalized_voiceprint_sample_path(sample, store_dir=resolved_store_dir)
        if _normalized_sample_is_current(sample.clip_path, normalized_path) and not rebuild:
            skipped_count += 1
            continue
        normalize_voiceprint_sample(sample, store_dir=resolved_store_dir)
        processed_count += 1
    return VoiceprintNormalizeSummary(
        resolved_store_dir,
        normalized_voiceprint_dir(resolved_store_dir),
        processed_count,
        skipped_count,
    )


def normalize_voiceprint_sample(sample: VoiceprintSampleRow, *, store_dir: Path | None) -> Path:
    """
    Normalize one stored sample for embedding.

    Args:
        sample: Stored sample metadata.
        store_dir: Optional voiceprint store directory.

    Returns:
        Normalized WAV path.
    """
    output_path = normalized_voiceprint_sample_path(sample, store_dir=store_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.norm-tmp{output_path.suffix}")
    try:
        _run_ffmpeg(_normalize_command(sample.clip_path, temp_path))
        trim_embedding_audio_silence(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return output_path


def trim_embedding_audio_silence(source: Path, output: Path) -> Path:
    """
    Trim leading and trailing silence from a WAV clip used for voice embeddings.

    Args:
        source: Mono 16 kHz s16 WAV clip.
        output: Trimmed WAV output path.

    Returns:
        Output path. If the source cannot be trimmed safely, a byte-for-byte
        copy is written instead.
    """
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        params, frames = _read_wav_frames(source)
    except (EOFError, OSError, wave.Error):
        output.write_bytes(source.read_bytes())
        return output
    if params.sampwidth != 2 or params.nchannels != 1 or not frames:
        output.write_bytes(source.read_bytes())
        return output
    sample_count = len(frames) // 2
    first, last = _voiced_sample_bounds(frames)
    if first is None or last is None:
        output.write_bytes(source.read_bytes())
        return output
    padding = round(params.framerate * EMBEDDING_SILENCE_PADDING_MS / 1000)
    start = max(0, first - padding)
    end = min(sample_count, last + padding + 1)
    if start == 0 and end == sample_count:
        output.write_bytes(source.read_bytes())
        return output
    _write_wav_frames(output, params, frames[start * 2 : end * 2])
    return output


def embedding_audio_stats(path: Path) -> EmbeddingAudioStats | None:
    """
    Return audio stats for a WAV clip used in voice embeddings.

    Args:
        path: Mono 16 kHz s16 WAV clip.

    Returns:
        Audio stats, or ``None`` when the clip cannot be analyzed.
    """
    try:
        params, frames = _read_wav_frames(path)
    except (EOFError, OSError, wave.Error):
        return None
    if params.sampwidth != 2 or params.nchannels != 1 or not frames:
        return None
    sample_count = len(frames) // 2
    if sample_count == 0:
        return None
    total_square = 0.0
    silent = 0
    clipped = 0
    voiced = 0
    for offset in range(0, len(frames), 2):
        value = int.from_bytes(frames[offset : offset + 2], "little", signed=True) / 32768.0
        absolute = abs(value)
        total_square += value * value
        if absolute < EMBEDDING_SILENCE_THRESHOLD:
            silent += 1
        else:
            voiced += 1
        if absolute > 0.98:
            clipped += 1
    return EmbeddingAudioStats(
        duration_seconds=sample_count / params.framerate,
        voiced_duration_seconds=voiced / params.framerate,
        rms=(total_square / sample_count) ** 0.5,
        silence_ratio=silent / sample_count,
        clipping_ratio=clipped / sample_count,
    )


def ensure_normalized_voiceprint_sample(sample: VoiceprintSampleRow, *, store_dir: Path | None) -> Path:
    """
    Return a current normalized sample, rebuilding stale derived audio.

    Args:
        sample: Stored sample metadata.
        store_dir: Optional voiceprint store directory.

    Returns:
        Current normalized WAV path.
    """
    output_path = normalized_voiceprint_sample_path(sample, store_dir=store_dir)
    if _normalized_sample_is_current(sample.clip_path, output_path):
        return output_path
    return normalize_voiceprint_sample(sample, store_dir=store_dir)


def normalized_voiceprint_sample_path(sample: VoiceprintSampleRow, *, store_dir: Path | None) -> Path:
    """
    Return the deterministic normalized WAV path for a sample.

    Args:
        sample: Stored sample metadata.
        store_dir: Optional voiceprint store directory.

    Returns:
        Derived normalized WAV path.
    """
    return normalized_voiceprint_dir(_resolve_store_dir(store_dir)) / sample.clip_rel_path


def normalized_voiceprint_clip_path(clip_path: Path, *, store_dir: Path | None) -> Path:
    """
    Return the normalized path corresponding to an original voiceprint clip.

    Args:
        clip_path: Original stored clip path.
        store_dir: Optional voiceprint store directory.

    Returns:
        Derived normalized WAV path.
    """
    resolved_store_dir = _resolve_store_dir(store_dir)
    relative_path = clip_path.expanduser().resolve().relative_to(resolved_store_dir)
    return normalized_voiceprint_dir(resolved_store_dir) / relative_path


def voiceprint_playback_clip_path(clip_path: Path, *, store_dir: Path | None) -> Path:
    """
    Prefer normalized audio for review playback when it exists.

    Args:
        clip_path: Original stored clip path.
        store_dir: Optional voiceprint store directory.

    Returns:
        Normalized clip path when available, otherwise the original clip path.
    """
    try:
        normalized_path = normalized_voiceprint_clip_path(clip_path, store_dir=store_dir)
    except ValueError:
        return clip_path
    return normalized_path if normalized_path.exists() else clip_path


def normalized_voiceprint_dir(store_dir: Path | None) -> Path:
    """
    Return the normalized sample directory for the current preprocessing version.

    Args:
        store_dir: Optional voiceprint store directory.

    Returns:
        Versioned normalized sample directory.
    """
    return _resolve_store_dir(store_dir) / VOICEPRINT_NORMALIZED_DIR / VOICEPRINT_AUDIO_PREPROCESS_VERSION


def _normalize_command(source: Path, output: Path) -> list[str]:
    """
    Build the ffmpeg command for conservative voiceprint normalization.

    Args:
        source: Original stored sample path.
        output: Normalized WAV path.

    Returns:
        ffmpeg command.
    """
    if not source.exists():
        raise FileNotFoundError(f"Voiceprint sample clip does not exist: {source}")
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-af",
        "loudnorm=I=-23:TP=-2:LRA=11:linear=true,alimiter=limit=0.95",
        str(output),
    ]


def _read_wav_frames(path: Path) -> tuple[wave._wave_params, bytes]:
    """Read WAV params and frames."""
    with wave.open(str(path), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    return params, frames


def _write_wav_frames(path: Path, params: wave._wave_params, frames: bytes) -> None:
    """Write WAV frames with existing params."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(params.nchannels)
        writer.setsampwidth(params.sampwidth)
        writer.setframerate(params.framerate)
        writer.writeframes(frames)


def _voiced_sample_bounds(frames: bytes) -> tuple[int | None, int | None]:
    """Return first and last sample indices above the silence threshold."""
    first: int | None = None
    last: int | None = None
    for index, offset in enumerate(range(0, len(frames), 2)):
        value = int.from_bytes(frames[offset : offset + 2], "little", signed=True) / 32768.0
        if abs(value) < EMBEDDING_SILENCE_THRESHOLD:
            continue
        if first is None:
            first = index
        last = index
    return first, last


def _run_ffmpeg(command: list[str]) -> None:
    """Run ffmpeg and surface readable errors."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found in PATH. Install ffmpeg first.") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}: {stderr}")


def _normalized_sample_is_current(source: Path, output: Path) -> bool:
    """Return whether normalized audio exists and is newer than the source."""
    if not output.exists():
        return False
    if not source.exists():
        raise FileNotFoundError(f"Voiceprint sample clip does not exist: {source}")
    return output.stat().st_mtime >= source.stat().st_mtime


def _resolve_store_dir(store_dir: Path | None) -> Path:
    """Resolve the voiceprint store directory."""
    return get_voiceprint_db_path(store_dir).parent
