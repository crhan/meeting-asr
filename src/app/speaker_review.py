"""Helpers for manual speaker review with local players."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from app.speaker_labeling import SpeakerSummary, load_transcript_result
from app.utils import format_ms_timestamp


def build_preview_command(
    *,
    video: Path,
    subtitle: Path,
    start_seconds: float,
    duration_seconds: float | None = None,
) -> list[str]:
    """
    Build a player command that can show external subtitles.

    Args:
        video: Source video.
        subtitle: Subtitle path.
        start_seconds: Playback start offset.
        duration_seconds: Optional playback duration limit.

    Returns:
        Command argv.
    """
    video_path = _existing_path(video, "Video")
    subtitle_path = _existing_path(subtitle, "Subtitle")
    if mpv := shutil.which("mpv"):
        return _mpv_preview_command(mpv, video_path, subtitle_path, start_seconds, duration_seconds)
    iina_cli = _find_iina_cli()
    if iina_cli:
        return _iina_preview_command(iina_cli, video_path, subtitle_path, start_seconds, duration_seconds)
    if _ffplay_supports_subtitles_filter():
        return _ffplay_preview_command(video_path, subtitle_path, start_seconds, duration_seconds)
    raise RuntimeError("No supported subtitle preview player found. Install mpv, IINA, or ffplay.")


def _mpv_preview_command(
    mpv: str,
    video: Path,
    subtitle: Path,
    start_seconds: float,
    duration_seconds: float | None,
) -> list[str]:
    """
    Build an mpv subtitle preview command.

    Args:
        mpv: mpv executable path.
        video: Resolved video path.
        subtitle: Resolved subtitle path.
        start_seconds: Playback start offset.
        duration_seconds: Optional playback duration limit.

    Returns:
        Command argv.
    """
    command = [mpv, "--resume-playback=no", f"--sub-file={subtitle}", "--sid=1", f"--start={start_seconds:.3f}"]
    if duration_seconds is not None:
        command.append(f"--length={duration_seconds:.3f}")
    command.append(str(video))
    return command


def _iina_preview_command(
    iina_cli: str,
    video: Path,
    subtitle: Path,
    start_seconds: float,
    duration_seconds: float | None,
) -> list[str]:
    """
    Build an IINA subtitle preview command.

    Args:
        iina_cli: IINA command line launcher.
        video: Resolved video path.
        subtitle: Resolved subtitle path.
        start_seconds: Playback start offset.
        duration_seconds: Optional playback duration limit.

    Returns:
        Command argv.
    """
    _stage_iina_subtitle(video, subtitle)
    raw_options = ["--resume-playback=no", "--sub-auto=fuzzy", "--sid=1", f"--start={start_seconds:.3f}"]
    if duration_seconds is not None:
        raw_options.append(f"--length={duration_seconds:.3f}")
    return [iina_cli, "--no-stdin", str(video), "--", *raw_options]


def _ffplay_preview_command(
    video: Path,
    subtitle: Path,
    start_seconds: float,
    duration_seconds: float | None,
) -> list[str]:
    """
    Build an ffplay subtitle preview command.

    Args:
        video: Resolved video path.
        subtitle: Resolved subtitle path.
        start_seconds: Playback start offset.
        duration_seconds: Optional playback duration limit.

    Returns:
        Command argv.
    """
    subtitle_filter = f"subtitles=filename='{_escape_subtitle_path_for_ffmpeg(subtitle)}'"
    command = [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-autoexit",
        "-window_title",
        "Meeting-ASR Speaker Review",
        "-ss",
        f"{start_seconds:.3f}",
    ]
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    command.extend(["-i", str(video), "-vf", subtitle_filter])
    return command


def build_audio_preview_command(
    *,
    media: Path,
    start_seconds: float,
    duration_seconds: float | None = None,
) -> list[str]:
    """
    Build an audio-only player command for local speaker review.

    Args:
        media: Source media file.
        start_seconds: Playback start offset.
        duration_seconds: Optional playback duration limit.

    Returns:
        Command argv.
    """
    media_path = _existing_path(media, "Media")
    mpv = shutil.which("mpv")
    if mpv:
        command = [
            mpv,
            "--really-quiet",
            "--resume-playback=no",
            "--vid=no",
            "--force-window=no",
            f"--start={start_seconds:.3f}",
        ]
        if duration_seconds is not None:
            command.append(f"--length={duration_seconds:.3f}")
        command.append(str(media_path))
        return command
    if shutil.which("ffplay"):
        command = [
            "ffplay",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-nodisp",
            "-autoexit",
            "-ss",
            f"{start_seconds:.3f}",
        ]
        if duration_seconds is not None:
            command.extend(["-t", f"{duration_seconds:.3f}"])
        command.extend(["-i", str(media_path)])
        return command
    raise RuntimeError("No supported audio preview player found. Install mpv or ffplay.")


def preview_start_seconds(sentences_json: Path, speaker_id: int | None, padding_seconds: int) -> float:
    """
    Resolve preview start time.

    Args:
        sentences_json: Sentences JSON path.
        speaker_id: Optional speaker id.
        padding_seconds: Seek padding.

    Returns:
        Start seconds.
    """
    if speaker_id is None:
        return 0.0
    start_ms = _find_first_segment_time_ms(sentences_json, speaker_id)
    return max(0.0, start_ms / 1000.0 - float(padding_seconds))


def render_speaker_summary(
    summary: SpeakerSummary,
    mapped_name: str | None = None,
    match_summary: str | None = None,
) -> str:
    """
    Render one speaker summary.

    Args:
        summary: Speaker summary.
        mapped_name: Optional human speaker name.
        match_summary: Optional voiceprint match summary.

    Returns:
        Terminal text.
    """
    lines = [
        f"{summary.anonymous_label} (speaker_id={summary.speaker_id})",
        f"  Segments: {summary.segment_count}",
        f"  First seen: {format_ms_timestamp(summary.first_begin_time_ms)}",
        "  Samples:",
    ]
    if mapped_name:
        lines.insert(1, _confirmed_name_line(mapped_name))
    if match_summary:
        insert_index = 2 if mapped_name else 1
        lines.insert(insert_index, f"  Voiceprint match: {match_summary}")
    for segment in summary.sample_segments:
        start = format_ms_timestamp(segment.begin_time_ms)
        end = format_ms_timestamp(segment.end_time_ms)
        lines.append(f"    - [{start} - {end}] {_preview_text(segment.text)}")
    return "\n".join(lines)


def _confirmed_name_line(mapped_name: str) -> str:
    """
    Format a manually confirmed speaker name for terminal output.

    Args:
        mapped_name: Human-confirmed speaker name.

    Returns:
        Styled terminal text.
    """
    return typer.style(f"  Name: {mapped_name}", fg=typer.colors.GREEN, bold=True)


def _find_first_segment_time_ms(sentences_json: Path, speaker_id: int) -> int:
    """Find first segment time for speaker."""
    result = load_transcript_result(sentences_json)
    for segment in result.sentences:
        if segment.speaker_id == speaker_id:
            return segment.begin_time_ms
    raise typer.BadParameter(f"speaker_id={speaker_id} was not found in {sentences_json}")


def _find_iina_cli() -> str | None:
    """Find IINA command line launcher on macOS."""
    cli = shutil.which("iina")
    if cli:
        return cli
    app_cli = Path("/Applications/IINA.app/Contents/MacOS/iina-cli")
    return str(app_cli) if app_cli.exists() else None


def _existing_path(path: Path, label: str) -> Path:
    """
    Resolve a required preview input.

    Args:
        path: Input path.
        label: Human-readable input label.

    Returns:
        Resolved path.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved


def _stage_iina_subtitle(video: Path, subtitle: Path) -> Path:
    """
    Copy subtitles next to the video so IINA's mpv core auto-loads them.

    Args:
        video: Source video path.
        subtitle: Generated subtitle path.

    Returns:
        Sidecar subtitle path.
    """
    sidecar = video.with_suffix(".srt")
    if sidecar == subtitle:
        return sidecar
    if not sidecar.exists() or sidecar.read_bytes() != subtitle.read_bytes():
        shutil.copyfile(subtitle, sidecar)
    return sidecar


def _ffplay_supports_subtitles_filter() -> bool:
    """Return whether ffplay has subtitles filter."""
    if shutil.which("ffplay") is None:
        return False
    completed = subprocess.run(["ffplay", "-hide_banner", "-filters"], capture_output=True, text=True, check=False)
    return " subtitles " in completed.stdout


def _escape_subtitle_path_for_ffmpeg(path: Path) -> str:
    """Escape subtitle path for ffplay subtitles filter."""
    escaped = str(path).replace("\\", "\\\\")
    escaped = escaped.replace("'", r"\'")
    return escaped.replace(":", r"\:")


def _preview_text(text: str, *, limit: int = 90) -> str:
    """Trim sample text for terminal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
