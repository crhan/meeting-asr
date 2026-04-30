"""Helpers for manual speaker review with local players."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from app.speaker_labeling import SpeakerSummary, load_transcript_result
from app.utils import format_ms_timestamp


def build_preview_command(*, video: Path, subtitle: Path, start_seconds: float) -> list[str]:
    """
    Build a player command that can show external subtitles.

    Args:
        video: Source video.
        subtitle: Subtitle path.
        start_seconds: Playback start offset.

    Returns:
        Command argv.
    """
    mpv = shutil.which("mpv")
    if mpv:
        return [
            mpv,
            f"--sub-file={subtitle.resolve()}",
            f"--start={start_seconds:.3f}",
            str(video.resolve()),
        ]
    iina_cli = _find_iina_cli()
    if iina_cli:
        return [
            iina_cli,
            "--no-stdin",
            f"--mpv-sub-file={subtitle.resolve()}",
            f"--mpv-start={start_seconds:.3f}",
            str(video.resolve()),
        ]
    if _ffplay_supports_subtitles_filter():
        subtitle_filter = f"subtitles=filename='{_escape_subtitle_path_for_ffmpeg(subtitle.resolve())}'"
        return [
            "ffplay",
            "-window_title",
            "Meeting-ASR Speaker Review",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(video.resolve()),
            "-vf",
            subtitle_filter,
        ]
    raise RuntimeError("No supported subtitle preview player found. Install mpv, IINA, or ffplay.")


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


def render_speaker_summary(summary: SpeakerSummary) -> str:
    """
    Render one speaker summary.

    Args:
        summary: Speaker summary.

    Returns:
        Terminal text.
    """
    lines = [
        f"{summary.anonymous_label} (speaker_id={summary.speaker_id})",
        f"  Segments: {summary.segment_count}",
        f"  First seen: {format_ms_timestamp(summary.first_begin_time_ms)}",
        "  Samples:",
    ]
    for segment in summary.sample_segments:
        start = format_ms_timestamp(segment.begin_time_ms)
        end = format_ms_timestamp(segment.end_time_ms)
        lines.append(f"    - [{start} - {end}] {_preview_text(segment.text)}")
    return "\n".join(lines)


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
