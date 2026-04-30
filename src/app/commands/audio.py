"""Audio utility commands."""

from __future__ import annotations

from pathlib import Path

import typer

from app.cli_errors import run_with_cli_errors
from app.ffmpeg_utils import extract_audio_for_asr

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("extract")
def extract(
    input_path: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o", file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("./output"), "--output-dir", file_okay=False, dir_okay=True),
    audio_format: str = typer.Option("flac", "--format", "--audio-format"),
) -> None:
    """Extract local media into ASR-ready mono audio."""
    output_path = output or output_dir / f"audio.{audio_format.lower()}"
    audio_path = run_with_cli_errors(lambda: extract_audio_for_asr(input_path, output_path, audio_format=audio_format))
    typer.echo(f"Audio written to: {audio_path}")
