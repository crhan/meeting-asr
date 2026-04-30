"""Project-oriented CLI commands."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional

import typer

from app.cli_errors import run_with_cli_errors
from app.completion_helpers import complete_audio_format, complete_model, complete_oss_upload_mode
from app.config import get_default_projects_dir
from app.project_manager import (
    ProjectTranscribeOptions,
    apply_project_speakers,
    create_project,
    init_project_git,
    load_manifest,
    parse_mapping_items,
    prepare_project_audio,
    project_paths,
    resolve_project_source_path,
    transcribe_project,
)
from app.speaker_labeling import build_speaker_summaries, load_transcript_result
from app.speaker_review import build_preview_command, preview_start_seconds, render_speaker_summary
from app.srt_compare import build_report, parse_srt
from app.utils import configure_logging, safe_write_text

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
speakers_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(speakers_app, name="speakers", help="Review and name project speakers.")


@app.command("create")
def create(
    input: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
    hash_source: bool = typer.Option(False, "--hash-source/--no-hash-source"),
) -> None:
    """Create a project directory with project.json metadata."""
    manifest = run_with_cli_errors(
        lambda: create_project(
            input,
            title=title,
            projects_dir=projects_dir,
            project_dir=project_dir,
            meeting_time=meeting_time,
            hash_source=hash_source,
        )
    )
    _echo_project_created(_created_project_root(project_dir, projects_dir, manifest), manifest)


@app.command("prepare")
def prepare(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
) -> None:
    """Extract project audio without starting cloud transcription."""
    configure_logging()
    audio_path = run_with_cli_errors(lambda: prepare_project_audio(project_dir, audio_format=audio_format))
    typer.echo(f"Audio written to: {audio_path}")


@app.command("transcribe")
def transcribe(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    speaker_count: Optional[int] = typer.Option(None, "--speaker-count", min=1),
    language: Optional[str] = typer.Option("zh,en", "--language"),
    model: str = typer.Option("fun-asr", "--model", autocompletion=complete_model),
    oss_upload: str = typer.Option("auto", "--oss-upload", autocompletion=complete_oss_upload_mode),
    file_url: Optional[str] = typer.Option(None, "--file-url"),
    generate_srt: bool = typer.Option(True, "--generate-srt/--no-generate-srt"),
    timestamp_alignment: bool = typer.Option(True, "--timestamp-alignment/--no-timestamp-alignment"),
    disfluency_removal: bool = typer.Option(False, "--disfluency-removal/--no-disfluency-removal"),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
) -> None:
    """Transcribe a project and write structured artifacts."""
    configure_logging()
    options = _project_transcribe_options(
        speaker_count=speaker_count,
        language=language,
        model=model,
        oss_upload=oss_upload,
        file_url=file_url,
        generate_srt=generate_srt,
        timestamp_alignment=timestamp_alignment,
        disfluency_removal=disfluency_removal,
        audio_format=audio_format,
    )
    summary = run_with_cli_errors(lambda: transcribe_project(project_dir, options))
    _echo_transcribe_summary(
        summary.project_dir,
        summary.task_id,
        summary.detected_speaker_count,
        summary.sentence_count,
    )


@app.command("run")
def run(
    input: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
    speaker_count: Optional[int] = typer.Option(None, "--speaker-count", min=1),
    language: Optional[str] = typer.Option("zh,en", "--language"),
    model: str = typer.Option("fun-asr", "--model", autocompletion=complete_model),
    oss_upload: str = typer.Option("auto", "--oss-upload", autocompletion=complete_oss_upload_mode),
    file_url: Optional[str] = typer.Option(None, "--file-url"),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
) -> None:
    """Create a project and run the default transcription workflow."""
    configure_logging()
    manifest = run_with_cli_errors(
        lambda: create_project(
            input,
            title=title,
            projects_dir=projects_dir,
            project_dir=project_dir,
            meeting_time=meeting_time,
            hash_source=False,
        )
    )
    root = _created_project_root(project_dir, projects_dir, manifest)
    options = _project_transcribe_options(
        speaker_count=speaker_count,
        language=language,
        model=model,
        oss_upload=oss_upload,
        file_url=file_url,
        generate_srt=True,
        timestamp_alignment=True,
        disfluency_removal=False,
        audio_format=audio_format,
    )
    summary = run_with_cli_errors(lambda: transcribe_project(root, options))
    _echo_transcribe_summary(
        summary.project_dir,
        summary.task_id,
        summary.detected_speaker_count,
        summary.sentence_count,
    )


@app.command("status")
def status(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
) -> None:
    """Print a project status summary."""
    manifest = run_with_cli_errors(lambda: load_manifest(project_dir))
    paths = project_paths(project_dir)
    typer.echo(f"Project: {paths.root}")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Title: {manifest.title}")
    typer.echo(f"Status: {manifest.status}")
    typer.echo(f"Source: {manifest.source.path}")
    if manifest.source.original_path:
        typer.echo(f"Original source: {manifest.source.original_path}")
    typer.echo(f"Audio: {manifest.audio.get('path', '-')}")
    typer.echo(f"Task ID: {manifest.asr.get('task_id', '-')}")
    typer.echo(f"Detected speakers: {manifest.speakers.get('detected_ids', [])}")


@app.command("git-init")
def git_init(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
) -> None:
    """Initialize optional Git tracking for human-edited project files."""
    gitignore_path = run_with_cli_errors(lambda: init_project_git(project_dir))
    typer.echo(f"Git initialized: {project_dir.expanduser().resolve()}")
    typer.echo(f"Git ignore written to: {gitignore_path}")


@speakers_app.command("inspect")
def speakers_inspect(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    sample_count: int = typer.Option(5, "--sample-count", min=1, max=20),
) -> None:
    """Print per-speaker samples for a project."""
    result = run_with_cli_errors(lambda: load_transcript_result(project_paths(project_dir).asr_dir / "sentences.json"))
    summaries = build_speaker_summaries(result, sample_count=sample_count)
    if not summaries:
        typer.echo("No detected speakers found in the transcript.")
        raise typer.Exit(code=1)
    for index, summary in enumerate(summaries):
        if index:
            typer.echo("")
        typer.echo(render_speaker_summary(summary))


@speakers_app.command("preview")
def speakers_preview(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    speaker_id: Optional[int] = typer.Option(None, "--speaker-id"),
    padding_seconds: int = typer.Option(8, "--padding-seconds", min=0),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Open the source video with the project's subtitle for speaker review."""
    manifest = run_with_cli_errors(lambda: load_manifest(project_dir))
    paths = project_paths(project_dir)
    start_seconds = preview_start_seconds(
        paths.asr_dir / "sentences.json",
        speaker_id,
        padding_seconds,
    )
    command = build_preview_command(
        video=resolve_project_source_path(paths.root, manifest),
        subtitle=paths.exports_dir / "subtitle.srt",
        start_seconds=start_seconds,
    )
    if dry_run:
        typer.echo(" ".join(command))
        return
    run_with_cli_errors(lambda: subprocess.run(command, check=True))


@speakers_app.command("apply")
def speakers_apply(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    mappings: list[str] = typer.Option([], "--map"),
) -> None:
    """Apply speaker_id=name mappings to a project."""
    sentences_path = project_paths(project_dir).asr_dir / "sentences.json"
    result = run_with_cli_errors(lambda: load_transcript_result(sentences_path))
    resolved = parse_mapping_items(mappings, set(result.detected_speakers))
    mapping_path, transcript_path, srt_path = run_with_cli_errors(
        lambda: apply_project_speakers(project_dir, resolved)
    )
    typer.echo(f"Mapping written to: {mapping_path}")
    typer.echo(f"Named transcript written to: {transcript_path}")
    typer.echo(f"Named subtitle written to: {srt_path}")


@speakers_app.command("compare-srt")
def speakers_compare_srt(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    dingtalk_srt: Path = typer.Option(..., "--dingtalk-srt", exists=True, file_okay=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Compare a DingTalk SRT with the project's subtitle."""
    paths = project_paths(project_dir)
    local_srt = _preferred_project_srt(paths)
    output_path = output or paths.exports_dir / "speaker_comparison.md"
    report = build_report(
        dingtalk=parse_srt(dingtalk_srt, source="dingtalk"),
        ours=parse_srt(local_srt, source="local"),
    )
    run_with_cli_errors(lambda: safe_write_text(output_path, report))
    typer.echo(f"Report written to: {output_path.resolve()}")


def _project_transcribe_options(
    *,
    speaker_count: int | None,
    language: str | None,
    model: str,
    oss_upload: str,
    file_url: str | None,
    generate_srt: bool,
    timestamp_alignment: bool,
    disfluency_removal: bool,
    audio_format: str,
) -> ProjectTranscribeOptions:
    """Build normalized project transcription options."""
    return ProjectTranscribeOptions(
        speaker_count=speaker_count,
        language=language,
        model=model,
        oss_upload=oss_upload,
        file_url=file_url,
        generate_srt=generate_srt,
        timestamp_alignment=timestamp_alignment,
        disfluency_removal=disfluency_removal,
        audio_format=audio_format,
    )


def _echo_transcribe_summary(project_dir: Path, task_id: str, speaker_count: int, sentence_count: int) -> None:
    """Print project transcription summary."""
    typer.echo("")
    typer.echo("Project transcription completed.")
    typer.echo(f"Project: {project_dir}")
    typer.echo(f"Task ID: {task_id}")
    typer.echo(f"Detected speakers: {speaker_count}")
    typer.echo(f"Sentence count: {sentence_count}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  cd {_shell_quote_path(project_dir)}")
    typer.echo("  meeting-asr project speakers inspect")
    typer.echo("  meeting-asr project speakers preview")


def _echo_project_created(project_dir: Path, manifest) -> None:
    """Print project creation output with copyable next commands."""
    resolved_dir = project_dir.expanduser().resolve()
    typer.echo("")
    typer.echo("Project created.")
    typer.echo(f"Project: {resolved_dir}")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Source: {manifest.source.path}")
    typer.echo(f"Status: {manifest.status}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  cd {_shell_quote_path(resolved_dir)}")
    typer.echo("  meeting-asr project transcribe")
    typer.echo("  meeting-asr project status")


def _shell_quote_path(path: Path) -> str:
    """Quote a path so users can paste it into POSIX shells."""
    return shlex.quote(str(path.expanduser().resolve()))


def _created_project_root(project_dir: Path | None, projects_dir: Path | None, manifest) -> Path:
    """Resolve the root for a freshly created project."""
    if project_dir is not None:
        return project_dir.expanduser().resolve()
    base_dir = projects_dir.expanduser().resolve() if projects_dir else get_default_projects_dir()
    return base_dir / manifest.project_id.replace("-", "_", 1)


def _preferred_project_srt(paths) -> Path:
    """Prefer named subtitles after speaker mapping, otherwise anonymous subtitles."""
    named_srt = paths.exports_dir / "subtitle_named.srt"
    if named_srt.exists():
        return named_srt
    return paths.exports_dir / "subtitle.srt"
