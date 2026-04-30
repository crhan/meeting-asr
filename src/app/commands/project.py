"""Project-oriented CLI commands."""

from __future__ import annotations

import json
import os
import select
import shlex
import signal
import subprocess
import sys
import termios
import tty
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from app.commands import transcript as transcript_commands
from app.cli_errors import run_with_cli_errors
from app.cli_ui import CliProgressReporter, run_with_progress
from app.completion_helpers import (
    complete_audio_format,
    complete_model,
    complete_oss_upload_mode,
    complete_voiceprint_model,
    complete_voiceprint_provider,
)
from app.config import get_default_projects_dir
from app.ffmpeg_utils import extract_audio_clip
from app.models import SentenceSegment, TranscriptResult
from app.project_manager import (
    ProjectListItem,
    ProjectManifest,
    ProjectTranscribeOptions,
    ProjectTranscribeSummary,
    apply_project_speakers,
    create_project,
    init_project_git,
    list_projects,
    load_manifest,
    parse_mapping_items,
    prepare_project_audio,
    project_paths,
    resolve_project_source_path,
    transcribe_project,
)
from app.speaker_labeling import build_speaker_summaries, load_transcript_result
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers
from app.speaker_review import (
    build_audio_preview_command,
    build_preview_command,
    preview_start_seconds,
    render_speaker_summary,
)
from app.srt_compare import build_report, parse_srt
from app.utils import configure_logging, format_ms_timestamp, safe_write_text

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
speakers_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(speakers_app, name="speakers", help="Review and name project speakers.")
app.add_typer(transcript_commands.app, name="transcript", help="View project transcript artifacts.")

MORE_SAMPLES_COMMAND = "/more"
AUDIO_PREVIEW_COMMAND = "/audio"
SPEAKER_APPLY_COMMANDS = (MORE_SAMPLES_COMMAND, AUDIO_PREVIEW_COMMAND)
SPEAKER_APPLY_PREVIEW_LEAD_SECONDS = 1.0
SPEAKER_APPLY_PREVIEW_TAIL_SECONDS = 1.0
SPEAKER_APPLY_PREVIEW_MIN_SECONDS = 4.0
SPEAKER_APPLY_PREVIEW_MAX_SECONDS = 18.0
SPEAKER_APPLY_PREVIEW_GAP_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class SpeakerApplyPreviewContext:
    """Inputs needed to play speaker previews during interactive apply."""

    project_root: Path
    video: Path


@dataclass(frozen=True, slots=True)
class SpeakerApplyPreviewTarget:
    """Selected speaker sample for finite playback."""

    segment: SentenceSegment
    start_seconds: float
    duration_seconds: float


@app.command("create")
def create(
    input: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
    hash_source: bool = typer.Option(False, "--hash-source/--no-hash-source"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Create a project directory with project.json metadata."""
    manifest = run_with_progress(
        lambda reporter: create_project(
            input,
            title=title,
            projects_dir=projects_dir,
            project_dir=project_dir,
            meeting_time=meeting_time,
            hash_source=hash_source,
            progress=reporter,
        ),
        description="Creating project",
        enabled=progress,
    )
    _echo_project_created(_created_project_root(project_dir, projects_dir, manifest), manifest)


@app.command("prepare")
def prepare(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Extract project audio without starting cloud transcription."""
    configure_logging()
    audio_path = run_with_progress(
        lambda reporter: prepare_project_audio(project_dir, audio_format=audio_format, progress=reporter),
        description="Preparing project audio",
        total=1,
        enabled=progress,
    )
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
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
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
    summary = run_with_progress(
        lambda reporter: transcribe_project(project_dir, options, progress=reporter),
        description="Transcribing project",
        total=7,
        enabled=progress,
    )
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
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Create a project and run the default transcription workflow."""
    configure_logging()
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
    _, summary = run_with_progress(
        lambda reporter: _create_and_transcribe_project(
            input,
            title=title,
            projects_dir=projects_dir,
            project_dir=project_dir,
            meeting_time=meeting_time,
            options=options,
            progress=reporter,
        ),
        description="Running project workflow",
        enabled=progress,
    )
    _echo_transcribe_summary(
        summary.project_dir,
        summary.task_id,
        summary.detected_speaker_count,
        summary.sentence_count,
    )


@app.command("list")
def list_command(
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
) -> None:
    """List projects under the default or specified projects directory."""
    result = run_with_cli_errors(lambda: list_projects(projects_dir))
    _echo_project_list(result.projects_dir, result.projects)


def _create_and_transcribe_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    options: ProjectTranscribeOptions,
    progress: CliProgressReporter | None,
) -> tuple[ProjectManifest, ProjectTranscribeSummary]:
    """
    Create a project and immediately transcribe it.

    Args:
        input_path: Local source media file.
        title: Optional human title.
        projects_dir: Optional parent directory.
        project_dir: Optional explicit project directory.
        meeting_time: Optional meeting start time string.
        options: Project transcription options.
        progress: Optional progress reporter.

    Returns:
        Created manifest and transcription summary.
    """
    manifest = create_project(
        input_path,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=meeting_time,
        hash_source=False,
        progress=progress,
    )
    root = _created_project_root(project_dir, projects_dir, manifest)
    summary = transcribe_project(root, options, progress=progress)
    return manifest, summary


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
    speaker_mapping = run_with_cli_errors(lambda: _load_existing_speaker_mapping(project_dir))
    speaker_matches = run_with_cli_errors(lambda: _load_speaker_match_summaries(project_dir, speaker_mapping))
    for index, summary in enumerate(summaries):
        if index:
            typer.echo("")
        typer.echo(
            render_speaker_summary(
                summary,
                mapped_name=speaker_mapping.get(summary.speaker_id),
                match_summary=speaker_matches.get(summary.speaker_id),
            )
        )


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
        subtitle=_preferred_project_srt(paths),
        start_seconds=start_seconds,
    )
    if dry_run:
        typer.echo(" ".join(command))
        return
    run_with_cli_errors(lambda: subprocess.run(command, check=True))


@speakers_app.command("apply")
def speakers_apply(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    mappings: list[str] = typer.Option([], "--map", help="Non-interactive speaker_id=name mapping."),
    sample_count: int = typer.Option(3, "--sample-count", min=1, max=20, help="Samples shown per speaker."),
) -> None:
    """Interactively apply speaker names to a project."""
    sentences_path = project_paths(project_dir).asr_dir / "sentences.json"
    result = run_with_cli_errors(lambda: load_transcript_result(sentences_path))
    resolved = _resolve_speaker_mappings(
        project_dir=project_dir,
        mappings=mappings,
        sample_count=sample_count,
        known_speakers=set(result.detected_speakers),
        result=result,
    )
    mapping_path, transcript_path, srt_path = run_with_cli_errors(
        lambda: apply_project_speakers(project_dir, resolved)
    )
    typer.echo(f"Mapping written to: {mapping_path}")
    typer.echo(f"Named transcript written to: {transcript_path}")
    typer.echo(f"Named subtitle written to: {srt_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  meeting-asr project speakers preview")
    typer.echo("  meeting-asr project transcript show")
    typer.echo("  meeting-asr voiceprint capture")
    typer.echo(f"  open {_shell_quote_path(transcript_path)}")


@speakers_app.command("match")
def speakers_match(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    provider: Optional[str] = typer.Option(None, "--provider", autocompletion=complete_voiceprint_provider),
    endpoint: Optional[str] = typer.Option(None, "--endpoint"),
    model: Optional[str] = typer.Option(None, "--model", autocompletion=complete_voiceprint_model),
    threshold: float = typer.Option(0.75, "--threshold", min=0.0, max=1.0),
    sample_count: int = typer.Option(2, "--sample-count", min=1, max=20),
    max_seconds: float = typer.Option(12.0, "--max-seconds", min=0.1),
    padding_seconds: float = typer.Option(0.5, "--padding-seconds", min=0.0),
    apply_matches: bool = typer.Option(False, "--apply"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Match project speakers against the cross-project voiceprint library."""
    summary = run_with_progress(
        lambda reporter: match_project_speakers(
            project_dir,
            store_dir=store_dir,
            provider=provider,
            endpoint=endpoint,
            model=model,
            threshold=threshold,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            progress=reporter,
        ),
        description="Matching project speakers",
        enabled=progress,
    )
    _echo_match_summary(summary)
    if apply_matches and summary.accepted_mapping:
        run_with_cli_errors(lambda: apply_project_speakers(project_dir, summary.accepted_mapping))
        typer.echo("Applied accepted speaker matches.")


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


def _echo_project_list(projects_dir: Path, projects: list[ProjectListItem]) -> None:
    """
    Print project list rows.

    Args:
        projects_dir: Resolved projects parent directory.
        projects: Project rows to print.
    """
    typer.echo(f"Projects: {projects_dir}")
    if not projects:
        typer.echo("No projects found.")
        return
    for project in projects:
        typer.echo(
            f"- {project.created_at} | {project.status} | {project.project_id} | "
            f"{project.title} | {project.project_dir}"
        )


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


def _echo_match_summary(summary: SpeakerMatchSummary) -> None:
    """
    Print speaker match results.

    Args:
        summary: Match summary.
    """
    typer.echo(f"Matches written to: {summary.match_path}")
    typer.echo(f"Provider: {summary.provider}")
    typer.echo(f"Model: {summary.model}")
    typer.echo(f"Threshold: {summary.threshold:.3f}")
    for match in summary.matches:
        name = match.name or "unknown"
        status = "accepted" if match.accepted else "review"
        typer.echo(f"{match.label} -> {name}  score={match.score:.3f}  {status}")


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


def _resolve_speaker_mappings(
    *,
    project_dir: Path,
    mappings: list[str],
    sample_count: int,
    known_speakers: set[int],
    result: TranscriptResult,
) -> dict[int, str]:
    """
    Resolve speaker mappings from CLI flags or prompts.

    Args:
        project_dir: Project root.
        mappings: Explicit ``--map`` values.
        sample_count: Number of samples to show per speaker.
        known_speakers: Speaker ids present in the transcript.
        result: Loaded transcript result.

    Returns:
        User-provided speaker mapping.
    """
    if mappings:
        return parse_mapping_items(mappings, known_speakers)
    return _prompt_speaker_mappings(project_dir, result, sample_count=sample_count)


def _prompt_speaker_mappings(project_dir: Path, result: TranscriptResult, *, sample_count: int) -> dict[int, str]:
    """
    Prompt for speaker names with transcript samples.

    Args:
        project_dir: Project root.
        result: Loaded transcript result.
        sample_count: Number of samples per speaker.

    Returns:
        Speaker mapping entered by the user.
    """
    summaries = build_speaker_summaries(result, sample_count=sample_count)
    if not summaries:
        typer.echo("No detected speakers found in the transcript.")
        raise typer.Exit(code=1)
    typer.echo("Enter a name for each speaker.")
    typer.echo("Commands: /more shows more samples, /audio plays the visible samples.")
    typer.echo("Press Enter to keep the default label.")
    existing = _load_existing_speaker_mapping(project_dir)
    matched = _load_speaker_match_defaults(project_dir)
    segments = _speaker_segments_by_id(result)
    preview_context = _speaker_apply_preview_context(project_dir)
    resolved: dict[int, str] = {}
    for index, summary in enumerate(summaries):
        if index:
            typer.echo("")
        typer.echo(render_speaker_summary(summary))
        default_name = existing.get(summary.speaker_id) or matched.get(summary.speaker_id) or summary.anonymous_label
        resolved[summary.speaker_id] = _prompt_speaker_name(
            speaker_label=summary.anonymous_label,
            default_name=default_name,
            segments=segments.get(summary.speaker_id, []),
            visible_segments=list(summary.sample_segments),
            next_offset=sample_count,
            sample_count=sample_count,
            preview_context=preview_context,
        )
    return resolved


def _prompt_speaker_name(
    *,
    speaker_label: str,
    default_name: str,
    segments: list[SentenceSegment],
    visible_segments: list[SentenceSegment],
    next_offset: int,
    sample_count: int,
    preview_context: SpeakerApplyPreviewContext,
) -> str:
    """
    Prompt for one speaker name, allowing more samples on demand.

    Args:
        speaker_label: Anonymous speaker label.
        default_name: Existing or anonymous default name.
        segments: All transcript segments for this speaker.
        visible_segments: Speaker segments currently visible in the terminal.
        next_offset: First segment offset not yet displayed.
        sample_count: Number of samples per extra batch.
        preview_context: Media inputs for slash-command previews.

    Returns:
        Confirmed speaker name.
    """
    offset = next_offset
    preview_segments = visible_segments or segments[:sample_count]
    while True:
        prompt = f"Name for {speaker_label} (/more /audio)"
        name = typer.prompt(prompt, default=default_name).strip()
        command = _speaker_apply_command(name)
        if command is None:
            return name or default_name
        _remember_prompt_history(command)
        if command == AUDIO_PREVIEW_COMMAND:
            _play_speaker_apply_preview(
                preview_context=preview_context,
                speaker_label=speaker_label,
                segments=preview_segments,
            )
            continue
        offset, samples = _show_more_speaker_samples(speaker_label, segments, offset, sample_count)
        if samples:
            preview_segments = samples


def _speaker_apply_preview_context(project_dir: Path) -> SpeakerApplyPreviewContext:
    """
    Build preview inputs for interactive speaker apply commands.

    Args:
        project_dir: Project root.

    Returns:
        Preview context with resolved media and subtitle paths.
    """
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    return SpeakerApplyPreviewContext(
        project_root=paths.root,
        video=resolve_project_source_path(paths.root, manifest),
    )


def _speaker_apply_command(value: str) -> str | None:
    """
    Parse a speaker apply slash command.

    Args:
        value: Raw prompt input.

    Returns:
        Normalized command, or ``None`` when the input is a speaker name.
    """
    command = value.strip().lower()
    return command if command in SPEAKER_APPLY_COMMANDS else None


def _play_speaker_apply_preview(
    *,
    preview_context: SpeakerApplyPreviewContext,
    speaker_label: str,
    segments: list[SentenceSegment],
) -> None:
    """
    Play the current speaker from the currently visible samples.

    Args:
        preview_context: Media inputs for playback.
        speaker_label: Anonymous speaker label.
        segments: Currently visible speaker segments.
    """
    try:
        _echo_preview_segments(speaker_label, segments)
        clip_path = _build_speaker_apply_audio_preview_clip(
            preview_context=preview_context,
            speaker_label=speaker_label,
            segments=segments,
        )
        command = build_audio_preview_command(media=clip_path, start_seconds=0.0)
        typer.echo(f"Starting audio preview for {speaker_label} with {len(segments)} displayed sample(s).")
        typer.echo("Controls: Space/P pauses, Q/Esc stops early, Ctrl-C also stops.")
        _run_speaker_apply_preview_command(command)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Preview failed for {speaker_label}: {exc}", err=True)


def _run_speaker_apply_preview_command(command: list[str]) -> None:
    """
    Run a preview player while handling controls in this CLI process.

    Args:
        command: Player command argv.
    """
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    stopped = False
    paused = False
    old_terminal_settings = None
    try:
        if not sys.stdin.isatty():
            returncode = process.wait()
        else:
            old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            while process.poll() is None:
                key = _read_preview_control_key()
                if key in {"q", "Q", "\x1b"}:
                    stopped = True
                    _terminate_preview_process(process, paused=paused)
                    break
                if key in {" ", "p", "P"}:
                    paused = _toggle_preview_pause(process, paused)
            returncode = process.wait()
    except KeyboardInterrupt:
        stopped = True
        _terminate_preview_process(process, paused=paused)
        returncode = process.wait()
    finally:
        if old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_settings)
    if stopped:
        typer.echo("Preview stopped.")
        return
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _read_preview_control_key() -> str | None:
    """
    Read one preview control key without blocking playback.

    Returns:
        Pressed key, or ``None`` when no key is available.
    """
    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
    if not ready:
        return None
    return sys.stdin.read(1)


def _toggle_preview_pause(process: subprocess.Popen, paused: bool) -> bool:
    """
    Pause or resume the preview process.

    Args:
        process: Running preview process.
        paused: Current pause state.

    Returns:
        Updated pause state.
    """
    if paused:
        os.kill(process.pid, signal.SIGCONT)
        typer.echo("Preview resumed.")
        return False
    os.kill(process.pid, signal.SIGSTOP)
    typer.echo("Preview paused. Press Space/P to resume, Q/Esc to stop.")
    return True


def _terminate_preview_process(process: subprocess.Popen, *, paused: bool) -> None:
    """
    Terminate a preview process.

    Args:
        process: Running preview process.
        paused: Whether the process is currently stopped.
    """
    if process.poll() is not None:
        return
    if paused:
        os.kill(process.pid, signal.SIGCONT)
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def _echo_preview_segments(speaker_label: str, segments: list[SentenceSegment]) -> None:
    """
    Print the visible sample batch that will be played.

    Args:
        speaker_label: Anonymous speaker label.
        segments: Currently visible speaker segments.
    """
    if len(segments) == 1:
        typer.echo(f"Preview sample for {speaker_label}: {_render_preview_target_sample(segments[0])}")
        return
    typer.echo(f"Preview samples for {speaker_label}:")
    for segment in segments:
        typer.echo(f"  - {_render_preview_target_sample(segment)}")


def _build_speaker_apply_audio_preview_clip(
    *,
    preview_context: SpeakerApplyPreviewContext,
    speaker_label: str,
    segments: list[SentenceSegment],
) -> Path:
    """
    Build one temporary WAV containing the visible sample batch.

    Args:
        preview_context: Media inputs for playback.
        speaker_label: Anonymous speaker label.
        segments: Currently visible speaker segments.

    Returns:
        Temporary WAV path.
    """
    if not segments:
        raise RuntimeError("No transcript samples found for this speaker.")
    clip_dir = _speaker_apply_preview_clip_dir(preview_context.project_root, speaker_label)
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for index, segment in enumerate(segments, start=1):
        target = _speaker_apply_preview_target([segment])
        clip_path = clip_dir / f"clip_{index:03d}.wav"
        extract_audio_clip(
            preview_context.video,
            clip_path,
            start_seconds=target.start_seconds,
            duration_seconds=target.duration_seconds,
        )
        clip_paths.append(clip_path)
    output_path = clip_dir / "preview.wav"
    _concatenate_wav_clips(clip_paths, output_path)
    return output_path


def _speaker_apply_preview_clip_dir(project_root: Path, speaker_label: str) -> Path:
    """
    Return the temporary clip directory for one speaker prompt.

    Args:
        project_root: Project root.
        speaker_label: Anonymous speaker label.

    Returns:
        Temporary clip directory.
    """
    safe_label = "".join(char if char.isalnum() else "_" for char in speaker_label).strip("_") or "speaker"
    return project_root / "tmp" / "speaker_apply_preview" / safe_label


def _concatenate_wav_clips(clip_paths: list[Path], output_path: Path) -> Path:
    """
    Concatenate WAV clips with a small silent gap.

    Args:
        clip_paths: WAV clips in playback order.
        output_path: Combined WAV output path.

    Returns:
        Combined WAV path.
    """
    if not clip_paths:
        raise RuntimeError("No audio preview clips were generated.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_format: tuple[int, int, int, str, str] | None = None
    with wave.open(str(output_path), "wb") as writer:
        for index, clip_path in enumerate(clip_paths):
            with wave.open(str(clip_path), "rb") as reader:
                params = reader.getparams()
                current_format = (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    params.comptype,
                    params.compname,
                )
                if expected_format is None:
                    writer.setparams(params)
                    expected_format = current_format
                elif current_format != expected_format:
                    raise RuntimeError(f"Audio preview clip format mismatch: {clip_path}")
                writer.writeframes(reader.readframes(reader.getnframes()))
                if index + 1 < len(clip_paths):
                    _write_wav_silence(writer, params)
    return output_path


def _write_wav_silence(
    writer: wave.Wave_write,
    params,
    seconds: float = SPEAKER_APPLY_PREVIEW_GAP_SECONDS,
) -> None:
    """
    Append silence to a WAV writer.

    Args:
        writer: Open WAV writer.
        params: WAV params from the current clip.
        seconds: Silence duration.
    """
    frame_count = int(round(params.framerate * seconds))
    writer.writeframes(b"\x00" * frame_count * params.nchannels * params.sampwidth)


def _speaker_apply_preview_target(segments: list[SentenceSegment]) -> SpeakerApplyPreviewTarget:
    """
    Select a finite preview target for the current speaker.

    Args:
        segments: Ordered speaker segments.

    Returns:
        Playback target with a bounded duration.
    """
    if not segments:
        raise RuntimeError("No transcript samples found for this speaker.")
    segment = _speaker_apply_preview_segment(segments)
    start_seconds = max(0.0, segment.begin_time_ms / 1000.0 - SPEAKER_APPLY_PREVIEW_LEAD_SECONDS)
    end_seconds = segment.end_time_ms / 1000.0 + SPEAKER_APPLY_PREVIEW_TAIL_SECONDS
    duration_seconds = min(
        SPEAKER_APPLY_PREVIEW_MAX_SECONDS,
        max(SPEAKER_APPLY_PREVIEW_MIN_SECONDS, end_seconds - start_seconds),
    )
    return SpeakerApplyPreviewTarget(segment, start_seconds, duration_seconds)


def _speaker_apply_preview_segment(segments: list[SentenceSegment]) -> SentenceSegment:
    """
    Pick the most useful sample segment for speaker preview.

    Args:
        segments: Ordered speaker segments.

    Returns:
        Longest segment, preferring richer text and earlier timeline position on ties.
    """
    return max(
        segments,
        key=lambda segment: (_segment_duration_ms(segment), len(segment.text.strip()), -segment.begin_time_ms),
    )


def _segment_duration_ms(segment: SentenceSegment) -> int:
    """
    Return a non-negative segment duration.

    Args:
        segment: Transcript segment.

    Returns:
        Duration in milliseconds.
    """
    return max(0, segment.end_time_ms - segment.begin_time_ms)


def _render_preview_target_sample(segment: SentenceSegment) -> str:
    """
    Render the selected preview segment for terminal confirmation.

    Args:
        segment: Selected transcript segment.

    Returns:
        Compact timestamped transcript text.
    """
    start = format_ms_timestamp(segment.begin_time_ms)
    end = format_ms_timestamp(segment.end_time_ms)
    return f"[{start} - {end}] {_trim_prompt_sample(segment.text, limit=120)}"


def _show_more_speaker_samples(
    speaker_label: str,
    segments: list[SentenceSegment],
    offset: int,
    sample_count: int,
) -> tuple[int, list[SentenceSegment]]:
    """
    Print the next batch of samples for a speaker.

    Args:
        speaker_label: Anonymous speaker label.
        segments: All transcript segments for this speaker.
        offset: Current sample offset.
        sample_count: Maximum samples to show.

    Returns:
        Updated sample offset and newly displayed samples.
    """
    samples = segments[offset : offset + sample_count]
    if not samples:
        typer.echo(f"No more samples for {speaker_label}.")
        return offset, []
    typer.echo("")
    typer.echo(f"More samples for {speaker_label}:")
    typer.echo(_render_speaker_samples(samples))
    return offset + len(samples), samples


def _remember_prompt_history(value: str) -> None:
    """
    Add a command to interactive input history when readline is available.

    Args:
        value: Prompt input value to remember.
    """
    if not value:
        return
    try:
        import readline
    except ImportError:
        return
    try:
        length = readline.get_current_history_length()
        if length > 0 and readline.get_history_item(length) == value:
            return
        readline.add_history(value)
    except (AttributeError, OSError):
        return


def _speaker_segments_by_id(result: TranscriptResult) -> dict[int, list[SentenceSegment]]:
    """
    Group non-empty transcript segments by speaker id.

    Args:
        result: Loaded transcript result.

    Returns:
        Speaker id to ordered transcript segments.
    """
    grouped: dict[int, list[SentenceSegment]] = {}
    for segment in result.sentences:
        if segment.speaker_id is None or not segment.text.strip():
            continue
        grouped.setdefault(segment.speaker_id, []).append(segment)
    return grouped


def _render_speaker_samples(samples: list[SentenceSegment]) -> str:
    """
    Render sample transcript segments.

    Args:
        samples: Transcript segments to render.

    Returns:
        Terminal text.
    """
    lines = []
    for segment in samples:
        start = format_ms_timestamp(segment.begin_time_ms)
        end = format_ms_timestamp(segment.end_time_ms)
        text = _trim_prompt_sample(segment.text)
        lines.append(f"  - [{start} - {end}] {text}")
    return "\n".join(lines)


def _trim_prompt_sample(text: str, *, limit: int = 90) -> str:
    """
    Trim a sample line for prompt display.

    Args:
        text: Segment text.
        limit: Maximum output length.

    Returns:
        Single-line preview text.
    """
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."


def _load_existing_speaker_mapping(project_dir: Path) -> dict[int, str]:
    """
    Load an existing speaker map as prompt defaults.

    Args:
        project_dir: Project root.

    Returns:
        Existing speaker mapping, or an empty mapping.
    """
    map_path = project_paths(project_dir).speakers_dir / "speaker_map.json"
    if not map_path.exists():
        return {}
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


def _load_speaker_match_defaults(project_dir: Path) -> dict[int, str]:
    """
    Load accepted speaker match suggestions as prompt defaults.

    Args:
        project_dir: Project root.

    Returns:
        Speaker id to matched speaker name.
    """
    match_path = project_paths(project_dir).speakers_dir / "speaker_matches.json"
    if not match_path.exists():
        return {}
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    return {
        int(item["speaker_id"]): str(item["name"])
        for item in matches
        if item.get("accepted") and item.get("name") is not None
    }


def _load_speaker_match_summaries(project_dir: Path, speaker_mapping: dict[int, str]) -> dict[int, str]:
    """
    Load speaker match results for inspect output.

    Args:
        project_dir: Project root.
        speaker_mapping: Existing confirmed speaker mapping.

    Returns:
        Speaker id to display-safe match summary.
    """
    match_path = project_paths(project_dir).speakers_dir / "speaker_matches.json"
    if not match_path.exists():
        return {}
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    summaries: dict[int, str] = {}
    for item in matches:
        if not isinstance(item, dict) or "speaker_id" not in item:
            continue
        speaker_id = int(item["speaker_id"])
        summaries[speaker_id] = _speaker_match_summary(item, mapped_name=speaker_mapping.get(speaker_id))
    return summaries


def _speaker_match_summary(item: dict[str, object], *, mapped_name: str | None = None) -> str:
    """
    Format one voiceprint match row for humans.

    Args:
        item: Raw match JSON item.
        mapped_name: Existing confirmed speaker name.

    Returns:
        Short match summary.
    """
    name = str(item.get("name") or "unknown")
    status = "accepted" if item.get("accepted") else "review"
    score = _safe_float(item.get("score"))
    suffix = " CONFLICT" if _speaker_match_conflicts(item, name, mapped_name) else ""
    if score is None:
        return f"{name} {status}{suffix}"
    return f"{name} score={score:.3f} {status}{suffix}"


def _speaker_match_conflicts(item: dict[str, object], match_name: str, mapped_name: str | None) -> bool:
    """
    Return whether a voiceprint match conflicts with a confirmed mapping.

    Args:
        item: Raw match JSON item.
        match_name: Display name from the match row.
        mapped_name: Existing confirmed speaker name.

    Returns:
        ``True`` when both sides name different real speakers.
    """
    if not mapped_name or match_name == "unknown":
        return False
    if not item.get("accepted"):
        return False
    if mapped_name == str(item.get("label") or ""):
        return False
    return mapped_name != match_name


def _safe_float(value: object) -> float | None:
    """
    Convert a JSON value to float when possible.

    Args:
        value: Raw JSON value.

    Returns:
        Float value, or ``None`` when conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
