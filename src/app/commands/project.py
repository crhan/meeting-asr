"""Project-oriented CLI commands."""

from __future__ import annotations

import json
import shlex
import subprocess
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
from app.speaker_review import build_preview_command, preview_start_seconds, render_speaker_summary
from app.srt_compare import build_report, parse_srt
from app.utils import configure_logging, format_ms_timestamp, safe_write_text

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
speakers_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(speakers_app, name="speakers", help="Review and name project speakers.")
app.add_typer(transcript_commands.app, name="transcript", help="View project transcript artifacts.")

MORE_SAMPLES_COMMAND = "/more"


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
    speaker_matches = run_with_cli_errors(lambda: _load_speaker_match_summaries(project_dir))
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
    typer.echo("Enter a name for each speaker. Type /more to show more samples.")
    typer.echo("Press Enter to keep the default label.")
    existing = _load_existing_speaker_mapping(project_dir)
    matched = _load_speaker_match_defaults(project_dir)
    segments = _speaker_segments_by_id(result)
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
            next_offset=sample_count,
            sample_count=sample_count,
        )
    return resolved


def _prompt_speaker_name(
    *,
    speaker_label: str,
    default_name: str,
    segments: list[SentenceSegment],
    next_offset: int,
    sample_count: int,
) -> str:
    """
    Prompt for one speaker name, allowing more samples on demand.

    Args:
        speaker_label: Anonymous speaker label.
        default_name: Existing or anonymous default name.
        segments: All transcript segments for this speaker.
        next_offset: First segment offset not yet displayed.
        sample_count: Number of samples per extra batch.

    Returns:
        Confirmed speaker name.
    """
    offset = next_offset
    while True:
        prompt = f"Name for {speaker_label} ({MORE_SAMPLES_COMMAND} for more samples)"
        name = typer.prompt(prompt, default=default_name).strip()
        if name != MORE_SAMPLES_COMMAND:
            return name or default_name
        _remember_prompt_history(MORE_SAMPLES_COMMAND)
        offset = _show_more_speaker_samples(speaker_label, segments, offset, sample_count)


def _show_more_speaker_samples(
    speaker_label: str,
    segments: list[SentenceSegment],
    offset: int,
    sample_count: int,
) -> int:
    """
    Print the next batch of samples for a speaker.

    Args:
        speaker_label: Anonymous speaker label.
        segments: All transcript segments for this speaker.
        offset: Current sample offset.
        sample_count: Maximum samples to show.

    Returns:
        Updated sample offset.
    """
    samples = segments[offset : offset + sample_count]
    if not samples:
        typer.echo(f"No more samples for {speaker_label}.")
        return offset
    typer.echo("")
    typer.echo(f"More samples for {speaker_label}:")
    typer.echo(_render_speaker_samples(samples))
    return offset + len(samples)


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


def _load_speaker_match_summaries(project_dir: Path) -> dict[int, str]:
    """
    Load speaker match results for inspect output.

    Args:
        project_dir: Project root.

    Returns:
        Speaker id to display-safe match summary.
    """
    match_path = project_paths(project_dir).speakers_dir / "speaker_matches.json"
    if not match_path.exists():
        return {}
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    return {
        int(item["speaker_id"]): _speaker_match_summary(item)
        for item in matches
        if isinstance(item, dict) and "speaker_id" in item
    }


def _speaker_match_summary(item: dict[str, object]) -> str:
    """
    Format one voiceprint match row for humans.

    Args:
        item: Raw match JSON item.

    Returns:
        Short match summary.
    """
    name = str(item.get("name") or "unknown")
    status = "accepted" if item.get("accepted") else "review"
    score = _safe_float(item.get("score"))
    if score is None:
        return f"{name} {status}"
    return f"{name} score={score:.3f} {status}"


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
