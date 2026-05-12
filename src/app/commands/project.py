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

from app.commands import project_correct as project_correct_commands
from app.commands import project_trash as project_trash_commands
from app.commands import transcript as transcript_commands
from app.core.progress import CliProgressReporter, emit_progress
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json
from app.presentation.cli.output import should_enable_verbose_logs
from app.presentation.cli.progress import run_with_progress
from app.presentation.cli.project_payloads import project_list_payload, project_status_payload
from app.presentation.cli.project_list import render_project_list
from app.presentation.cli.project_show import ProjectShowView, render_project_show
from app.presentation.cli.project_run_summary import ProjectRunSummaryView, render_project_run_summary
from app.presentation.cli.speaker_match_table import speaker_match_rows
from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper
from app.core.project_workflow import (
    project_outputs_text,
    project_workflow_summary,
)
from app.correction_types import CorrectionEditOptions, CorrectionEditSummary
from app.transcript_corrections import apply_lexicon_corrections
from app.completion_helpers import (
    complete_asr_hotwords,
    complete_audio_format,
    complete_model,
    complete_oss_upload_mode,
    complete_voiceprint_model,
)
from app.config import get_default_projects_dir
from app.asr_pricing import AsrCostEstimate, format_asr_cost
from app.core.project_models import (
    ProjectCreateSummary,
    ProjectDeleteSummary,
    ProjectManifest,
    ProjectMeetingSummary,
    ProjectTranscribeOptions,
    ProjectTranscribeSummary,
    ProjectUpdateSummary,
)
from app.core.project_refs import list_projects, resolve_project_ref
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment, TranscriptResult
from app.project_manager import (
    apply_project_speakers,
    create_or_reuse_project,
    delete_project,
    init_project_git,
    load_manifest,
    parse_mapping_items,
    prepare_project_audio,
    project_paths,
    record_project_stage,
    resolve_project_source_path,
    save_manifest,
    summarize_project,
    transcribe_project,
    update_project_metadata,
)
from app.presentation.tui.project import (
    load_project_picker_session,
    render_project_picker_summary,
    run_project_picker_tui,
)
from app.sentence_reassignment import (
    SentenceReassignmentApplyResult,
    apply_project_sentence_reassignments,
)
from app.speaker_labeling import (
    SentenceReassignmentSpec,
    build_speaker_summaries,
    load_project_ignored_speakers,
    load_transcript_result,
)
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_IGNORED,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    effective_match_status,
    match_threshold,
    speaker_id_from_match,
    voiceprint_match_status,
)
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers
from app.speaker_review import (
    build_audio_preview_command,
    build_preview_command,
    preview_start_seconds,
    render_speaker_summary,
)
from app.presentation.tui.speaker import (
    SpeakerReviewDecision,
    load_speaker_review_session,
    run_speaker_review_tui,
)
from app.presentation.tui.speaker_save import SpeakerReviewSaveOutcome
from app.presentation.tui.speaker_summary import render_speaker_review_summary
from app.srt_compare import build_report, parse_srt
from app.utils import configure_logging, format_ms_timestamp, safe_write_text

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
speakers_app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(speakers_app, name="speakers", help="Review and name project speakers.", context_settings=HELP_CONTEXT)
app.add_typer(
    transcript_commands.app,
    name="transcript",
    help="View project transcript artifacts.",
    context_settings=HELP_CONTEXT,
)
app.add_typer(
    project_trash_commands.app,
    name="trash",
    help="Restore or permanently remove deleted projects.",
    context_settings=HELP_CONTEXT,
)
app.add_typer(
    project_correct_commands.app,
    name="correct",
    help="Review and apply transcript correction proposals.",
    context_settings=HELP_CONTEXT,
)

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


@dataclass(frozen=True, slots=True)
class ProjectRunSummary:
    """Summary for the full project run workflow."""

    project: ProjectCreateSummary
    transcription: ProjectTranscribeSummary
    meeting_summary: ProjectMeetingSummary | None
    correction_summary: CorrectionEditSummary | None
    lexicon_correction_summary: CorrectionEditSummary | None
    matches: SpeakerMatchSummary
    applied_mapping: dict[int, str]


@dataclass(frozen=True, slots=True)
class ProjectReviewCorrectionOptions:
    """Options used when project review launches transcript correction."""

    edit_options: CorrectionEditOptions
    yes: bool
    store_dir: Path | None = None


@app.command("create")
def create(
    input: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
    hash_source: bool = typer.Option(
        False,
        "--hash-source/--no-hash-source",
        help="Deprecated compatibility flag. Source identity is always hashed.",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Create a project directory with project.json metadata."""
    summary = run_with_progress(
        lambda reporter: create_or_reuse_project(
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
    _echo_project_created(summary.project_dir, summary.manifest, created=summary.created)


@app.command("prepare")
def prepare(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Extract project audio without starting cloud transcription."""
    configure_logging(verbose=should_enable_verbose_logs())
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    audio_path = run_with_progress(
        lambda reporter: prepare_project_audio(resolved_project_dir, audio_format=audio_format, progress=reporter),
        description="Preparing project audio",
        total=1,
        enabled=progress,
    )
    typer.echo(f"Audio written to: {audio_path}")


@app.command("transcribe")
def transcribe(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    speaker_count: Optional[int] = typer.Option(None, "--speaker-count", min=1),
    language: Optional[str] = typer.Option("zh,en", "--language"),
    model: str = typer.Option("fun-asr", "--model", autocompletion=complete_model),
    oss_upload: str = typer.Option("auto", "--oss-upload", autocompletion=complete_oss_upload_mode),
    file_url: Optional[str] = typer.Option(None, "--file-url"),
    generate_srt: bool = typer.Option(True, "--generate-srt/--no-generate-srt"),
    timestamp_alignment: bool = typer.Option(True, "--timestamp-alignment/--no-timestamp-alignment"),
    disfluency_removal: bool = typer.Option(False, "--disfluency-removal/--no-disfluency-removal"),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
    asr_hotwords: str = typer.Option("auto", "--asr-hotwords", autocompletion=complete_asr_hotwords),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Transcribe a project and write structured artifacts."""
    configure_logging(verbose=should_enable_verbose_logs())
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
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
        asr_hotwords=asr_hotwords,
    )
    summary = run_with_progress(
        lambda reporter: transcribe_project(resolved_project_dir, options, progress=reporter),
        description="Transcribing project",
        total=7,
        enabled=progress,
    )
    _echo_transcribe_summary(
        summary.project_dir,
        summary.task_id,
        summary.detected_speaker_count,
        summary.sentence_count,
        summary.cost,
    )


@app.command("summarize")
def summarize(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    model: Optional[str] = typer.Option(None, "--model", help="DashScope text model for meeting memory index."),
    update_title: bool = typer.Option(True, "--update-title/--no-update-title"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Generate a meeting title and memory index from a transcribed project."""
    configure_logging(verbose=should_enable_verbose_logs())
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    summary = run_with_progress(
        lambda reporter: summarize_project(
            resolved_project_dir,
            model=model,
            update_title=update_title,
            progress=reporter,
        ),
        description="Summarizing project",
        total=1,
        enabled=progress,
    )
    manifest = load_manifest(resolved_project_dir)
    typer.echo("Project memory index completed.")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Title: {manifest.title}")
    typer.echo(f"Memory index: {summary.summary_path.resolve()}")
    typer.echo(f"Memory index JSON: {summary.json_path.resolve()}")


@app.command("run")
def run(
    input: Path = typer.Argument(..., metavar="INPUT", exists=True, file_okay=True, dir_okay=False),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", file_okay=False, dir_okay=True),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
    speaker_count: Optional[int] = typer.Option(None, "--speaker-count", min=1),
    language: Optional[str] = typer.Option("zh,en", "--language"),
    model: str = typer.Option("fun-asr", "--model", autocompletion=complete_model),
    oss_upload: str = typer.Option("auto", "--oss-upload", autocompletion=complete_oss_upload_mode),
    file_url: Optional[str] = typer.Option(None, "--file-url"),
    audio_format: str = typer.Option("flac", "--audio-format", autocompletion=complete_audio_format),
    asr_hotwords: str = typer.Option("auto", "--asr-hotwords", autocompletion=complete_asr_hotwords),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    voiceprint_model: Optional[str] = typer.Option(
        None,
        "--voiceprint-model",
        autocompletion=complete_voiceprint_model,
    ),
    match_threshold: float = typer.Option(0.75, "--match-threshold", min=0.0, max=1.0),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Generate title and memory index after ASR."),
    summary_model: Optional[str] = typer.Option(None, "--summary-model", help="DashScope model for meeting memory index."),
    polish: bool = typer.Option(True, "--polish/--no-polish", help="Generate transcript polish proposal after ASR."),
    local_correction: bool = typer.Option(
        True,
        "--local-correction/--no-local-correction",
        help="Apply accepted local lexicon corrections after ASR.",
    ),
    correction_model: Optional[str] = typer.Option(None, "--correction-model", help="DashScope model for transcript polish."),
    polish_concurrency: Optional[int] = typer.Option(
        None,
        "--polish-concurrency",
        min=1,
        max=64,
        help="Parallel DashScope batch requests for transcript polish.",
    ),
    polish_legacy: bool = typer.Option(
        False,
        "--legacy-polish",
        help="Use the legacy aggressive-rewrite polish prompt (pre-2026 behavior). "
        "Default is the strict downstream-summary-friendly polish.",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
    agent_log: bool = typer.Option(
        False,
        "--agent-log/--no-agent-log",
        help="Print structured stage and heartbeat logs for agents; combine with --no-progress for clean logs.",
    ),
) -> None:
    """Create a project, transcribe, summarize, and match speakers automatically."""
    configure_logging(verbose=should_enable_verbose_logs())
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
        asr_hotwords=asr_hotwords,
    )
    summary = run_with_progress(
        lambda reporter: _run_project_workflow(
            input,
            title=title,
            projects_dir=projects_dir,
            project_dir=project_dir,
            meeting_time=meeting_time,
            options=options,
            store_dir=store_dir,
            voiceprint_model=voiceprint_model,
            match_threshold=match_threshold,
            summarize=summarize,
            summary_model=summary_model,
            polish=polish,
            local_correction=local_correction,
            correction_model=correction_model,
            polish_concurrency=polish_concurrency,
            polish_legacy=polish_legacy,
            progress=reporter,
        ),
        description="Running project workflow",
        enabled=progress,
        structured_log=agent_log,
    )
    _echo_run_summary(summary)


@app.command("list")
def list_command(
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    plain: bool = typer.Option(False, "--plain", help="Print stable tab-separated output."),
) -> None:
    """List projects under the XDG project store."""
    result = run_with_cli_errors(lambda: list_projects(projects_dir))
    if as_json:
        emit_json(project_list_payload(result.projects_dir, result.projects))
        return
    render_project_list(result.projects_dir, result.projects, plain=plain)


@app.command("show")
def show(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show a human-friendly project overview and where outputs live."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    manifest = run_with_cli_errors(lambda: load_manifest(resolved_project_dir))
    paths = project_paths(resolved_project_dir)
    if as_json:
        emit_json(project_status_payload(paths, manifest))
        return
    workflow = project_workflow_summary(paths.root, manifest, project_ref=manifest.project_id)
    render_project_show(ProjectShowView(paths.root, manifest.project_id, manifest, workflow))


@app.command("update")
def update(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    meeting_time: Optional[str] = typer.Option(None, "--meeting-time"),
) -> None:
    """Update editable project metadata."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    summary = run_with_cli_errors(
        lambda: update_project_metadata(
            resolved_project_dir,
            title=title,
            meeting_time=meeting_time,
        )
    )
    _echo_project_updated(summary)


@app.command("delete")
def delete(
    project_dir: Path = typer.Argument(..., metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation."),
    permanent: bool = typer.Option(False, "--permanent", help="Physically remove instead of moving to trash."),
) -> None:
    """Delete a project. By default the project is moved to Meeting-ASR trash."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    manifest = run_with_cli_errors(lambda: load_manifest(resolved_project_dir))
    if not yes and not _confirm_project_delete(resolved_project_dir, manifest, permanent=permanent):
        typer.echo("Project delete cancelled.")
        return
    summary = run_with_cli_errors(lambda: delete_project(resolved_project_dir, permanent=permanent))
    _echo_project_deleted(summary)


@app.command("review")
def review(
    project: Optional[str] = typer.Argument(None, metavar="PROJECT"),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    page_size: Optional[int] = typer.Option(
        None,
        "--page-size",
        min=1,
        max=50,
        help="Override samples per page. By default the TUI uses the pane height.",
    ),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    summary: bool = typer.Option(False, "--summary", help="Print without opening a TUI."),
    editor: Optional[str] = typer.Option(None, "--editor", help="Editor command used by transcript correction."),
    no_ai: bool = typer.Option(False, "--no-ai", help="Disable DashScope correction proposals in TUI text correction."),
    no_proposal_open: bool = typer.Option(False, "--no-proposal-open", help="Do not open the generated correction proposal."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept generated correction proposals without prompting."),
    model: Optional[str] = typer.Option(None, "--model", help="DashScope correction model id."),
    category: str = typer.Option("unknown", "--category", help="Category for learned correction terms."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    from_original: bool = typer.Option(False, "--from-original", help="Correct from the original ASR transcript."),
) -> None:
    """Open the recommended human review workflow for project outputs, including unresolved speaker names."""
    project_dir = _resolve_review_project(project, projects_dir, summary=summary)
    if project_dir is None:
        return
    correction_options = ProjectReviewCorrectionOptions(
        edit_options=CorrectionEditOptions(
            editor=editor,
            review_file=None,
            open_editor=True,
            open_proposal=not no_proposal_open,
            category=category,
            lexicon_db=lexicon_db,
            from_original=from_original,
            use_ai=not no_ai,
            model=model,
        ),
        yes=yes,
        store_dir=store_dir,
    )
    _run_speaker_review(
        project_dir,
        page_size=page_size,
        store_dir=store_dir,
        summary=summary,
        speaker_only=False,
        correction_options=correction_options,
    )


def _run_project_workflow(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    options: ProjectTranscribeOptions,
    store_dir: Path | None,
    voiceprint_model: str | None,
    match_threshold: float,
    summarize: bool,
    summary_model: str | None,
    polish: bool,
    local_correction: bool,
    correction_model: str | None,
    polish_concurrency: int | None,
    polish_legacy: bool = False,
    progress: CliProgressReporter | None = None,
) -> ProjectRunSummary:
    """
    Create or reuse a project, then run the automatic workflow.

    Args:
        input_path: Local source media file.
        title: Optional human title.
        projects_dir: Optional parent directory.
        project_dir: Optional explicit project directory.
        meeting_time: Optional meeting start time string.
        options: Project transcription options.
        store_dir: Optional voiceprint store directory.
        voiceprint_model: Optional voiceprint model override.
        match_threshold: Voiceprint match acceptance threshold.
        summarize: Generate meeting memory index when true.
        summary_model: Optional DashScope text model override.
        local_correction: Apply active local lexicon rules when true.
        polish_concurrency: Optional transcript polish request concurrency override.
        progress: Optional progress reporter.

    Returns:
        Project run summary.
    """
    step_total = 7 + int(local_correction) + int(polish) + int(summarize) + 2
    step_descriptions = _project_run_step_descriptions(summarize, polish, local_correction)
    emit_progress(
        progress,
        "Creating or reusing project",
        step_index=1,
        step_total=step_total,
        reset_total=True,
        step_descriptions=step_descriptions,
    )
    project = create_or_reuse_project(
        input_path,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=meeting_time,
        hash_source=False,
        progress=progress,
    )
    _record_and_emit_run_stage(
        project.project_dir,
        input_path,
        "project created",
        progress,
        external_ids={"source": "created" if project.created else "reused"},
        last_success="project ready",
        description="Creating or reusing project",
        step_index=1,
        step_total=step_total,
    )
    transcription = transcribe_project(project.project_dir, options, progress=progress, step_offset=1, step_total=step_total)
    lexicon_correction_summary = None
    if local_correction:
        correction_step = 8
        _record_and_emit_run_stage(
            project.project_dir,
            input_path,
            "local correction",
            progress,
            description="Applying local vocabulary corrections",
            step_index=correction_step,
            step_total=step_total,
            reset_total=True,
        )
        lexicon_correction_summary = _apply_run_lexicon_corrections(project.project_dir, progress=progress)
        _record_local_correction_runtime(project.project_dir, lexicon_correction_summary)
    correction_summary = None
    if polish:
        polish_step = 8 + int(local_correction)
        _record_and_emit_run_stage(
            project.project_dir,
            input_path,
            "polish",
            progress,
            external_ids={
                "model": correction_model or "configured-default",
                "concurrency": polish_concurrency or "configured-default",
            },
            description="Generating transcript polish proposal",
            step_index=polish_step,
            step_total=step_total,
        )
        correction_summary = _prepare_run_transcript_polish(
            project.project_dir,
            correction_model,
            polish_concurrency=polish_concurrency,
            polish_legacy=polish_legacy,
            progress=progress,
        )
        _record_polish_runtime(project.project_dir, correction_summary)
    meeting_summary = None
    if summarize:
        summary_step = 8 + int(local_correction) + int(polish)
        _record_and_emit_run_stage(
            project.project_dir,
            input_path,
            "summary",
            progress,
            external_ids={"model": summary_model or "configured-default"},
            description="Summarizing meeting",
            step_index=summary_step,
            step_total=step_total,
            reset_total=True,
        )
        meeting_summary = summarize_project(
            project.project_dir,
            model=summary_model,
            update_title=title is None,
            progress=progress,
        )
    match_step = 8 + int(local_correction) + int(polish) + int(summarize)
    apply_step = match_step + 1
    _record_and_emit_run_stage(
        project.project_dir,
        input_path,
        "speaker match",
        progress,
        external_ids={"provider": "local-speechbrain"},
        description="Matching speakers with voiceprints",
        step_index=match_step,
        step_total=step_total,
        reset_total=True,
    )
    matches = match_project_speakers(
        project.project_dir,
        store_dir=store_dir,
        provider=None,
        model=voiceprint_model,
        threshold=match_threshold,
        sample_count=2,
        max_seconds=12.0,
        padding_seconds=0.5,
        progress=progress,
    )
    _record_and_emit_run_stage(
        project.project_dir,
        input_path,
        "final artifact write",
        progress,
        last_success="speaker matching complete",
        description="Applying accepted speaker matches",
        step_index=apply_step,
        step_total=step_total,
        reset_total=True,
    )
    applied_mapping = matches.accepted_mapping
    if applied_mapping:
        apply_project_speakers(
            project.project_dir,
            applied_mapping,
            person_mapping=matches.accepted_person_mapping,
            person_public_mapping=matches.accepted_person_public_mapping,
        )
    record_project_stage(
        project.project_dir,
        stage="complete",
        input_file=input_path,
        last_success="project run complete",
    )
    emit_progress(progress, "Project run complete", completed=1, total=1)
    return ProjectRunSummary(
        project,
        transcription,
        meeting_summary,
        correction_summary,
        lexicon_correction_summary,
        matches,
        applied_mapping,
    )


def _project_run_step_descriptions(summarize: bool, polish: bool, local_correction: bool) -> tuple[str, ...]:
    """
    Return the planned step names for ``project run``.

    Args:
        summarize: Whether the summary step is enabled.
        polish: Whether the transcript polish step is enabled.
        local_correction: Whether local lexicon correction is enabled.

    Returns:
        Ordered step descriptions.
    """
    transcription_steps = (
        "Creating or reusing project",
        "Preparing audio",
        "Resolving audio URL",
        "Submitting DashScope task",
        "Waiting for DashScope ASR",
        "Downloading transcription result",
        "Writing transcript artifacts",
    )
    steps = list(transcription_steps)
    if local_correction:
        steps.append("Applying local vocabulary corrections")
    if polish:
        steps.append("Generating transcript polish proposal")
    if summarize:
        steps.append("Summarizing meeting")
    steps.extend(("Matching speakers with voiceprints", "Applying accepted speaker matches"))
    return tuple(steps)


def _record_and_emit_run_stage(
    project_dir: Path,
    input_path: Path,
    stage: str,
    progress: CliProgressReporter | None,
    *,
    external_ids: dict[str, object] | None = None,
    last_success: str | None = None,
    description: str | None = None,
    step_index: int | None = None,
    step_total: int | None = None,
    reset_total: bool = False,
) -> ProjectManifest:
    """Persist and emit one project-run stage transition."""
    manifest = record_project_stage(
        project_dir,
        stage=stage,
        input_file=input_path,
        external_ids=external_ids,
        last_success=last_success,
    )
    emit_progress(
        progress,
        description or stage,
        step_index=step_index,
        step_total=step_total,
        reset_total=reset_total,
        log_kind="stage",
        stage=stage,
        project_id=manifest.project_id,
        project_path=str(project_dir),
        input_file=str(input_path),
        timestamp=_now_local_iso(),
        last_success=last_success,
        log_fields=tuple((external_ids or {}).items()),
    )
    return manifest


def _now_local_iso() -> str:
    """Return a local timestamp for CLI progress logs."""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _record_polish_runtime(project_dir: Path, summary: CorrectionEditSummary) -> None:
    """Persist the transcript polish business state for project show."""
    manifest = load_manifest(project_dir)
    runtime = dict(manifest.runtime)
    runtime["polish"] = _polish_runtime_payload(project_dir, summary)
    manifest.runtime = runtime
    save_manifest(project_dir, manifest)


def _polish_runtime_payload(project_dir: Path, summary: CorrectionEditSummary) -> dict[str, object]:
    """Build JSON-safe transcript polish state."""
    status = "failed" if summary.model_error else "no_changes"
    if summary.proposed_change_count:
        status = "proposal_ready"
    return {
        "status": status,
        "updated_at": _now_local_iso(),
        "model": summary.model,
        "error": summary.model_error,
        "proposed_changes": summary.proposed_change_count,
        "proposal_json": _relative_optional_path(project_dir, summary.proposal_json_path),
        "proposal_diff": _relative_optional_path(project_dir, summary.proposal_diff_path),
    }


def _apply_run_lexicon_corrections(
    project_dir: Path,
    *,
    progress: CliProgressReporter | None,
) -> CorrectionEditSummary:
    """
    Apply accepted local lexicon rules during ``project run``.

    Args:
        project_dir: Project root.
        progress: Optional progress reporter.

    Returns:
        Local correction summary.
    """
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    speaker_mapping = project_correct_commands.load_speaker_mapping_for_correction(paths.root)
    summary = apply_lexicon_corrections(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        options=CorrectionEditOptions(
            open_editor=False,
            open_proposal=False,
            category="lexicon",
            use_ai=False,
        ),
        progress=progress,
    )
    if summary.accepted and summary.change_count:
        project_correct_commands.record_correction_outputs(paths.root, manifest, summary)
        save_manifest(paths.root, manifest)
    return summary


def _record_local_correction_runtime(project_dir: Path, summary: CorrectionEditSummary) -> None:
    """Persist local lexicon correction state for project show."""
    manifest = load_manifest(project_dir)
    runtime = dict(manifest.runtime)
    runtime["local_correction"] = _local_correction_runtime_payload(project_dir, summary)
    manifest.runtime = runtime
    save_manifest(project_dir, manifest)


def _local_correction_runtime_payload(project_dir: Path, summary: CorrectionEditSummary) -> dict[str, object]:
    """Build JSON-safe local correction runtime state."""
    status = "applied" if summary.accepted and summary.change_count else "no_changes"
    return {
        "status": status,
        "updated_at": _now_local_iso(),
        "model": summary.model,
        "changed_sentences": summary.change_count,
        "rules_applied": len(summary.understanding),
        "corrected_transcript": _relative_optional_path(project_dir, summary.corrected_named_transcript_path),
        "hotwords": _relative_optional_path(project_dir, summary.hotwords_path),
        "lexicon_db": str(summary.lexicon_db) if summary.lexicon_db else None,
    }


def _relative_optional_path(project_dir: Path, path: Path | None) -> str | None:
    """Return a project-relative path when possible."""
    if path is None:
        return None
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def _prepare_run_transcript_polish(
    project_dir: Path,
    correction_model: str | None,
    *,
    polish_concurrency: int | None = None,
    polish_legacy: bool = False,
    progress: CliProgressReporter | None = None,
) -> CorrectionEditSummary:
    """
    Prepare the default transcript polish proposal used by ``project run``.

    Args:
        project_dir: Project root.
        correction_model: Optional DashScope model override.
        polish_legacy: When True, use the legacy aggressive-rewrite polish.

    Returns:
        Pending polish summary, or a no-change/error summary.
    """
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    speaker_mapping = project_correct_commands.load_speaker_mapping_for_correction(paths.root)
    options = CorrectionEditOptions(
        open_editor=False,
        open_proposal=False,
        category="polish",
        use_ai=True,
        model=correction_model,
        polish_concurrency=polish_concurrency,
        polish_legacy=polish_legacy,
    )
    return project_correct_commands.prepare_transcript_polish_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        options=options,
        progress=progress,
    )


@app.command("status")
def status(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Print a project status summary."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    manifest = run_with_cli_errors(lambda: load_manifest(resolved_project_dir))
    paths = project_paths(resolved_project_dir)
    if as_json:
        emit_json(project_status_payload(paths, manifest))
        return
    workflow = project_workflow_summary(
        paths.root,
        manifest,
        project_ref=manifest.project_id,
    )
    typer.echo(f"Project: {paths.root}")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Title: {manifest.title}")
    typer.echo(f"Meeting time: {manifest.source.meeting_time or '-'}")
    typer.echo(f"Status: {manifest.status}")
    typer.echo(f"State: {workflow.state}")
    typer.echo(f"Next: {workflow.next_command_short}")
    typer.echo(f"Artifacts: {project_outputs_text(workflow.outputs)}")
    if workflow.missing:
        typer.echo(f"Missing: {', '.join(workflow.missing)}")
    typer.echo(f"Source: {manifest.source.path}")
    if manifest.source.original_path:
        typer.echo(f"Original source: {manifest.source.original_path}")
    typer.echo(f"Audio: {manifest.audio.get('path', '-')}")
    typer.echo(f"Task ID: {manifest.asr.get('task_id', '-')}")
    runtime = manifest.runtime
    if runtime:
        typer.echo(f"Current stage: {runtime.get('current_stage', '-')}")
        typer.echo(f"Stage updated: {runtime.get('last_heartbeat_at') or runtime.get('stage_started_at') or '-'}")
        if runtime.get("external_ids"):
            typer.echo(f"External IDs: {runtime.get('external_ids')}")
        if runtime.get("last_error"):
            typer.echo(f"Last error: {runtime.get('last_error')}")
    typer.echo(f"Detected speakers: {manifest.speakers.get('detected_ids', [])}")


@app.command("git-init")
def git_init(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
) -> None:
    """Initialize optional Git tracking for human-edited project files."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    gitignore_path = run_with_cli_errors(lambda: init_project_git(resolved_project_dir))
    typer.echo(f"Git initialized: {resolved_project_dir}")
    typer.echo(f"Git ignore written to: {gitignore_path}")


@speakers_app.command("inspect")
def speakers_inspect(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    sample_count: int = typer.Option(5, "--sample-count", min=1, max=20),
) -> None:
    """Print diagnostic speaker samples; read-only, does not apply names."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    result = run_with_cli_errors(
        lambda: load_transcript_result(project_paths(resolved_project_dir).asr_dir / "sentences.json")
    )
    summaries = build_speaker_summaries(result, sample_count=sample_count)
    if not summaries:
        typer.echo("No detected speakers found in the transcript.")
        raise typer.Exit(code=1)
    speaker_mapping = run_with_cli_errors(lambda: _load_existing_speaker_mapping(resolved_project_dir))
    ignored_speakers = run_with_cli_errors(lambda: load_project_ignored_speakers(resolved_project_dir))
    speaker_matches = run_with_cli_errors(
        lambda: _load_speaker_match_summaries(
            resolved_project_dir,
            speaker_mapping,
            ignored_speaker_ids=ignored_speakers,
        )
    )
    for index, summary in enumerate(summaries):
        if index:
            typer.echo("")
        is_ignored = summary.speaker_id in ignored_speakers
        typer.echo(
            render_speaker_summary(
                summary,
                mapped_name=None if is_ignored else speaker_mapping.get(summary.speaker_id),
                match_summary=None if is_ignored else speaker_matches.get(summary.speaker_id),
                ignored=is_ignored,
            )
        )
    if _project_has_unresolved_match(resolved_project_dir, ignored_speaker_ids=ignored_speakers):
        manifest = run_with_cli_errors(lambda: load_manifest(resolved_project_dir))
        _echo_unresolved_speaker_next_steps(manifest.project_id, speaker_only=True)


@speakers_app.command("preview")
def speakers_preview(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    speaker_id: Optional[int] = typer.Option(None, "--speaker-id"),
    padding_seconds: int = typer.Option(8, "--padding-seconds", min=0),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Open the source video with the project's subtitle for speaker review."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    manifest = run_with_cli_errors(lambda: load_manifest(resolved_project_dir))
    paths = project_paths(resolved_project_dir)
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
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    mappings: list[str] = typer.Option([], "--map", help="Non-interactive speaker_id=name mapping."),
    sample_count: int = typer.Option(3, "--sample-count", min=1, max=20, help="Samples shown per speaker."),
) -> None:
    """Apply known speaker mappings non-interactively; intended for scripts or already confirmed mappings."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    sentences_path = project_paths(resolved_project_dir).asr_dir / "sentences.json"
    result = run_with_cli_errors(lambda: load_transcript_result(sentences_path))
    resolved = _resolve_speaker_mappings(
        project_dir=resolved_project_dir,
        mappings=mappings,
        sample_count=sample_count,
        known_speakers=set(result.detected_speakers),
        result=result,
    )
    mapping_path, transcript_path, srt_path = run_with_cli_errors(
        lambda: apply_project_speakers(resolved_project_dir, resolved)
    )
    typer.echo(f"Mapping written to: {mapping_path}")
    typer.echo(f"Named transcript written to: {transcript_path}")
    typer.echo(f"Named subtitle written to: {srt_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  meeting-asr project speakers preview")
    typer.echo("  meeting-asr project transcript show")
    typer.echo("  meeting-asr voiceprint review PROJECT_ID")
    typer.echo(f"  open {_shell_quote_path(transcript_path)}")


@speakers_app.command("review")
def speakers_review(
    project_dir: Path = typer.Argument(
        Path("."),
        metavar="PROJECT",
        file_okay=False,
        dir_okay=True,
    ),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    page_size: Optional[int] = typer.Option(
        None,
        "--page-size",
        min=1,
        max=50,
        help="Override samples per page. By default the TUI uses the pane height.",
    ),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print the review queue without opening the TUI.",
    ),
) -> None:
    """Open the recommended interactive speaker identity review."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    _run_speaker_review(
        resolved_project_dir,
        page_size=page_size,
        store_dir=store_dir,
        summary=summary,
        speaker_only=True,
        correction_options=None,
    )


def _run_speaker_review(
    project_dir: Path,
    *,
    page_size: int | None,
    store_dir: Path | None,
    summary: bool,
    speaker_only: bool,
    correction_options: ProjectReviewCorrectionOptions | None,
) -> None:
    """Run speaker review for one resolved project directory."""
    session = run_with_cli_errors(
        lambda: load_speaker_review_session(
            project_dir,
            page_size=page_size,
            store_dir=store_dir,
            allow_correction=correction_options is not None,
        )
    )
    if summary:
        typer.echo(render_speaker_review_summary(session, speaker_only=speaker_only))
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise typer.BadParameter(
            "Speaker review TUI requires an interactive terminal. Use --summary to inspect."
        )
    if correction_options is not None:
        def save_active_project(
            active_project_dir: Path,
            decision: SpeakerReviewDecision,
        ) -> SpeakerReviewSaveOutcome:
            """Save review state for the project currently shown in the TUI."""
            return _save_review_from_tui(active_project_dir, decision, correction_options)

        def accept_active_project(
            active_project_dir: Path,
            proposal_path: Path | None,
            selected_indices: tuple[int, ...] | None,
        ) -> SpeakerReviewSaveOutcome:
            """Accept correction proposal for the project currently shown in the TUI."""
            return _accept_review_correction_from_tui(
                active_project_dir,
                proposal_path,
                correction_options,
                selected_indices,
            )

        run_speaker_review_tui(
            session,
            project_save_handler=save_active_project,
            project_accept_handler=accept_active_project,
        )
        return
    decision = run_speaker_review_tui(session)
    _handle_speaker_review_decision(project_dir, decision, correction_options, store_dir=store_dir)


def _handle_speaker_review_decision(
    project_dir: Path,
    decision: SpeakerReviewDecision,
    correction_options: ProjectReviewCorrectionOptions | None,
    *,
    store_dir: Path | None = None,
) -> None:
    """Persist one TUI decision and run the requested follow-up action."""
    if not decision.saved:
        typer.echo("Speaker review exited without saving.")
        return
    project_dir = decision.project_dir or project_dir
    effective_store_dir = store_dir if store_dir is not None else (
        correction_options.store_dir if correction_options else None
    )
    reassignment_result = _persist_sentence_reassignments(
        project_dir, decision, store_dir=effective_store_dir
    )
    mapping_path, transcript_path, srt_path = run_with_cli_errors(
        lambda: apply_project_speakers(
            project_dir,
            decision.mapping,
            person_mapping=decision.person_mapping,
            person_public_mapping=decision.person_public_mapping,
            ignored_speaker_ids=decision.ignored_speaker_ids,
        )
    )
    typer.echo(f"Mapping written to: {mapping_path}")
    typer.echo(f"Named transcript written to: {transcript_path}")
    typer.echo(f"Named subtitle written to: {srt_path}")
    _echo_reassignment_result(reassignment_result)
    if decision.action == "correct-inline":
        _run_review_inline_correction(project_dir, decision, correction_options)
        return
    if decision.action == "correct":
        _run_review_transcript_correction(project_dir, correction_options)
        return
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  meeting-asr project speakers preview")
    typer.echo("  meeting-asr project transcript show")
    typer.echo("  meeting-asr voiceprint review PROJECT_ID")
    typer.echo("  meeting-asr voiceprint embed")


def _echo_reassignment_result(result: SentenceReassignmentApplyResult | None) -> None:
    """Print a compact summary when sentence reassignments were applied."""
    if result is None:
        return
    typer.echo("")
    typer.echo(f"Sentence reassignments applied to: {', '.join(str(path) for path in result.sentence_files)}")
    if result.anonymous_transcript_path is not None:
        typer.echo(f"Anonymous transcript regenerated: {result.anonymous_transcript_path}")
    if result.deleted_samples:
        typer.echo(
            f"Voiceprint samples invalidated: {len(result.deleted_samples)} (re-capture from voiceprint review)"
        )
    if result.match_summary is not None:
        typer.echo(f"Voiceprint matches refreshed: {result.match_summary.match_path}")
    elif result.rematch_skipped_reason is not None:
        typer.echo(f"Voiceprint rematch skipped: {result.rematch_skipped_reason}")


def _save_review_from_tui(
    project_dir: Path,
    decision: SpeakerReviewDecision,
    correction_options: ProjectReviewCorrectionOptions,
) -> SpeakerReviewSaveOutcome:
    """Persist project review state from inside the TUI."""
    reassignment_result = _persist_sentence_reassignments(
        project_dir, decision, store_dir=correction_options.store_dir
    )
    mapping_path, transcript_path, srt_path = apply_project_speakers(
        project_dir,
        decision.mapping,
        person_mapping=decision.person_mapping,
        person_public_mapping=decision.person_public_mapping,
        ignored_speaker_ids=decision.ignored_speaker_ids,
    )
    correction_summary = None
    if decision.action == "correct-inline":
        correction_summary = _prepare_review_correction_from_tui(project_dir, decision, correction_options)
        if correction_options.yes and correction_summary.proposal_json_path is not None:
            correction_summary = _accept_review_correction_from_tui(
                project_dir,
                correction_summary.proposal_json_path,
                correction_options,
            ).correction_summary
    return SpeakerReviewSaveOutcome(
        mapping_path,
        transcript_path,
        srt_path,
        correction_summary,
        reassignment_result=reassignment_result,
    )


def _prepare_review_correction_from_tui(
    project_dir: Path,
    decision: SpeakerReviewDecision,
    correction_options: ProjectReviewCorrectionOptions,
) -> CorrectionEditSummary:
    """Prepare a pending correction proposal from TUI sentence edits."""
    correction_edits = _decision_correction_edits(decision)
    if not correction_edits:
        raise RuntimeError("No transcript correction edits were staged.")
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    speaker_mapping = project_correct_commands.load_speaker_mapping_for_correction(paths.root)
    return project_correct_commands.prepare_inline_corrections_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=correction_edits,
        options=correction_options.edit_options,
    )


def _accept_review_correction_from_tui(
    project_dir: Path,
    proposal_path: Path | None,
    correction_options: ProjectReviewCorrectionOptions,
    selected_change_indices: tuple[int, ...] | None = None,
) -> SpeakerReviewSaveOutcome:
    """Accept a pending correction proposal from inside the TUI."""
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    speaker_mapping = project_correct_commands.load_speaker_mapping_for_correction(paths.root)
    summary = project_correct_commands.accept_correction_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        proposal_path=proposal_path,
        lexicon_db=correction_options.edit_options.lexicon_db,
        selected_change_indices=selected_change_indices,
    )
    return SpeakerReviewSaveOutcome(None, None, None, summary)


def _persist_sentence_reassignments(
    project_dir: Path,
    decision: SpeakerReviewDecision,
    *,
    store_dir: Path | None = None,
    rematch: bool = True,
) -> SentenceReassignmentApplyResult | None:
    """Apply pending reassignments and refresh every dependent artifact.

    Args:
        project_dir: Project root directory.
        decision: Save decision returned by the speaker review TUI.
        store_dir: Optional voiceprint store directory.
        rematch: Whether to rerun voiceprint matching after invalidation.

    Returns:
        Apply result describing rewritten sentence files, dropped voiceprint
        samples, and the new match summary; ``None`` when there were no
        reassignments to apply.

    Notes:
        Reassignments rewrite ``sentences.json`` (and ``sentences_corrected.json``
        when present), regenerate ``exports/transcript_speakers.txt``, drop
        voiceprint samples whose audio now belongs to another speaker, and
        rerun ``speaker_matches.json``. The named transcript and SRT remain
        the responsibility of the caller's ``apply_project_speakers`` step.
    """
    if not decision.sentence_reassignments:
        return None
    specs = [
        SentenceReassignmentSpec(
            sentence_id=item.sentence_id,
            begin_time_ms=item.begin_time_ms,
            end_time_ms=item.end_time_ms,
            new_speaker_id=item.new_speaker_id,
            original_speaker_id=item.original_speaker_id,
        )
        for item in decision.sentence_reassignments
    ]
    return run_with_cli_errors(
        lambda: apply_project_sentence_reassignments(
            project_dir,
            specs,
            store_dir=store_dir,
            rematch=rematch,
        )
    )


def _decision_correction_edits(decision: SpeakerReviewDecision) -> list[object]:
    """Return all staged correction edits from a review decision."""
    if decision.correction_edits:
        return list(decision.correction_edits)
    if decision.correction_edit is None:
        return []
    return [decision.correction_edit]


def _run_review_inline_correction(
    project_dir: Path,
    decision: SpeakerReviewDecision,
    correction_options: ProjectReviewCorrectionOptions | None,
) -> None:
    """Run transcript correction from TUI-edited sentences."""
    correction_edits = _decision_correction_edits(decision)
    if correction_options is None or not correction_edits:
        typer.echo("Transcript correction is only available from meeting-asr project review.")
        return
    paths = project_paths(project_dir)
    manifest = run_with_cli_errors(lambda: load_manifest(paths.root))
    speaker_mapping = run_with_cli_errors(lambda: project_correct_commands.load_speaker_mapping_for_correction(paths.root))
    typer.echo("")
    typer.echo("Transcript correction:")
    run_with_cli_errors(
        lambda: project_correct_commands.finish_inline_corrections(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            correction_edits=correction_edits,
            options=correction_options.edit_options,
            yes=correction_options.yes,
        )
    )


def _run_review_transcript_correction(
    project_dir: Path,
    correction_options: ProjectReviewCorrectionOptions | None,
) -> None:
    """Run transcript correction after project review saves speaker names."""
    if correction_options is None:
        typer.echo("Transcript correction is only available from meeting-asr project review.")
        return
    paths = project_paths(project_dir)
    manifest = run_with_cli_errors(lambda: load_manifest(paths.root))
    speaker_mapping = run_with_cli_errors(lambda: project_correct_commands.load_speaker_mapping_for_correction(paths.root))
    typer.echo("")
    typer.echo("Transcript correction:")
    run_with_cli_errors(
        lambda: project_correct_commands.finish_editor_correction(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            options=correction_options.edit_options,
            yes=correction_options.yes,
        )
    )


def _resolve_review_project(project: str | None, projects_dir: Path | None, *, summary: bool) -> Path | None:
    """Resolve a project review target or run the project picker."""
    if project is not None:
        return run_with_cli_errors(lambda: resolve_project_ref(project, projects_dir))
    session = run_with_cli_errors(lambda: load_project_picker_session(projects_dir))
    if summary:
        typer.echo(render_project_picker_summary(session))
        return None
    if not session.projects:
        typer.echo("No projects found.")
        return None
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise typer.BadParameter("Project review TUI requires an interactive terminal. Provide PROJECT or use --summary.")
    selected = run_project_picker_tui(session)
    if selected is None:
        typer.echo("Project review exited without selecting a project.")
    return selected


@speakers_app.command("match")
def speakers_match(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    model: Optional[str] = typer.Option(None, "--model", autocompletion=complete_voiceprint_model),
    threshold: float = typer.Option(0.75, "--threshold", min=0.0, max=1.0),
    sample_count: int = typer.Option(2, "--sample-count", min=1, max=20),
    max_seconds: float = typer.Option(12.0, "--max-seconds", min=0.1),
    padding_seconds: float = typer.Option(0.5, "--padding-seconds", min=0.0),
    apply_matches: bool = typer.Option(False, "--apply"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Match project speakers against the cross-project voiceprint library."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    summary = run_with_progress(
        lambda reporter: match_project_speakers(
            resolved_project_dir,
            store_dir=store_dir,
            provider=None,
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
        run_with_cli_errors(
            lambda: apply_project_speakers(
                resolved_project_dir,
                summary.accepted_mapping,
                person_mapping=summary.accepted_person_mapping,
                person_public_mapping=summary.accepted_person_public_mapping,
            )
        )
        typer.echo("Applied accepted speaker matches.")


@speakers_app.command("compare-srt")
def speakers_compare_srt(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    dingtalk_srt: Path = typer.Option(..., "--dingtalk-srt", exists=True, file_okay=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Compare a DingTalk SRT with the project's subtitle."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    paths = project_paths(resolved_project_dir)
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
    asr_hotwords: str,
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
        asr_hotwords=asr_hotwords,
    )


def _echo_transcribe_summary(
    project_dir: Path,
    task_id: str,
    speaker_count: int,
    sentence_count: int,
    cost: AsrCostEstimate | None,
) -> None:
    """Print project transcription summary."""
    manifest = load_manifest(project_dir)
    project_ref = manifest.project_id
    typer.echo("")
    typer.echo("Project transcription completed.")
    typer.echo(f"Project: {project_dir}")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Title: {manifest.title}")
    typer.echo(f"Task ID: {task_id}")
    typer.echo(f"Detected speakers: {speaker_count}")
    typer.echo(f"Sentence count: {sentence_count}")
    typer.echo(f"ASR cost: {format_asr_cost(cost)}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  meeting-asr project review {shlex.quote(project_ref)}")
    typer.echo(f"  meeting-asr project transcript show {shlex.quote(project_ref)}")
    typer.echo(f"  meeting-asr project speakers preview {shlex.quote(project_ref)}")


def _echo_run_summary(summary: ProjectRunSummary) -> None:
    """
    Print full project run results.

    Args:
        summary: Full run summary.
    """
    view = _run_summary_view(summary)
    typer.echo("")
    render_project_run_summary(view)
    if summary.meeting_summary is not None:
        typer.echo(f"Memory index: {summary.meeting_summary.summary_path.resolve()}")
        typer.echo(f"Memory index JSON: {summary.meeting_summary.json_path.resolve()}")


def _run_summary_view(summary: ProjectRunSummary) -> ProjectRunSummaryView:
    """Build presentation data for project run output."""
    project_dir = summary.project.project_dir
    manifest = load_manifest(project_dir)
    total_matches = len(summary.matches.matches)
    accepted_matches = len(summary.applied_mapping)
    ignored_ids = load_project_ignored_speakers(project_dir)
    statuses = [
        effective_match_status(match, ignored_speaker_ids=ignored_ids) for match in summary.matches.matches
    ]
    below_threshold_matches = statuses.count(MATCH_STATUS_BELOW_THRESHOLD)
    no_candidate_matches = statuses.count(MATCH_STATUS_NO_CANDIDATE)
    return ProjectRunSummaryView(
        project_dir=project_dir,
        project_ref=manifest.project_id,
        manifest=manifest,
        total_matches=total_matches,
        accepted_matches=accepted_matches,
        below_threshold_matches=below_threshold_matches,
        no_candidate_matches=no_candidate_matches,
        unresolved_matches=below_threshold_matches + no_candidate_matches,
        source_label="new project" if summary.project.created else "reused project",
        meeting_summary=summary.meeting_summary,
        correction_summary=summary.correction_summary,
        lexicon_correction_summary=summary.lexicon_correction_summary,
        transcription=summary.transcription,
        speaker_matches=speaker_match_rows(
            summary.matches.matches,
            default_threshold=summary.matches.threshold,
            ignored_speaker_ids=ignored_ids,
        ),
    )


def _echo_project_created(
    project_dir: Path,
    manifest: ProjectManifest,
    *,
    created: bool,
) -> None:
    """Print project creation output with copyable next commands."""
    resolved_dir = project_dir.expanduser().resolve()
    project_ref = manifest.project_id
    typer.echo("")
    typer.echo("Project created." if created else "Project already exists; reusing it.")
    typer.echo(f"Project: {resolved_dir}")
    typer.echo(f"Project ID: {manifest.project_id}")
    typer.echo(f"Source: {manifest.source.path}")
    typer.echo(f"Meeting time: {manifest.source.meeting_time or '-'}")
    typer.echo(f"Status: {manifest.status}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  meeting-asr project transcribe {shlex.quote(project_ref)}")
    typer.echo(f"  meeting-asr project status {shlex.quote(project_ref)}")
    typer.echo(f"  meeting-asr project review {shlex.quote(project_ref)}")


def _echo_project_updated(summary: ProjectUpdateSummary) -> None:
    """
    Print project metadata update output.

    Args:
        summary: Project update summary.

    Returns:
        None.
    """
    typer.echo("Project updated.")
    typer.echo(f"Project ID: {summary.manifest.project_id}")
    typer.echo(f"Title: {summary.manifest.title}")
    typer.echo(f"Meeting time: {summary.manifest.source.meeting_time or '-'}")


def _confirm_project_delete(project_dir: Path, manifest: ProjectManifest, *, permanent: bool) -> bool:
    """
    Ask for destructive project delete confirmation.

    Args:
        project_dir: Project root.
        manifest: Project manifest.
        permanent: Whether this is a physical delete.

    Returns:
        True when deletion should proceed.
    """
    mode = "permanently delete" if permanent else "move to trash"
    return typer.confirm(f"{mode} project '{manifest.title}' at {project_dir}?")


def _echo_project_deleted(summary: ProjectDeleteSummary) -> None:
    """
    Print project deletion output.

    Args:
        summary: Project deletion summary.

    Returns:
        None.
    """
    if summary.permanent:
        typer.echo("Project permanently deleted.")
        typer.echo(f"Project: {summary.project_dir}")
        return
    typer.echo("Project moved to trash.")
    typer.echo(f"Project: {summary.project_dir}")
    typer.echo(f"Trash: {summary.destination}")
    if summary.destination is not None:
        restore_ref = shlex.quote(str(summary.destination))
        typer.echo("")
        typer.echo("Next steps:")
        typer.echo(f"  meeting-asr project trash restore {restore_ref}")
        typer.echo("  meeting-asr project trash list")
        typer.echo("  meeting-asr project trash cleanup --older-than-days 30 --yes")


def _relative_project_output(project_dir: Path, output_path: Path) -> str:
    """
    Return a project-relative output path for display.

    Args:
        project_dir: Project root.
        output_path: Output path.

    Returns:
        Project-relative path when possible.
    """
    try:
        return output_path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(output_path.resolve())


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
    project_dir = summary.match_path.parent.parent
    ignored = load_project_ignored_speakers(project_dir)
    for match in summary.matches:
        typer.echo(_voiceprint_match_cli_line(match, default_threshold=summary.threshold, ignored_speaker_ids=ignored))
    if any(
        effective_match_status(match, ignored_speaker_ids=ignored) not in (MATCH_STATUS_MATCHED, MATCH_STATUS_IGNORED)
        for match in summary.matches
    ):
        _echo_unresolved_speaker_next_steps(_project_ref_from_match_path(summary.match_path), speaker_only=True)


def _voiceprint_match_cli_line(
    match: object,
    *,
    default_threshold: float | None = None,
    ignored_speaker_ids: set[int] | None = None,
) -> str:
    """
    Format one voiceprint match line for CLI output.

    Args:
        match: Match dataclass or JSON-like mapping.
        default_threshold: Fallback threshold.
        ignored_speaker_ids: Speaker ids the user has marked as ignored.

    Returns:
        Status-explicit CLI line.
    """
    status = effective_match_status(match, ignored_speaker_ids=ignored_speaker_ids)
    label = str(getattr(match, "label", None) or "Speaker")
    threshold = match_threshold(match, default_threshold)
    threshold_text = "" if threshold is None else f" threshold={threshold:.3f}"
    if status == MATCH_STATUS_IGNORED:
        return f"{label} status=ignored"
    if status == MATCH_STATUS_MATCHED:
        name = accepted_match_name(match) or "unknown"
        score = best_candidate_score(match)
        score_text = "" if score is None else f" score={score:.3f}"
        return f"{label} status=matched name={name}{score_text}{threshold_text}"
    if status == MATCH_STATUS_BELOW_THRESHOLD:
        name = best_candidate_name(match) or "unrecorded"
        score = best_candidate_score(match)
        score_text = "" if score is None else f" score={score:.3f}"
        return f"{label} status=below-threshold best={name}{score_text}{threshold_text}"
    return f"{label} status=no-candidate{threshold_text}"


def _project_ref_from_match_path(match_path: Path) -> str:
    """
    Resolve a project reference for match remediation commands.

    Args:
        match_path: Path to speakers/speaker_matches.json.

    Returns:
        Project id when available, otherwise the project path.
    """
    project_dir = match_path.parent.parent
    try:
        return load_manifest(project_dir).project_id
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return str(project_dir)


def _echo_unresolved_speaker_next_steps(project_ref: str, *, speaker_only: bool) -> None:
    """
    Print concrete commands for unresolved speaker remediation.

    Args:
        project_ref: Project id or path accepted by project commands.
        speaker_only: Whether the current command is scoped to speaker review.
    """
    quoted_ref = shlex.quote(project_ref)
    review_command = (
        f"meeting-asr project speakers review {quoted_ref}"
        if speaker_only
        else f"meeting-asr project review {quoted_ref}"
    )
    typer.echo("")
    typer.echo(f"Recommended next step: {review_command}")
    typer.echo("This opens the human review workflow for unresolved speakers.")
    typer.echo("")
    typer.echo("Diagnostic/read-only:")
    typer.echo(f"  meeting-asr project speakers inspect {quoted_ref} --sample-count 5")
    typer.echo("")
    typer.echo("Advanced/scripted alternative (not the recommended human path):")
    typer.echo(f"  meeting-asr project speakers apply {quoted_ref} --map 0=Name")
    typer.echo("")
    typer.echo("After saving names:")
    typer.echo(f"  meeting-asr voiceprint review {quoted_ref}")
    typer.echo("  meeting-asr voiceprint embed")


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
    for candidate in (
        paths.exports_dir / "subtitle_named_corrected.srt",
        paths.exports_dir / "subtitle_named.srt",
        paths.exports_dir / "subtitle_corrected.srt",
        paths.exports_dir / "subtitle.srt",
    ):
        if candidate.exists():
            return candidate
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


def _load_speaker_match_summaries(
    project_dir: Path,
    speaker_mapping: dict[int, str],
    *,
    ignored_speaker_ids: set[int] | None = None,
) -> dict[int, str]:
    """
    Load speaker match results for inspect output.

    Args:
        project_dir: Project root.
        speaker_mapping: Existing confirmed speaker mapping.
        ignored_speaker_ids: Speaker ids the user has marked as ignored.

    Returns:
        Speaker id to display-safe match summary.
    """
    match_path = project_paths(project_dir).speakers_dir / "speaker_matches.json"
    if not match_path.exists():
        return {}
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    default_threshold = _safe_float(payload.get("threshold"))
    ignored = ignored_speaker_ids or set()
    summaries: dict[int, str] = {}
    for item in matches:
        if not isinstance(item, dict) or "speaker_id" not in item:
            continue
        if default_threshold is not None and item.get("threshold") is None:
            item = {**item, "threshold": default_threshold}
        speaker_id = int(item["speaker_id"])
        if speaker_id in ignored:
            continue
        summaries[speaker_id] = _speaker_match_summary(item, mapped_name=speaker_mapping.get(speaker_id))
    return summaries


def _project_has_unresolved_match(
    project_dir: Path,
    *,
    ignored_speaker_ids: set[int] | None = None,
) -> bool:
    """
    Return whether a project has any unresolved non-ignored speaker match.

    Args:
        project_dir: Project root.
        ignored_speaker_ids: Speaker ids the user has marked as ignored.

    Returns:
        True when at least one non-ignored match row is not automatically matched.
    """
    match_path = project_paths(project_dir).speakers_dir / "speaker_matches.json"
    if not match_path.exists():
        return False
    payload = json.loads(match_path.read_text(encoding="utf-8"))
    ignored = ignored_speaker_ids or set()
    for item in payload.get("matches", []):
        if not isinstance(item, dict):
            continue
        speaker_id = speaker_id_from_match(item)
        if speaker_id is not None and speaker_id in ignored:
            continue
        if voiceprint_match_status(item) != MATCH_STATUS_MATCHED:
            return True
    return False


def _speaker_match_summary(item: dict[str, object], *, mapped_name: str | None = None) -> str:
    """
    Format one voiceprint match row for humans.

    Args:
        item: Raw match JSON item.
        mapped_name: Existing confirmed speaker name.

    Returns:
        Short match summary.
    """
    status = voiceprint_match_status(item)
    accepted = status == MATCH_STATUS_MATCHED
    score = best_candidate_score(item)
    threshold = match_threshold(item)
    name = accepted_match_name(item) if accepted else best_candidate_name(item)
    conflict = _speaker_match_conflicts(item, name, mapped_name)
    suffix = " CONFLICT" if conflict else ""
    if status == MATCH_STATUS_NO_CANDIDATE:
        threshold_text = "" if threshold is None else f" threshold={threshold:.3f}"
        return _style_speaker_match_summary(
            f"status=no-candidate{threshold_text}{suffix}",
            accepted=accepted,
            conflict=conflict,
        )
    threshold_text = "" if threshold is None else f" threshold={threshold:.3f}"
    if score is None:
        label = f"name={name}" if accepted else f"best={name or 'unrecorded'}"
        return _style_speaker_match_summary(
            f"{label} status={status}{threshold_text}{suffix}",
            accepted=accepted,
            conflict=conflict,
        )
    label = f"name={name}" if accepted else f"best={name or 'unrecorded'}"
    return _style_speaker_match_summary(
        f"{label} score={score:.3f}{threshold_text} status={status}{suffix}",
        accepted=accepted,
        conflict=conflict,
    )


def _style_speaker_match_summary(text: str, *, accepted: bool, conflict: bool) -> str:
    """
    Color a voiceprint match summary by review state.

    Args:
        text: Plain match summary.
        accepted: Whether the match passed the configured threshold.
        conflict: Whether the match conflicts with a manual name.

    Returns:
        Styled terminal text.
    """
    if conflict:
        return typer.style(text, fg=typer.colors.RED, bold=True)
    if accepted:
        return typer.style(text, fg=typer.colors.GREEN)
    return typer.style(text, fg=typer.colors.YELLOW)


def _speaker_match_conflicts(item: dict[str, object], match_name: str | None, mapped_name: str | None) -> bool:
    """
    Return whether a voiceprint match conflicts with a confirmed mapping.

    Args:
        item: Raw match JSON item.
        match_name: Display name from the match row.
        mapped_name: Existing confirmed speaker name.

    Returns:
        ``True`` when both sides name different real speakers.
    """
    if not mapped_name or not match_name or match_name == "unknown":
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
