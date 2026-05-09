"""Voiceprint registry commands."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table
import typer

from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json
from app.presentation.cli.output import cli_console
from app.presentation.cli.plain import echo_plain_table
from app.presentation.cli.voiceprint_quality import (
    voiceprint_quality_payload,
    voiceprint_quality_summary_lines,
    voiceprint_quality_table,
)
from app.presentation.cli.progress import run_with_progress
from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper
from app.completion_helpers import complete_voiceprint_model
from app.core.project_refs import resolve_project_ref
from app.utils import format_ms_timestamp
from app.voiceprint_audio import normalize_voiceprint_samples
from app.voiceprint_playback import build_voiceprint_play_command
from app.voiceprint_people import (
    create_voiceprint_person,
    get_voiceprint_person,
    rename_voiceprint_person,
)
from app.voiceprint_store import (
    delete_voiceprint_sample,
    delete_voiceprint_speaker,
    get_voiceprint_clip_dir,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    list_voiceprint_speakers,
    VoiceprintSampleRow,
    VoiceprintSpeakerRow,
)
from app.presentation.tui.voiceprint import (
    load_voiceprint_library_session,
    render_voiceprint_library_summary,
    run_voiceprint_library_tui,
)
from app.presentation.tui.voiceprint_review import (
    load_voiceprint_review_session,
    run_voiceprint_review_tui,
)
from app.presentation.tui.voiceprint_review_context import render_voiceprint_review_summary
from app.presentation.tui.voiceprint_quality import (
    persist_quality_decision,
    run_voiceprint_quality_review_tui,
)
from app.voiceprint_embedding import embed_voiceprint_samples
from app.voiceprint_quality import (
    VoiceprintQualityReport,
    analyze_voiceprint_quality,
)
from app.voiceprints import (
    VoiceprintCaptureSummary,
    capture_voiceprints,
    persist_voiceprint_capture_selection,
)

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
people_app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(people_app, name="people", help="Manage stable voiceprint people.")


@app.command("review")
def review_command(
    project_dir: Optional[Path] = typer.Argument(None, metavar="[PROJECT]", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    sample_count: int = typer.Option(3, "--sample-count", min=1, max=20),
    max_seconds: float = typer.Option(12.0, "--max-seconds", min=0.1),
    padding_seconds: float = typer.Option(0.5, "--padding-seconds", min=0.0),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    page_size: Optional[int] = typer.Option(
        None,
        "--page-size",
        min=1,
        max=50,
        help="Override samples per page. By default the TUI uses the pane height.",
    ),
    summary: bool = typer.Option(False, "--summary", help="Print project candidates and library without opening TUI."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Open the unified voiceprint TUI for project candidates and the global library."""
    resolved_project_dir = None
    if project_dir is not None:
        resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    session, planned = run_with_cli_errors(
        lambda: load_voiceprint_review_session(
            project_dir=resolved_project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            page_size=page_size,
        )
    )
    if summary:
        typer.echo(render_voiceprint_review_summary(session))
        return
    if session.capture is None and not session.library.speakers:
        typer.echo("No project was provided and no voiceprints are recorded.")
        typer.echo("Start with: meeting-asr voiceprint review PROJECT_ID")
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(render_voiceprint_review_summary(session))
        raise typer.BadParameter("Voiceprint review TUI requires an interactive terminal. Use --summary to inspect.")
    decision = run_voiceprint_review_tui(session)
    if not decision.saved:
        typer.echo("Voiceprint review closed; no samples were written.")
        return
    if resolved_project_dir is None or planned is None:
        typer.echo("No project candidates were saved.")
        return
    captured = _persist_review_decision(
        project_dir=resolved_project_dir,
        planned=planned,
        selected_clip_rel_paths=decision.selected_clip_rel_paths,
        progress=progress,
    )
    _echo_capture_summary(captured)


@app.command("capture")
def capture_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    sample_count: int = typer.Option(3, "--sample-count", min=1, max=20),
    max_seconds: float = typer.Option(12.0, "--max-seconds", min=0.1),
    padding_seconds: float = typer.Option(0.5, "--padding-seconds", min=0.0),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    review: bool = typer.Option(False, "--review", help="Compatibility shortcut; prefer `voiceprint review PROJECT`."),
    page_size: Optional[int] = typer.Option(
        None,
        "--page-size",
        min=1,
        max=50,
        help="Override samples per page when --review is used.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Capture this project's named speakers into the global voiceprint store."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    if dry_run:
        summary = run_with_progress(
            lambda reporter: capture_voiceprints(
                resolved_project_dir,
                sample_count=sample_count,
                max_seconds=max_seconds,
                padding_seconds=padding_seconds,
                store_dir=store_dir,
                dry_run=True,
                progress=reporter,
            ),
            description="Planning voiceprints",
            enabled=progress,
        )
        _echo_capture_summary(summary)
        return
    if review:
        summary = _run_capture_review_workflow(
            project_dir=resolved_project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            page_size=page_size,
            progress=progress,
        )
        if summary is not None:
            _echo_capture_summary(summary)
        return
    summary = run_with_progress(
        lambda reporter: capture_voiceprints(
            resolved_project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            dry_run=dry_run,
            progress=reporter,
        ),
        description="Capturing voiceprints",
        enabled=progress,
    )
    _echo_capture_summary(summary)


def _run_capture_review_workflow(
    *,
    project_dir: Path,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None,
    page_size: int | None,
    progress: bool,
) -> VoiceprintCaptureSummary | None:
    """
    Plan voiceprint clips, run human review, and persist selected samples.

    Args:
        project_dir: Project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional voiceprint store directory.
        page_size: Optional TUI samples per page.
        progress: Whether to show capture progress after review.

    Returns:
        Captured summary, or None when the review was cancelled.
    """
    return _run_unified_voiceprint_review_workflow(
        project_dir=project_dir,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        store_dir=store_dir,
        page_size=page_size,
        progress=progress,
    )


def _run_unified_voiceprint_review_workflow(
    *,
    project_dir: Path,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    store_dir: Path | None,
    page_size: int | None,
    progress: bool,
) -> VoiceprintCaptureSummary | None:
    """
    Run unified review with a project loaded, then persist selected samples.

    Args:
        project_dir: Project root.
        sample_count: Maximum clips per speaker.
        max_seconds: Maximum seconds per output clip.
        padding_seconds: Extra context around each sentence.
        store_dir: Optional voiceprint store directory.
        page_size: Optional TUI samples per page.
        progress: Whether to show capture progress after review.

    Returns:
        Captured summary, or None when the review was cancelled.
    """
    session, planned = run_with_cli_errors(
        lambda: load_voiceprint_review_session(
            project_dir=project_dir,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            store_dir=store_dir,
            page_size=page_size,
        )
    )
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(render_voiceprint_review_summary(session))
        raise typer.BadParameter(
            "Voiceprint review TUI requires an interactive terminal. "
            "Use --dry-run from capture, or `meeting-asr voiceprint review PROJECT --summary` to inspect."
        )
    decision = run_voiceprint_review_tui(session)
    if not decision.saved:
        typer.echo("Voiceprint review closed; no samples were written.")
        return None
    if planned is None:
        return None
    return _persist_review_decision(
        project_dir=project_dir,
        planned=planned,
        selected_clip_rel_paths=decision.selected_clip_rel_paths,
        progress=progress,
    )


def _persist_review_decision(
    *,
    project_dir: Path,
    planned: VoiceprintCaptureSummary,
    selected_clip_rel_paths: frozenset[str],
    progress: bool,
) -> VoiceprintCaptureSummary:
    """
    Persist selected review samples into the global voiceprint store.

    Args:
        project_dir: Project root.
        planned: Dry-run capture plan.
        selected_clip_rel_paths: Human-approved planned clip relative paths.
        progress: Whether to show capture progress.

    Returns:
        Captured voiceprint summary.
    """
    return run_with_progress(
        lambda reporter: persist_voiceprint_capture_selection(
            project_dir,
            planned=planned,
            selected_clip_rel_paths=selected_clip_rel_paths,
            progress=reporter,
        ),
        description="Capturing selected voiceprints",
        enabled=progress,
    )


@app.command("list")
def list_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    plain: bool = typer.Option(False, "--plain", help="Print stable tab-separated output."),
) -> None:
    """List speakers recorded in the global voiceprint registry."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_speakers(db_path))
    if as_json:
        emit_json(_voiceprint_speakers_payload(db_path, rows))
        return
    if plain:
        _echo_voiceprint_speaker_table_plain(rows)
        return
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo("No voiceprints recorded.")
        return
    _echo_voiceprint_speaker_table(rows)


@app.command("browse", hidden=True)
def browse_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    page_size: Optional[int] = typer.Option(
        None,
        "--page-size",
        min=1,
        max=50,
        help="Override samples per page. By default the TUI uses the pane height.",
    ),
    summary: bool = typer.Option(False, "--summary", help="Print the library without opening the TUI."),
) -> None:
    """Open the legacy global-library TUI; prefer `voiceprint review`."""
    session = run_with_cli_errors(
        lambda: load_voiceprint_library_session(store_dir=store_dir, page_size=page_size)
    )
    if summary:
        typer.echo(render_voiceprint_library_summary(session))
        return
    if not session.speakers:
        typer.echo("No voiceprints recorded.")
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise typer.BadParameter(
            "Voiceprint browser TUI requires an interactive terminal. Use --summary to inspect."
        )
    run_voiceprint_library_tui(session)


@people_app.command("list")
def people_list_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    plain: bool = typer.Option(False, "--plain", help="Print stable tab-separated output."),
) -> None:
    """List stable voiceprint people and their sample coverage."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_speakers(db_path))
    if as_json:
        emit_json(_voiceprint_speakers_payload(db_path, rows))
        return
    if plain:
        _echo_voiceprint_speaker_table_plain(rows)
        return
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo("No voiceprint people recorded.")
        typer.echo("Create one with: meeting-asr voiceprint people add NAME")
        return
    _echo_voiceprint_speaker_table(rows, title="People")


@people_app.command("add")
def people_add_command(
    name: str = typer.Argument(..., metavar="NAME"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Create one stable voiceprint person; names cannot be silently reused."""
    db_path = get_voiceprint_db_path(store_dir)
    row = run_with_cli_errors(lambda: create_voiceprint_person(name, db_path))
    if as_json:
        emit_json(_voiceprint_speaker_payload(row))
        return
    typer.echo(f"Created person: {row.name}")
    typer.echo(f"Person ID: {row.public_id}")
    typer.echo("Use this ID as the stable identity; display names may change later.")


@people_app.command("rename")
def people_rename_command(
    person_id: str = typer.Argument(..., metavar="PERSON_ID"),
    name: str = typer.Argument(..., metavar="NAME"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Rename one stable voiceprint person by ID."""
    db_path = get_voiceprint_db_path(store_dir)
    row = run_with_cli_errors(lambda: rename_voiceprint_person(person_id, name, db_path))
    if as_json:
        emit_json(_voiceprint_speaker_payload(row))
        return
    typer.echo(f"Renamed person {row.public_id}: {row.name}")


@people_app.command("show")
def people_show_command(
    person_id: str = typer.Argument(..., metavar="PERSON_ID"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show one stable voiceprint person by ID."""
    db_path = get_voiceprint_db_path(store_dir)
    row = run_with_cli_errors(lambda: get_voiceprint_person(person_id, db_path))
    if row is None:
        raise typer.BadParameter(f"No voiceprint person found for id: {person_id}")
    if as_json:
        emit_json(_voiceprint_speaker_payload(row))
        return
    typer.echo(f"Person ID: {row.public_id}")
    typer.echo(f"Name: {row.name}")
    typer.echo(f"Samples: {row.sample_count}")
    typer.echo(f"Projects: {row.project_count}")
    typer.echo(f"Embedded: {row.embedded_sample_count}/{row.sample_count}")


@app.command("embed")
def embed_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    model: Optional[str] = typer.Option(None, "--model", autocompletion=complete_voiceprint_model),
    rebuild: bool = typer.Option(False, "--rebuild"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
) -> None:
    """Normalize audio and generate embeddings for stored voiceprint samples."""
    summary = run_with_progress(
        lambda reporter: embed_voiceprint_samples(
            store_dir=store_dir,
            provider=None,
            model=model,
            rebuild=rebuild,
            progress=reporter,
        ),
        description="Embedding voiceprints",
        enabled=progress,
    )
    typer.echo(f"Database: {summary.db_path}")
    typer.echo(f"Provider: {summary.provider}")
    typer.echo(f"Model: {summary.model}")
    typer.echo("Audio preprocessing: audio-norm-v1")
    typer.echo(f"Embedded: {summary.embedded_count}")
    typer.echo(f"Skipped: {summary.skipped_count}")


@app.command("normalize")
def normalize_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    rebuild: bool = typer.Option(False, "--rebuild"),
) -> None:
    """Normalize stored voiceprint sample audio without modifying original clips."""
    summary = run_with_cli_errors(lambda: normalize_voiceprint_samples(store_dir=store_dir, rebuild=rebuild))
    typer.echo(f"Store: {summary.store_dir}")
    typer.echo(f"Normalized: {summary.normalized_dir}")
    typer.echo(f"Processed: {summary.processed_count}")
    typer.echo(f"Skipped: {summary.skipped_count}")


@app.command("quality")
def quality_command(
    speaker: Optional[str] = typer.Argument(None, metavar="[SPEAKER]"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    model: Optional[str] = typer.Option(None, "--model", autocompletion=complete_voiceprint_model),
    review: bool = typer.Option(False, "--review", help="Open an interactive review TUI for suspicious samples."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Find and review voiceprint sample outliers without deleting user data."""
    report = run_with_cli_errors(
        lambda: analyze_voiceprint_quality(store_dir=store_dir, speaker=speaker, model=model)
    )
    if as_json:
        emit_json(voiceprint_quality_payload(report))
        return
    if review:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            _echo_voiceprint_quality_report(report)
            raise typer.BadParameter("Voiceprint quality TUI requires an interactive terminal.")
        decision = run_voiceprint_quality_review_tui(report, store_dir=store_dir, speaker=speaker, model=model)
        changes = run_with_cli_errors(lambda: persist_quality_decision(decision, store_dir=store_dir))
        if not decision.saved:
            typer.echo("Voiceprint quality review closed; no changes were written.")
            return
        typer.echo(f"Voiceprint quality changes saved: {len(changes)}")
        for sample_id, status in changes.items():
            typer.echo(f"  {sample_id} -> {status}")
        return
    _echo_voiceprint_quality_report(report)


@app.command("show")
def show_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show voiceprint samples for one speaker name or id."""
    db_path = get_voiceprint_db_path(store_dir)
    rows = run_with_cli_errors(lambda: list_voiceprint_samples(speaker, db_path))
    if as_json:
        emit_json(_voiceprint_samples_payload(db_path, speaker, rows))
        return
    typer.echo(f"Database: {db_path}")
    if not rows:
        typer.echo(f"No voiceprint samples found for: {speaker}")
        raise typer.Exit(code=1)
    for index, row in enumerate(rows, start=1):
        start = format_ms_timestamp(row.source_begin_time_ms)
        end = format_ms_timestamp(row.source_end_time_ms)
        typer.echo(f"[{index}] {row.speaker_name} | {row.project_id} | speaker {row.project_speaker_id}")
        typer.echo(f"  person_id: {row.speaker_public_id}")
        typer.echo(f"  sample_id: {row.public_id}")
        typer.echo(f"  clip: {row.clip_path}")
        typer.echo(f"  time: {start} - {end}")
        typer.echo(f"  sha256: {row.clip_sha256}")
        typer.echo(f"  status: {row.sample_status}")
        typer.echo(f"  text: {row.transcript_text}")


@app.command("play")
def play_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    sample: int = typer.Option(1, "--sample", "-s", min=1),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Play one numbered voiceprint sample."""
    db_path = get_voiceprint_db_path(store_dir)
    row = run_with_cli_errors(lambda: _select_sample(speaker, sample, db_path))
    command = build_voiceprint_play_command(row.clip_path)
    if dry_run:
        typer.echo(" ".join(command))
        return
    run_with_cli_errors(lambda: subprocess.run(command, check=True))


@app.command("delete-sample")
def delete_sample_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    sample: int = typer.Option(..., "--sample", "-s", min=1),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    keep_clip: bool = typer.Option(False, "--keep-clip"),
) -> None:
    """Delete one numbered voiceprint sample and its WAV file."""
    db_path = get_voiceprint_db_path(store_dir)
    deleted = run_with_cli_errors(
        lambda: delete_voiceprint_sample(speaker, sample, db_path=db_path, delete_clip=not keep_clip)
    )
    _echo_deleted_sample(deleted.clip_path, deleted.clip_deleted, kept=keep_clip)


@app.command("delete-speaker")
def delete_speaker_command(
    speaker: str = typer.Argument(..., metavar="SPEAKER"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
    keep_clips: bool = typer.Option(False, "--keep-clips"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete one speaker and all of their voiceprint samples."""
    db_path = get_voiceprint_db_path(store_dir)
    if not yes and not typer.confirm(f"Delete all voiceprint samples for {_speaker_label(speaker, db_path)}?"):
        raise typer.Exit(code=1)
    deleted = run_with_cli_errors(
        lambda: delete_voiceprint_speaker(speaker, db_path=db_path, delete_clips=not keep_clips)
    )
    typer.echo(f"Deleted speaker: {deleted[0].speaker_name} (id {deleted[0].speaker_public_id})")
    for item in deleted:
        _echo_deleted_sample(item.clip_path, item.clip_deleted, kept=keep_clips)


@app.command("path")
def path_command(
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", file_okay=False, dir_okay=True),
) -> None:
    """Print the global voiceprint store paths."""
    db_path = get_voiceprint_db_path(store_dir)
    typer.echo(f"Store: {db_path.parent}")
    typer.echo(f"Database: {db_path}")
    typer.echo(f"Clips: {get_voiceprint_clip_dir(store_dir)}")


def _echo_voiceprint_speaker_table(rows: list[VoiceprintSpeakerRow], *, title: str = "Speakers") -> None:
    """
    Print voiceprint speakers as a compact summary table.

    Args:
        rows: Speaker summary rows.
        title: Human label for the first count.
    """
    sample_total = sum(row.sample_count for row in rows)
    embedded_total = sum(row.embedded_sample_count for row in rows)
    typer.echo(f"{title}: {len(rows)} | Samples: {sample_total} | Embedded samples: {embedded_total}/{sample_total}")
    _voiceprint_table_console().print(_voiceprint_speaker_table(rows))


def _echo_voiceprint_speaker_table_plain(rows: list[VoiceprintSpeakerRow]) -> None:
    """
    Print voiceprint speakers as stable tab-separated values.

    Args:
        rows: Speaker summary rows.
    """
    plain_rows = [
        (
            row.public_id,
            row.speaker_id,
            row.name,
            row.sample_count,
            row.project_count,
            f"{row.embedded_sample_count}/{row.sample_count}",
            row.embedding_model_count,
            _format_updated_at(row.updated_at),
        )
        for row in rows
    ]
    echo_plain_table(("id", "internal_id", "speaker", "samples", "projects", "embedded", "models", "updated"), plain_rows)


def _voiceprint_speaker_table(rows: list[VoiceprintSpeakerRow]) -> Table:
    """
    Build the voiceprint speaker summary table.

    Args:
        rows: Speaker summary rows.

    Returns:
        Rich table ready to print.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("ID", no_wrap=True, style="bold cyan")
    table.add_column("Speaker")
    table.add_column("Samples", justify="right", no_wrap=True)
    table.add_column("Projects", justify="right", no_wrap=True)
    table.add_column("Embedded", justify="right", no_wrap=True)
    table.add_column("Models", justify="right", no_wrap=True)
    table.add_column("Updated", no_wrap=True)
    for row in rows:
        table.add_row(
            row.public_id,
            row.name,
            str(row.sample_count),
            str(row.project_count),
            _embedded_count_text(row),
            str(row.embedding_model_count),
            _format_updated_at(row.updated_at),
        )
    return table


def _voiceprint_speakers_payload(db_path: Path, rows: list[VoiceprintSpeakerRow]) -> dict[str, object]:
    """
    Build a machine-readable voiceprint speaker list payload.

    Args:
        db_path: Voiceprint SQLite database path.
        rows: Speaker summary rows.

    Returns:
        JSON-ready speaker list payload.
    """
    sample_total = sum(row.sample_count for row in rows)
    embedded_total = sum(row.embedded_sample_count for row in rows)
    return {
        "database": db_path,
        "count": len(rows),
        "sample_count": sample_total,
        "embedded_sample_count": embedded_total,
        "speakers": [_voiceprint_speaker_payload(row) for row in rows],
    }


def _voiceprint_speaker_payload(row: VoiceprintSpeakerRow) -> dict[str, object]:
    """
    Build one voiceprint speaker JSON row.

    Args:
        row: Speaker summary row.

    Returns:
        JSON-ready speaker row.
    """
    return {
        "speaker_id": row.speaker_id,
        "public_id": row.public_id,
        "name": row.name,
        "sample_count": row.sample_count,
        "project_count": row.project_count,
        "embedded_sample_count": row.embedded_sample_count,
        "embedding_model_count": row.embedding_model_count,
        "updated_at": row.updated_at,
    }


def _voiceprint_samples_payload(
    db_path: Path,
    speaker: str,
    rows: list[VoiceprintSampleRow],
) -> dict[str, object]:
    """
    Build a machine-readable voiceprint sample payload.

    Args:
        db_path: Voiceprint SQLite database path.
        speaker: User-provided speaker reference.
        rows: Sample rows.

    Returns:
        JSON-ready sample list payload.
    """
    return {
        "database": db_path,
        "speaker": speaker,
        "count": len(rows),
        "samples": [_voiceprint_sample_payload(index, row) for index, row in enumerate(rows, start=1)],
    }


def _voiceprint_sample_payload(index: int, row: VoiceprintSampleRow) -> dict[str, object]:
    """
    Build one voiceprint sample JSON row.

    Args:
        index: One-based sample index for CLI selection.
        row: Sample row.

    Returns:
        JSON-ready sample row.
    """
    return {
        "index": index,
        "sample_id": row.sample_id,
        "public_id": row.public_id,
        "speaker_id": row.speaker_id,
        "speaker_public_id": row.speaker_public_id,
        "speaker_name": row.speaker_name,
        "project_id": row.project_id,
        "project_speaker_id": row.project_speaker_id,
        "clip_path": row.clip_path,
        "clip_rel_path": row.clip_rel_path,
        "clip_sha256": row.clip_sha256,
        "source_begin_time_ms": row.source_begin_time_ms,
        "source_end_time_ms": row.source_end_time_ms,
        "status": row.sample_status,
        "transcript_text": row.transcript_text,
    }


def _echo_voiceprint_quality_report(report: VoiceprintQualityReport) -> None:
    """
    Print voiceprint quality diagnostics.

    Args:
        report: Quality report.
    """
    for line in voiceprint_quality_summary_lines(report):
        typer.echo(line)
    if not report.people:
        typer.echo("No embedded voiceprint samples found.")
        return
    _voiceprint_table_console().print(voiceprint_quality_table(report))
    if report.suspicious_count:
        typer.echo("")
        typer.echo("Review suspicious samples:")
        typer.echo("  meeting-asr voiceprint quality --review")
        typer.echo("Quarantined samples are kept in the library but excluded from future matching.")


def _embedded_count_text(row: VoiceprintSpeakerRow) -> str:
    """
    Return styled embedding coverage for one speaker.

    Args:
        row: Speaker summary row.

    Returns:
        Rich markup string for embedded sample coverage.
    """
    text = f"{row.embedded_sample_count}/{row.sample_count}"
    if row.sample_count == 0:
        return f"[yellow]{text}[/]"
    if row.embedded_sample_count == row.sample_count:
        return f"[green]{text}[/]"
    if row.embedded_sample_count == 0:
        return f"[red]{text}[/]"
    return f"[yellow]{text}[/]"


def _voiceprint_table_console() -> Console:
    """
    Build the stdout console used for voiceprint tables.

    Returns:
        Rich console instance.
    """
    return cli_console(width=140)


def _format_updated_at(value: str | None) -> str:
    """
    Format an ISO timestamp for table display.

    Args:
        value: Optional ISO timestamp.

    Returns:
        Compact timestamp or ``-`` when absent.
    """
    if not value:
        return "-"
    date_text, separator, time_text = value.partition("T")
    if not separator:
        return value
    return f"{date_text} {time_text[:8]}"


def _echo_capture_summary(summary: VoiceprintCaptureSummary) -> None:
    """
    Print capture results.

    Args:
        summary: Capture summary.
    """
    status = "Planned" if summary.dry_run else "Captured"
    typer.echo(f"{status} voiceprint samples: {summary.sample_count}")
    typer.echo(f"Store: {summary.store_dir}")
    typer.echo(f"Database: {summary.db_path}")
    typer.echo(f"Clips: {summary.clip_dir}")
    for speaker in summary.speakers:
        person = "" if speaker.person_public_id is None else f", person {speaker.person_public_id}"
        typer.echo(f"{speaker.name} (speaker {speaker.speaker_id}{person}): {len(speaker.clips)} sample(s)")
        for clip in speaker.clips:
            typer.echo(
                f"  - {clip.path} "
                f"(score={clip.selection_score:.3f}; {clip.selection_reason}; {clip.audio_reason})"
            )
    if not summary.dry_run and summary.sample_count:
        typer.echo("")
        typer.echo("Next steps:")
        typer.echo(f"  {_voiceprint_embed_command(summary.store_dir)}")
        typer.echo("  meeting-asr voiceprint review")
        typer.echo("  meeting-asr voiceprint list")


def _voiceprint_embed_command(store_dir: Path) -> str:
    """
    Build the next embedding command after capture.

    Args:
        store_dir: Store directory used by capture.

    Returns:
        Copyable voiceprint embed command.
    """
    default_store_dir = get_voiceprint_db_path().parent
    if store_dir.expanduser().resolve() == default_store_dir:
        return "meeting-asr voiceprint embed"
    return f"meeting-asr voiceprint embed --store-dir {shlex.quote(str(store_dir))}"


def _select_sample(speaker: str, sample: int, db_path: Path) -> VoiceprintSampleRow:
    """
    Select one sample for CLI playback.

    Args:
        speaker: Speaker name or speaker id.
        sample: One-based sample number.
        db_path: SQLite database path.

    Returns:
        Selected sample row.
    """
    rows = list_voiceprint_samples(speaker, db_path)
    if sample < 1 or sample > len(rows):
        raise IndexError(f"Sample {sample} is out of range for {speaker}. Available: {len(rows)}.")
    return rows[sample - 1]


def _speaker_label(speaker: str, db_path: Path) -> str:
    """
    Return a confirmation label for a speaker reference.

    Args:
        speaker: Speaker name or speaker id.
        db_path: SQLite database path.

    Returns:
        Human-readable confirmation label.
    """
    rows = run_with_cli_errors(lambda: list_voiceprint_samples(speaker, db_path))
    if not rows:
        return speaker
    return f"{rows[0].speaker_name} (id {rows[0].speaker_public_id})"


def _echo_deleted_sample(path: Path, clip_deleted: bool, *, kept: bool = False) -> None:
    """
    Print deletion result for one sample.

    Args:
        path: Clip path.
        clip_deleted: Whether the clip file was deleted.
        kept: Whether the user requested keeping the file.
    """
    status = "kept" if kept else ("deleted" if clip_deleted else "not found")
    typer.echo(f"Deleted sample: {path}")
    typer.echo(f"  clip file: {status}")
