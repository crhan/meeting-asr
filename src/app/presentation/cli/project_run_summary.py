"""Rich rendering for project run summaries."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.asr_pricing import format_asr_cost
from app.core.project_models import ProjectManifest, ProjectMeetingSummary, ProjectTranscribeSummary
from app.presentation.cli.output import cli_console


@dataclass(frozen=True, slots=True)
class ProjectRunSummaryView:
    """Presentation data for project run output."""

    project_dir: Path
    project_ref: str
    manifest: ProjectManifest
    total_matches: int
    accepted_matches: int
    unresolved_matches: int
    source_label: str
    meeting_summary: ProjectMeetingSummary | None
    transcription: ProjectTranscribeSummary


def render_project_run_summary(view: ProjectRunSummaryView) -> None:
    """
    Render a scan-friendly project run summary.

    Args:
        view: Fully prepared presentation data.

    Returns:
        None.
    """
    console = _summary_console()
    console.print(_status_panel(view))
    console.print(_metrics_table(view))
    console.print(_outputs_table(view))
    console.print(_next_steps_table(view))
    if view.unresolved_matches:
        console.print(_agent_prompt_panel(view))


def _status_panel(view: ProjectRunSummaryView) -> Panel:
    """Build the high-level project run status panel."""
    completed = view.unresolved_matches == 0
    status = "Project automation completed." if completed else "Project automation needs review."
    style = "green" if completed else "yellow"
    grid = Table.grid(expand=True)
    grid.add_column(ratio=3)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(f"[bold]{view.manifest.title}[/]", f"[bold cyan]Project {view.project_ref}[/]")
    grid.add_row(f"[dim]{view.manifest.project_id}[/]", f"[{style}]{_voiceprint_label(view)}[/]")
    return Panel(grid, title=f"[bold {style}]{status}[/]", border_style=style, expand=False)


def _metrics_table(view: ProjectRunSummaryView) -> Table:
    """Build compact run metrics."""
    table = Table(title="Run", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("Item", style="bold", no_wrap=True)
    table.add_column("Value")
    for item, value in _metric_rows(view):
        table.add_row(item, value)
    return table


def _outputs_table(view: ProjectRunSummaryView) -> Table:
    """Build project output artifact table."""
    table = Table(title="Outputs", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("Artifact", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Path")
    for artifact, status, path in _output_rows(view):
        table.add_row(artifact, _status_text(status), path)
    return table


def _next_steps_table(view: ProjectRunSummaryView) -> Table:
    """Build next-step commands."""
    table = Table(title="Next", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Action", style="bold", no_wrap=True)
    table.add_column("Command")
    for index, (action, command) in enumerate(_next_step_rows(view), start=1):
        table.add_row(str(index), action, command)
    return table


def _agent_prompt_panel(view: ProjectRunSummaryView) -> Panel:
    """Build an agent-friendly remediation prompt."""
    prompt = (
        f"Open project review for {view.project_ref}, resolve {view.unresolved_matches} unaccepted speaker(s), "
        "save named outputs, run vocabulary correction, then verify the corrected transcript and subtitle preview."
    )
    return Panel(prompt, title="[bold yellow]Agent prompt:[/]", border_style="yellow", expand=False)


def _metric_rows(view: ProjectRunSummaryView) -> list[tuple[str, str]]:
    """Return compact run metric rows."""
    rows = [
        ("Title", view.manifest.title),
        ("Project", str(view.project_dir)),
        ("Project ID", view.manifest.project_id),
        ("Source", view.source_label),
        ("Speakers", f"{view.transcription.detected_speaker_count} detected"),
        ("Sentences", str(view.transcription.sentence_count)),
        ("ASR cost", format_asr_cost(view.transcription.cost)),
        ("Voiceprint matches", _voiceprint_label(view)),
        ("ASR task", view.transcription.task_id),
    ]
    if view.meeting_summary is not None:
        rows.append(("Summary", _relative_project_output(view.project_dir, view.meeting_summary.summary_path)))
        if view.meeting_summary.title_updated:
            rows.append(("Auto title", view.meeting_summary.title))
    return rows


def _output_rows(view: ProjectRunSummaryView) -> list[tuple[str, str, str]]:
    """Return output artifact rows for the run summary."""
    final_status = _final_output_status(view)
    rows = [
        ("Final transcript", final_status, "exports/transcript_named.txt"),
        ("Final subtitles", final_status, "exports/subtitle_named.srt"),
        ("Speaker transcript", "ready", "exports/transcript_speakers.txt"),
        ("Plain transcript", "ready", "exports/transcript.txt"),
    ]
    if view.meeting_summary is not None:
        rows.append(("Meeting summary", "ready", "exports/meeting_summary.md"))
        rows.append(("Summary JSON", "supporting", "exports/meeting_summary.json"))
    return rows


def _next_step_rows(view: ProjectRunSummaryView) -> list[tuple[str, str]]:
    """Return action rows for next-step commands."""
    quoted_ref = shlex.quote(view.project_ref)
    if view.unresolved_matches == 0:
        return [
            ("Correct vocabulary", f"meeting-asr project correct edit {quoted_ref}"),
            ("View corrected transcript", f"meeting-asr project transcript show {quoted_ref} --kind corrected"),
            ("Preview subtitles", f"meeting-asr project speakers preview {quoted_ref}"),
        ]
    return [
        ("Resolve speakers", f"meeting-asr project review {quoted_ref}"),
        ("Then correct vocabulary", f"meeting-asr project correct edit {quoted_ref}"),
        ("Then view corrected transcript", f"meeting-asr project transcript show {quoted_ref} --kind corrected"),
        ("Then preview subtitles", f"meeting-asr project speakers preview {quoted_ref}"),
    ]


def _voiceprint_label(view: ProjectRunSummaryView) -> str:
    """Return a compact voiceprint match label."""
    return f"{view.accepted_matches}/{view.total_matches} accepted"


def _final_output_status(view: ProjectRunSummaryView) -> str:
    """Return the final named output readiness status."""
    if view.unresolved_matches == 0:
        return "ready"
    if view.accepted_matches:
        return "partial"
    return "blocked"


def _status_text(status: str) -> str:
    """Return styled status text for output tables."""
    styles = {
        "ready": "green",
        "supporting": "cyan",
        "partial": "yellow",
        "blocked": "red",
    }
    return f"[{styles.get(status, 'white')}]{status}[/]"


def _relative_project_output(project_dir: Path, output_path: Path) -> str:
    """Display an output path relative to the project when possible."""
    try:
        return str(output_path.relative_to(project_dir))
    except ValueError:
        return str(output_path)


def _summary_console() -> Console:
    """Build the stdout console used for project run summaries."""
    return cli_console(width=120)
