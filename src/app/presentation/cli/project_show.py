"""Rich rendering for project overview output."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.asr_pricing import asr_cost_from_dict, format_asr_cost
from app.core.project_models import ProjectManifest
from app.core.project_workflow import ProjectWorkflowSummary
from app.presentation.cli.output import cli_console


@dataclass(frozen=True, slots=True)
class ProjectShowView:
    """Presentation data for one project overview."""

    project_dir: Path
    project_ref: str
    manifest: ProjectManifest
    workflow: ProjectWorkflowSummary


@dataclass(frozen=True, slots=True)
class _OutputRow:
    """One project output row."""

    label: str
    kind: str
    path: Path | None


def render_project_show(view: ProjectShowView) -> None:
    """
    Render a project overview for humans.

    Args:
        view: Fully prepared project overview data.

    Returns:
        None.
    """
    console = _show_console()
    console.print(_status_panel(view))
    console.print(_details_table(view))
    summary_panel = _summary_panel(view)
    if summary_panel is not None:
        console.print(summary_panel)
    console.print(_outputs_table(view))
    console.print(_commands_table(view))


def _status_panel(view: ProjectShowView) -> Panel:
    """Build the high-level project status panel."""
    style = _state_style(view.workflow.state_key)
    grid = Table.grid(expand=True)
    grid.add_column(ratio=3)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(f"[bold]{view.manifest.title}[/]", f"[bold cyan]{view.project_ref}[/]")
    grid.add_row(f"[dim]{view.project_dir}[/]", f"[{style}]{view.workflow.state}[/]")
    return Panel(grid, title=f"[bold {style}]Project[/]", border_style=style, expand=False)


def _details_table(view: ProjectShowView) -> Table:
    """Build key project metadata rows."""
    table = Table(title="Details", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("Item", style="bold", no_wrap=True)
    table.add_column("Value")
    for item, value in _detail_rows(view):
        table.add_row(item, value)
    return table


def _detail_rows(view: ProjectShowView) -> list[tuple[str, str]]:
    """Return scan-friendly project metadata rows."""
    manifest = view.manifest
    rows = [
        ("Project ID", manifest.project_id),
        ("State", view.workflow.state),
        ("Updated", _compact_timestamp(manifest.updated_at)),
        ("Source file", manifest.source.original_path or manifest.source.path),
    ]
    rows.extend(
        [
            ("Duration", _duration_label(manifest.audio.get("duration_seconds"))),
            ("Speakers", _speakers_label(view)),
            ("ASR cost", _cost_label(manifest.asr.get("cost"))),
        ]
    )
    if view.workflow.missing:
        rows.append(("Missing", ", ".join(view.workflow.missing)))
    return rows


def _outputs_table(view: ProjectShowView) -> Table:
    """Build output artifact rows."""
    table = Table(title="Outputs", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("Artifact", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Command", no_wrap=True)
    for row in _output_rows(view):
        table.add_row(row.label, _output_status(row), _view_command(view, row))
    return table


def _summary_panel(view: ProjectShowView) -> Panel | None:
    """Build a direct meeting-summary preview when the artifact exists."""
    row = _manifest_output(view.project_dir, view.manifest, "Meeting summary", "summary", ("meeting_summary",))
    if row.path is None:
        return None
    try:
        content = _display_summary_content(row.path.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not content:
        return None
    return Panel(Markdown(content), title="[bold]Meeting Summary[/]", border_style="cyan", expand=False)


def _output_rows(view: ProjectShowView) -> list[_OutputRow]:
    """Return project output rows in user-facing order."""
    root = view.project_dir
    return [
        _manifest_output(
            root,
            view.manifest,
            "Final transcript",
            "auto",
            ("corrected_named_transcript", "named_transcript", "anonymous_transcript", "plain_transcript"),
        ),
        _manifest_output(
            root,
            view.manifest,
            "Final subtitles",
            "srt",
            ("corrected_named_subtitle", "named_subtitle", "subtitle"),
        ),
        _manifest_output(root, view.manifest, "Speaker transcript", "speakers", ("anonymous_transcript",)),
        _manifest_output(root, view.manifest, "Plain transcript", "plain", ("plain_transcript",)),
    ]


def _commands_table(view: ProjectShowView) -> Table:
    """Build copyable next commands."""
    table = Table(title="Commands", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Action", style="bold", no_wrap=True)
    table.add_column("Command")
    for index, (action, command) in enumerate(_command_rows(view), start=1):
        table.add_row(str(index), action, command)
    return table


def _command_rows(view: ProjectShowView) -> list[tuple[str, str]]:
    """Return common project follow-up commands."""
    quoted_ref = shlex.quote(view.project_ref)
    rows = [
        ("Next", view.workflow.next_command),
        ("Show transcript", f"meeting-asr project transcript show {quoted_ref}"),
        ("List outputs", f"meeting-asr project transcript list {quoted_ref}"),
        ("Review speakers", f"meeting-asr project review {quoted_ref}"),
        ("Preview subtitles", f"meeting-asr project speakers preview {quoted_ref}"),
    ]
    return _unique_command_rows(rows)


def _manifest_output(root: Path, manifest: ProjectManifest, label: str, kind: str, keys: tuple[str, ...]) -> _OutputRow:
    """Resolve the first existing manifest output for a row."""
    for key in keys:
        value = manifest.outputs.get(key)
        if isinstance(value, str) and value:
            path = _stored_path(root, value)
            if path.exists():
                return _OutputRow(label, kind, path)
    return _OutputRow(label, kind, None)


def _display_summary_content(content: str) -> str:
    """Return meeting summary markdown without generation metadata."""
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("模型：", "Model:")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _output_status(row: _OutputRow) -> str:
    """Return styled artifact availability."""
    return "[green]ready[/]" if row.path else "[red]missing[/]"


def _view_command(view: ProjectShowView, row: _OutputRow) -> str:
    """Return a complete command for viewing one artifact."""
    if row.path is None:
        return "-"
    return f"meeting-asr project transcript show {shlex.quote(view.project_ref)} --kind {row.kind}"


def _unique_command_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate commands while preserving order."""
    seen: set[str] = set()
    unique_rows = []
    for action, command in rows:
        if command in seen:
            continue
        seen.add(command)
        unique_rows.append((action, command))
    return unique_rows


def _state_style(state_key: str) -> str:
    """Return a Rich style for a workflow state."""
    styles = {
        "created": "yellow",
        "prepared": "yellow",
        "transcribed": "cyan",
        "completed": "green",
        "corrected": "green",
        "broken": "red",
    }
    return styles.get(state_key, "white")


def _speakers_label(view: ProjectShowView) -> str:
    """Return detected and mapped speaker summary."""
    manifest = view.manifest
    detected = manifest.speakers.get("detected_ids") or []
    mapped = manifest.speakers.get("mapped") or {}
    active_ids = _active_speaker_ids(view.project_dir)
    visible_names = _mapped_speaker_names(mapped, active_ids)
    names = ", ".join(visible_names)
    detected_count = len(active_ids) if active_ids is not None else len(detected)
    ignored_count = _ignored_speaker_count(detected, active_ids)
    base = f"{detected_count} detected"
    if ignored_count:
        base = f"{base}; ignored {ignored_count} low-info"
    return f"{base}; {names}" if names else base


def _cost_label(value: object) -> str:
    """Return formatted ASR cost from manifest data."""
    if not isinstance(value, dict):
        return "-"
    try:
        return format_asr_cost(asr_cost_from_dict(value))
    except (TypeError, ValueError, KeyError):
        return "-"


def _active_speaker_ids(project_dir: Path) -> set[int] | None:
    """Return speaker ids that survive low-information filtering."""
    path = project_dir / "asr" / "sentences.json"
    if not path.exists():
        return None
    try:
        from app.speaker_labeling import load_transcript_result

        return set(load_transcript_result(path).detected_speakers)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _mapped_speaker_names(mapped: object, active_ids: set[int] | None) -> list[str]:
    """Return mapped names for active speakers only."""
    if not isinstance(mapped, dict):
        return []
    names = []
    for speaker_id, name in sorted(mapped.items(), key=lambda item: str(item[0])):
        try:
            numeric_id = int(speaker_id)
        except (TypeError, ValueError):
            continue
        if active_ids is not None and numeric_id not in active_ids:
            continue
        names.append(str(name))
    return names


def _ignored_speaker_count(detected: object, active_ids: set[int] | None) -> int:
    """Return how many manifest speakers are now ignored as low-information tracks."""
    if active_ids is None or not isinstance(detected, list):
        return 0
    detected_ids = set()
    for speaker_id in detected:
        try:
            detected_ids.add(int(speaker_id))
        except (TypeError, ValueError):
            continue
    return len(detected_ids - active_ids)


def _duration_label(value: object) -> str:
    """Return a compact duration label."""
    if not isinstance(value, int | float):
        return "-"
    seconds = int(value)
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{remaining_minutes:02d}m{remaining_seconds:02d}s"
    return f"{remaining_minutes}m{remaining_seconds:02d}s"


def _compact_timestamp(value: str) -> str:
    """Return a compact timestamp label."""
    return value[:16].replace("T", " ")


def _stored_path(root: Path, value: str) -> Path:
    """Resolve a project-stored path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _show_console() -> Console:
    """Build the stdout console used for project overview tables."""
    return cli_console(width=120)
