"""Terminal UI helpers for interactive CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
import re
import time
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Column
from rich.text import Text

from app.core.progress import CliProgressEvent, CliProgressReporter, emit_progress
from app.presentation.cli.errors import run_with_cli_errors

T = TypeVar("T")


def run_with_progress(
    operation: Callable[[CliProgressReporter | None], T],
    *,
    description: str,
    total: int | None = None,
    enabled: bool = True,
) -> T:
    """
    Run a CLI operation with Rich progress when the terminal supports it.

    Args:
        operation: Callable receiving an optional progress reporter.
        description: Initial progress description.
        total: Optional total units for the initial phase.
        enabled: Whether the command allows progress rendering.

    Returns:
        Operation result.
    """
    console = _console()
    if not _should_render_progress(console, enabled):
        return run_with_cli_errors(lambda: operation(None))
    return run_with_cli_errors(lambda: _run_with_rich_progress(operation, console, description, total))


def _run_with_rich_progress(
    operation: Callable[[CliProgressReporter | None], T],
    console: Console,
    description: str,
    total: int | None,
) -> T:
    """
    Render progress for one operation.

    Args:
        operation: Callable receiving the Rich-backed reporter.
        console: Rich console bound to stderr.
        description: Initial progress description.
        total: Optional initial total.

    Returns:
        Operation result.
    """
    display_description, detail_label = _split_progress_description(description)
    with Progress(
        SpinnerColumn(),
        _DescriptionColumn(),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        TextColumn("[dim]step[/]"),
        _StepElapsedColumn(),
        TextColumn("[dim]total[/]"),
        _TotalElapsedColumn(),
        console=console,
    ) as progress:
        now = time.monotonic()
        task_id = progress.add_task(
            display_description,
            total=total,
            step_label="",
            detail_label=detail_label,
            step_started_at=now,
            workflow_started_at=now,
        )

        def report(event: CliProgressEvent) -> None:
            _apply_progress_event(progress, task_id, event)

        return operation(report)


def _apply_progress_event(progress: Progress, task_id, event: CliProgressEvent) -> None:
    """
    Apply one event to a Rich progress task.

    Args:
        progress: Rich progress renderer.
        task_id: Rich task identifier.
        event: Progress event to apply.

    Returns:
        None.
    """
    updates = {}
    if event.step_index is not None:
        updates["step_label"] = _format_step_label(event.step_index, event.step_total)
        updates["step_started_at"] = time.monotonic()
    if event.description is not None:
        display_description, detail_label = _split_progress_description(event.description)
        updates["description"] = display_description
        updates["detail_label"] = detail_label
    if event.reset_total:
        _reset_progress_task(progress, task_id, event, updates)
        return
    elif event.total is not None:
        updates["total"] = event.total
    if event.completed is not None:
        updates["completed"] = event.completed
    if updates:
        progress.update(task_id, **updates)
    if event.advance:
        progress.advance(task_id, event.advance)


def _reset_progress_task(progress: Progress, task_id, event: CliProgressEvent, fields: dict[str, object]) -> None:
    """
    Reset per-step progress without resetting the workflow clock.

    Args:
        progress: Rich progress renderer.
        task_id: Rich task identifier.
        event: Progress event to apply.
        fields: Prepared field updates.

    Returns:
        None.
    """
    # Rich has no public API that clears task.total back to None while keeping
    # the original task clock, so reset the mutable task state directly.
    with progress._lock:
        task = progress._tasks[task_id]
        task._reset()
        task.total = event.total
        task.completed = 0 if event.completed is None else event.completed
        field_updates = dict(fields)
        description = field_updates.pop("description", None)
        if isinstance(description, str):
            task.description = description
        task.fields.update(field_updates)
        if task.total is not None and task.completed >= task.total:
            task.finished_time = task.elapsed
    if event.advance:
        progress.advance(task_id, event.advance)
    else:
        progress.refresh()


class _DescriptionColumn(ProgressColumn):
    """Render the step label, main action, and optional detail line."""

    def get_table_column(self) -> Column:
        """
        Return a non-wrapping table column for progress descriptions.

        Returns:
            Rich table column configuration.
        """
        return Column(width=42, no_wrap=True, overflow="ellipsis")

    def render(self, task: Task) -> Text:
        """
        Render a compact multiline progress description.

        Args:
            task: Rich progress task.

        Returns:
            Description text for the current task.
        """
        text = Text()
        step_label = str(task.fields.get("step_label") or "")
        if step_label:
            text.append(step_label, style="bold cyan")
            text.append(" ")
        text.append(task.description, style="progress.description")
        detail_label = str(task.fields.get("detail_label") or "")
        if detail_label:
            text.append("\n  ")
            text.append(detail_label, style="dim")
        return text


class _StepElapsedColumn(ProgressColumn):
    """Render elapsed time for the current workflow step."""

    def get_table_column(self) -> Column:
        """
        Return a fixed-width elapsed-time column.

        Returns:
            Rich table column configuration.
        """
        return Column(width=7, no_wrap=True)

    def render(self, task: Task) -> Text:
        """
        Render the current step duration.

        Args:
            task: Rich progress task.

        Returns:
            Duration text.
        """
        started_at = task.fields.get("step_started_at")
        if not isinstance(started_at, int | float):
            started_at = task.start_time or time.monotonic()
        return Text(_format_elapsed_seconds(time.monotonic() - float(started_at)), style="progress.elapsed")


class _TotalElapsedColumn(ProgressColumn):
    """Render elapsed time for the whole workflow."""

    def get_table_column(self) -> Column:
        """
        Return a fixed-width elapsed-time column.

        Returns:
            Rich table column configuration.
        """
        return Column(width=7, no_wrap=True)

    def render(self, task: Task) -> Text:
        """
        Render the total workflow duration.

        Args:
            task: Rich progress task.

        Returns:
            Duration text.
        """
        started_at = task.fields.get("workflow_started_at")
        if not isinstance(started_at, int | float):
            started_at = task.start_time or time.monotonic()
        return Text(_format_elapsed_seconds(time.monotonic() - float(started_at)), style="progress.elapsed")


def _format_step_label(step_index: int, step_total: int | None) -> str:
    """
    Format a workflow step label.

    Args:
        step_index: 1-based step index.
        step_total: Optional total step count.

    Returns:
        Display label such as ``[3/8]``.
    """
    if step_total is None:
        return f"[{step_index}]"
    return f"[{step_index}/{step_total}]"


def _format_elapsed_seconds(seconds: float) -> str:
    """
    Format elapsed seconds as ``H:MM:SS``.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Human-readable duration.
    """
    return str(timedelta(seconds=max(0, int(seconds))))


def _split_progress_description(description: str) -> tuple[str, str]:
    """
    Split a long progress description into main action and detail metadata.

    Args:
        description: Raw workflow description.

    Returns:
        Tuple of ``(main_action, detail_label)``.
    """
    segments = [segment.strip() for segment in description.split(" | ") if segment.strip()]
    if not segments:
        return description, ""
    main_action, task_id = _extract_trailing_task_id(segments[0])
    details = [_compact_detail_segment(segment) for segment in segments[1:]]
    if task_id:
        details.append(f"task {_short_task_id(task_id)}")
    return main_action, " | ".join(details)


def _extract_trailing_task_id(text: str) -> tuple[str, str | None]:
    """
    Extract a long parenthesized task id from a description prefix.

    Args:
        text: Candidate description prefix.

    Returns:
        ``(cleaned_text, task_id)`` when an id is found, otherwise ``(text, None)``.
    """
    match = re.fullmatch(r"(?P<label>.+) \((?P<token>[0-9A-Za-z][0-9A-Za-z_.:-]{7,})\)", text)
    if match is None:
        return text, None
    return match.group("label"), match.group("token")


def _short_task_id(task_id: str) -> str:
    """
    Shorten a provider task id for live progress output.

    Args:
        task_id: Provider task identifier.

    Returns:
        Short id safe for compact terminal rendering.
    """
    first_segment = task_id.split("-", 1)[0]
    if len(first_segment) >= 8:
        return first_segment
    return task_id[:12]


def _compact_detail_segment(segment: str) -> str:
    """
    Compact known progress metadata phrases for live rendering.

    Args:
        segment: Raw detail segment.

    Returns:
        Shorter detail segment.
    """
    if segment == "baseline: collecting":
        return "ETA collecting"
    return segment


def _console() -> Console:
    """
    Build the stderr console used for interactive progress.

    Returns:
        Rich console instance.
    """
    return Console(stderr=True, highlight=False)


def _should_render_progress(console: Console, enabled: bool) -> bool:
    """
    Decide whether progress UI should be rendered.

    Args:
        console: Rich console instance.
        enabled: User-facing command switch.

    Returns:
        True when progress should render.
    """
    return enabled and console.is_terminal and not console.is_dumb_terminal
