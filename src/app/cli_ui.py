"""Terminal UI helpers for interactive CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
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
from rich.text import Text

from app.cli_errors import run_with_cli_errors

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CliProgressEvent:
    """One progress update emitted by a workflow."""

    description: str | None = None
    total: int | None = None
    completed: int | None = None
    advance: int = 0
    step_index: int | None = None
    step_total: int | None = None
    reset_total: bool = False


CliProgressReporter = Callable[[CliProgressEvent], None]


def emit_progress(
    reporter: CliProgressReporter | None,
    description: str | None = None,
    *,
    total: int | None = None,
    completed: int | None = None,
    advance: int = 0,
    step_index: int | None = None,
    step_total: int | None = None,
    reset_total: bool = False,
) -> None:
    """
    Emit one progress event when a reporter is available.

    Args:
        reporter: Optional progress reporter callback.
        description: Current task description.
        total: Optional total work units for the current phase.
        completed: Optional absolute completed work units.
        advance: Optional relative completed work units.
        step_index: Optional 1-based workflow step number.
        step_total: Optional total workflow step count.
        reset_total: Reset the current progress bar before applying this event.

    Returns:
        None.
    """
    if reporter is None:
        return
    reporter(CliProgressEvent(description, total, completed, advance, step_index, step_total, reset_total))


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
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[step_label]}[/] [progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]step[/]"),
        _StepElapsedColumn(),
        TextColumn("[dim]total[/]"),
        _TotalElapsedColumn(),
        console=console,
    ) as progress:
        now = time.monotonic()
        task_id = progress.add_task(
            description,
            total=total,
            step_label="",
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
        updates["description"] = event.description
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
        if event.description is not None:
            task.description = event.description
        field_updates = dict(fields)
        field_updates.pop("description", None)
        task.fields.update(field_updates)
        if task.total is not None and task.completed >= task.total:
            task.finished_time = task.elapsed
    if event.advance:
        progress.advance(task_id, event.advance)
    else:
        progress.refresh()


class _StepElapsedColumn(ProgressColumn):
    """Render elapsed time for the current workflow step."""

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
