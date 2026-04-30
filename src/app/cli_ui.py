"""Terminal UI helpers for interactive CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from app.cli_errors import run_with_cli_errors

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CliProgressEvent:
    """One progress update emitted by a workflow."""

    description: str | None = None
    total: int | None = None
    completed: int | None = None
    advance: int = 0


CliProgressReporter = Callable[[CliProgressEvent], None]


def emit_progress(
    reporter: CliProgressReporter | None,
    description: str | None = None,
    *,
    total: int | None = None,
    completed: int | None = None,
    advance: int = 0,
) -> None:
    """
    Emit one progress event when a reporter is available.

    Args:
        reporter: Optional progress reporter callback.
        description: Current task description.
        total: Optional total work units for the current phase.
        completed: Optional absolute completed work units.
        advance: Optional relative completed work units.

    Returns:
        None.
    """
    if reporter is None:
        return
    reporter(CliProgressEvent(description, total, completed, advance))


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
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=total)

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
    if event.description is not None:
        updates["description"] = event.description
    if event.total is not None:
        updates["total"] = event.total
    if event.completed is not None:
        updates["completed"] = event.completed
    if updates:
        progress.update(task_id, **updates)
    if event.advance:
        progress.advance(task_id, event.advance)


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
