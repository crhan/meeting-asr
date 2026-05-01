"""Presentation-neutral workflow progress events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
