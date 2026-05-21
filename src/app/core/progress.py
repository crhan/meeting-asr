"""Presentation-neutral workflow progress events."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    step_descriptions: tuple[str, ...] = ()
    log_kind: str | None = None
    stage: str | None = None
    project_id: str | None = None
    project_path: str | None = None
    input_file: str | None = None
    timestamp: str | None = None
    elapsed_seconds: float | None = None
    last_success: str | None = None
    next_action: str | None = None
    log_fields: tuple[tuple[str, str], ...] = ()


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
    step_descriptions: Sequence[str] | None = None,
    log_kind: str | None = None,
    stage: str | None = None,
    project_id: str | None = None,
    project_path: str | None = None,
    input_file: str | None = None,
    timestamp: str | None = None,
    elapsed_seconds: float | None = None,
    last_success: str | None = None,
    next_action: str | None = None,
    log_fields: Sequence[tuple[str, object]] | None = None,
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
        step_descriptions: Optional full workflow step plan.
        log_kind: Optional structured log kind, such as ``stage`` or ``heartbeat``.
        stage: Stable stage name for long-running workflow observability.
        project_id: Project identifier associated with the event.
        project_path: Project root path associated with the event.
        input_file: Source input path associated with the event.
        timestamp: Local timestamp for the event.
        elapsed_seconds: Elapsed seconds for heartbeat events.
        last_success: Last successful operation for heartbeat events.
        next_action: Next poll or batch action for heartbeat events.
        log_fields: Extra non-secret key/value fields for structured progress logs.

    Returns:
        None.
    """
    if reporter is None:
        return
    reporter(
        CliProgressEvent(
            description=description,
            total=total,
            completed=completed,
            advance=advance,
            step_index=step_index,
            step_total=step_total,
            reset_total=reset_total,
            step_descriptions=tuple(step_descriptions or ()),
            log_kind=log_kind,
            stage=stage,
            project_id=project_id,
            project_path=project_path,
            input_file=input_file,
            timestamp=timestamp,
            elapsed_seconds=elapsed_seconds,
            last_success=last_success,
            next_action=next_action,
            log_fields=tuple(
                (str(key), str(value)) for key, value in (log_fields or ())
            ),
        )
    )
