"""Terminal UI helpers for interactive CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
import re
import shutil
import time
from typing import TypeVar

from rich.console import Console
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
)
from rich.table import Column
from rich.text import Text

from app.core.progress import CliProgressEvent, CliProgressReporter
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.output import cli_console

T = TypeVar("T")
PROGRESS_MAX_WIDTH = 120
PROGRESS_DESCRIPTION_MIN_WIDTH = 24
PROGRESS_DESCRIPTION_BASE_WIDTH = 36
PROGRESS_DESCRIPTION_MAX_WIDTH = 70
PROGRESS_BAR_MIN_WIDTH = 6
PROGRESS_BAR_BASE_WIDTH = 10
PROGRESS_BAR_MAX_WIDTH = 40
PROGRESS_ELAPSED_WIDTH = 8
PROGRESS_COLUMN_GAP_WIDTH = 2


def run_with_progress(
    operation: Callable[[CliProgressReporter | None], T],
    *,
    description: str,
    total: int | None = None,
    enabled: bool = True,
    structured_log: bool = False,
) -> T:
    """
    Run a CLI operation with Rich progress when the terminal supports it.

    Args:
        operation: Callable receiving an optional progress reporter.
        description: Initial progress description.
        total: Optional total units for the initial phase.
        enabled: Whether the command allows progress rendering.
        structured_log: Whether to print durable stage/heartbeat lines for agents.

    Returns:
        Operation result.
    """
    console = _console()
    if not enabled:
        reporter = _LineProgressReporter(console) if structured_log else None
        return run_with_cli_errors(lambda: operation(reporter))
    if not _should_render_progress(console, enabled):
        reporter = _LineProgressReporter(console) if structured_log else None
        return run_with_cli_errors(lambda: operation(reporter))
    return run_with_cli_errors(lambda: _run_with_rich_progress(operation, console, description, total, structured_log))


def _run_with_rich_progress(
    operation: Callable[[CliProgressReporter | None], T],
    console: Console,
    description: str,
    total: int | None,
    structured_log: bool,
) -> T:
    """
    Render progress for one operation.

    Args:
        operation: Callable receiving the Rich-backed reporter.
        console: Rich console bound to stderr.
        description: Initial progress description.
        total: Optional initial total.
        structured_log: Whether to print durable stage/heartbeat lines for agents.

    Returns:
        Operation result.
    """
    display_description, detail_label = _split_progress_description(description)
    layout = _progress_layout(console)
    with Progress(
        _DescriptionColumn(layout.description_width),
        _WorkflowBarColumn(layout.bar_width),
        _ElapsedColumn(layout.elapsed_width),
        console=console,
    ) as progress:
        now = time.monotonic()
        task_id = progress.add_task(
            display_description,
            total=total,
            step_label="",
            detail_label=detail_label,
            row_kind="step",
            step_state="active",
            step_started_at=now,
            workflow_started_at=now,
        )
        renderer = _RichProgressRenderer(progress, task_id, now, structured_log=structured_log)

        def report(event: CliProgressEvent) -> None:
            renderer.report(event)

        result = operation(report)
        renderer.finish()
        return result


@dataclass(frozen=True, slots=True)
class _ProgressLayout:
    """Column widths used by the workflow progress renderer."""

    console_width: int
    description_width: int
    bar_width: int
    elapsed_width: int = PROGRESS_ELAPSED_WIDTH


@dataclass(frozen=True, slots=True)
class _LineProgressReporter:
    """Print structured progress lines when Rich live rendering is unavailable."""

    console: Console

    def __call__(self, event: CliProgressEvent) -> None:
        """
        Render one structured log event.

        Args:
            event: Progress event emitted by workflow code.

        Returns:
            None.
        """
        line = _progress_log_line(event)
        if line:
            self.console.print(line, soft_wrap=True)


@dataclass(slots=True)
class _RichProgressRenderer:
    """Stateful Rich renderer for single-step and workflow progress."""

    progress: Progress
    fallback_task_id: int
    workflow_started_at: float
    structured_log: bool = False
    workflow_enabled: bool = False
    step_total: int | None = None
    current_step_index: int | None = None
    total_task_id: int | None = None
    step_task_ids: dict[int, int] = field(default_factory=dict)

    def report(self, event: CliProgressEvent) -> None:
        """
        Apply one workflow progress event.

        Args:
            event: Progress event emitted by core workflow code.

        Returns:
            None.
        """
        self._emit_log(event)
        if _is_workflow_event(event):
            self._apply_workflow_event(event)
            return
        if self.workflow_enabled and self.current_step_index is not None:
            _apply_progress_event(self.progress, self.step_task_ids[self.current_step_index], event)
            return
        _apply_progress_event(self.progress, self.fallback_task_id, event)

    def _emit_log(self, event: CliProgressEvent) -> None:
        """Print structured observability lines above the live progress UI."""
        if not self.structured_log:
            return
        line = _progress_log_line(event)
        if line:
            self.progress.console.print(line, soft_wrap=True)

    def finish(self) -> None:
        """
        Mark the current workflow step complete before Rich writes the final frame.

        Returns:
            None.
        """
        if self.workflow_enabled and self.current_step_index is not None:
            self._finish_step(self.current_step_index)
            self.current_step_index = None

    def _apply_workflow_event(self, event: CliProgressEvent) -> None:
        """Apply an event that identifies a workflow step."""
        self._ensure_workflow(event.step_total or event.step_index or 1, event.step_descriptions)
        step_index = event.step_index or 1
        if step_index != self.current_step_index:
            if self.current_step_index is not None:
                self._finish_step(self.current_step_index)
            self.current_step_index = step_index
            self._start_step(step_index)
        _apply_progress_event(self.progress, self.step_task_ids[step_index], event)

    def _ensure_workflow(self, step_total: int, step_descriptions: tuple[str, ...]) -> None:
        """Create persistent step rows once."""
        if self.workflow_enabled:
            self._apply_workflow_plan(step_descriptions)
            return
        self.workflow_enabled = True
        self.step_total = max(1, step_total)
        self.progress.update(self.fallback_task_id, visible=False)
        for index in range(1, self.step_total + 1):
            self.step_task_ids[index] = self.progress.add_task(
                _planned_step_description(index, step_descriptions),
                total=1,
                completed=0,
                step_label=_format_step_label(index, self.step_total),
                detail_label="",
                row_kind="step",
                step_state="pending",
                step_started_at=None,
                step_finished_at=None,
                workflow_started_at=self.workflow_started_at,
            )
        self.total_task_id = self.progress.add_task(
            "Total",
            total=None,
            completed=0,
            step_label="",
            detail_label="",
            row_kind="total",
            step_state="active",
            step_started_at=None,
            step_finished_at=None,
            workflow_started_at=self.workflow_started_at,
        )

    def _apply_workflow_plan(self, step_descriptions: tuple[str, ...]) -> None:
        """Fill placeholder descriptions when a workflow plan arrives late."""
        if not step_descriptions:
            return
        for index, description in enumerate(step_descriptions, start=1):
            task_id = self.step_task_ids.get(index)
            if task_id is None:
                continue
            task = self.progress._tasks[task_id]
            if task.description == f"Step {index}":
                self.progress.update(task_id, description=description)

    def _start_step(self, step_index: int) -> None:
        """Mark one workflow step active."""
        task_id = self.step_task_ids[step_index]
        self.progress.update(
            task_id,
            step_state="active",
            step_started_at=time.monotonic(),
            step_finished_at=None,
        )

    def _finish_step(self, step_index: int) -> None:
        """Mark one workflow step complete and freeze its elapsed duration."""
        task_id = self.step_task_ids[step_index]
        task = self.progress._tasks[task_id]
        total = task.total or 1
        self.progress.update(
            task_id,
            total=total,
            completed=total,
            step_state="done",
            step_finished_at=time.monotonic(),
        )


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

    def __init__(self, width: int = PROGRESS_DESCRIPTION_BASE_WIDTH) -> None:
        """
        Create a description column.

        Args:
            width: Column width in terminal cells.
        """
        super().__init__()
        self._width = width

    def get_table_column(self) -> Column:
        """
        Return a non-wrapping table column for progress descriptions.

        Returns:
            Rich table column configuration.
        """
        return Column(width=self._width, no_wrap=True, overflow="ellipsis")

    def render(self, task: Task) -> Text:
        """
        Render a compact multiline progress description.

        Args:
            task: Rich progress task.

        Returns:
            Description text for the current task.
        """
        text = Text()
        state_label = _task_state_label(task)
        if state_label:
            label, style = state_label
            text.append(label, style=style)
            text.append(" ")
        step_label = str(task.fields.get("step_label") or "")
        if step_label:
            text.append(step_label, style="bold cyan")
            text.append(" ")
        text.append(task.description, style="progress.description")
        detail_label = str(task.fields.get("detail_label") or "")
        if detail_label:
            text.append("\n")
            text.append(" " * _detail_indent_width(task))
            text.append(detail_label, style="dim")
        return text


class _WorkflowBarColumn(ProgressColumn):
    """Render progress bars only for active or completed step rows."""

    def __init__(self, width: int = PROGRESS_BAR_BASE_WIDTH) -> None:
        """
        Create a workflow bar column.

        Args:
            width: Bar width in terminal cells.
        """
        super().__init__()
        self._width = width
        self._bar = BarColumn(bar_width=width)

    def get_table_column(self) -> Column:
        """
        Return a fixed-width bar column.

        Returns:
            Rich table column configuration.
        """
        return Column(width=self._width, no_wrap=True)

    def render(self, task: Task) -> RenderableType:
        """
        Render a bar for active or completed steps.

        Args:
            task: Rich progress task.

        Returns:
            Bar renderable or blank text.
        """
        if task.fields.get("row_kind") == "total" or task.fields.get("step_state") == "pending":
            return Text("")
        return self._bar.render(task)


class _ElapsedColumn(ProgressColumn):
    """Render elapsed time for step rows and the total row in one column."""

    def __init__(self, width: int = PROGRESS_ELAPSED_WIDTH) -> None:
        """
        Create an elapsed-time column.

        Args:
            width: Column width in terminal cells.
        """
        super().__init__()
        self._width = width

    def get_table_column(self) -> Column:
        """
        Return a fixed-width elapsed-time column.

        Returns:
            Rich table column configuration.
        """
        return Column(width=self._width, no_wrap=True)

    def render(self, task: Task) -> Text:
        """
        Render the current step or workflow duration.

        Args:
            task: Rich progress task.

        Returns:
            Duration text.
        """
        if task.fields.get("row_kind") == "total":
            return _render_total_elapsed(task)
        if task.fields.get("step_state") == "pending":
            return Text("")
        return _render_step_elapsed(task)


class _StepElapsedColumn(_ElapsedColumn):
    """Backward-compatible private alias for step elapsed rendering."""

    def render(self, task: Task) -> Text:
        """
        Render only per-step elapsed time.

        Args:
            task: Rich progress task.

        Returns:
            Duration text or blank text for non-step rows.
        """
        if task.fields.get("row_kind") == "total" or task.fields.get("step_state") == "pending":
            return Text("")
        return _render_step_elapsed(task)


class _TotalElapsedColumn(_ElapsedColumn):
    """Backward-compatible private alias for total elapsed rendering."""

    def render(self, task: Task) -> Text:
        """
        Render only total elapsed time.

        Args:
            task: Rich progress task.

        Returns:
            Duration text or blank text for non-total rows.
        """
        if task.fields.get("row_kind") != "total":
            return Text("")
        return _render_total_elapsed(task)


def _render_step_elapsed(task: Task) -> Text:
    """
    Render elapsed time for one workflow step.

    Args:
        task: Rich progress task.

    Returns:
        Duration text.
    """
    started_at = task.fields.get("step_started_at")
    if not isinstance(started_at, int | float):
        started_at = task.start_time or time.monotonic()
    finished_at = task.fields.get("step_finished_at")
    end_at = float(finished_at) if isinstance(finished_at, int | float) else time.monotonic()
    return Text(_format_elapsed_seconds(end_at - float(started_at)), style="progress.elapsed")


def _render_total_elapsed(task: Task) -> Text:
    """
    Render elapsed time for the whole workflow.

    Args:
        task: Rich progress task.

    Returns:
        Duration text.
    """
    started_at = task.fields.get("workflow_started_at")
    if not isinstance(started_at, int | float):
        started_at = task.start_time or time.monotonic()
    return Text(_format_elapsed_seconds(time.monotonic() - float(started_at)), style="progress.elapsed")


def _progress_layout(console: Console) -> _ProgressLayout:
    """
    Compute progress column widths from the current console width.

    Args:
        console: Rich console used for rendering.

    Returns:
        Width allocation for progress columns.
    """
    target_width = max(1, min(console.size.width, PROGRESS_MAX_WIDTH))
    fixed_width = PROGRESS_DESCRIPTION_BASE_WIDTH + PROGRESS_BAR_BASE_WIDTH
    fixed_width += PROGRESS_ELAPSED_WIDTH + PROGRESS_COLUMN_GAP_WIDTH
    if target_width <= fixed_width:
        bar_width = max(PROGRESS_BAR_MIN_WIDTH, min(PROGRESS_BAR_BASE_WIDTH, target_width // 6))
        description_width = max(
            PROGRESS_DESCRIPTION_MIN_WIDTH,
            target_width - bar_width - PROGRESS_ELAPSED_WIDTH - PROGRESS_COLUMN_GAP_WIDTH,
        )
        return _ProgressLayout(target_width, description_width, bar_width)

    extra_width = target_width - fixed_width
    description_extra = min(
        PROGRESS_DESCRIPTION_MAX_WIDTH - PROGRESS_DESCRIPTION_BASE_WIDTH,
        int(extra_width * 0.65),
    )
    bar_extra = min(PROGRESS_BAR_MAX_WIDTH - PROGRESS_BAR_BASE_WIDTH, extra_width - description_extra)
    return _ProgressLayout(
        console_width=target_width,
        description_width=PROGRESS_DESCRIPTION_BASE_WIDTH + description_extra,
        bar_width=PROGRESS_BAR_BASE_WIDTH + bar_extra,
    )


def _detail_indent_width(task: Task) -> int:
    """
    Return the indentation that aligns detail text under the action label.

    Args:
        task: Rich progress task.

    Returns:
        Number of leading spaces for the detail line.
    """
    indent_width = 0
    state_label = _task_state_label(task)
    if state_label:
        indent_width += len(state_label[0]) + 1
    step_label = str(task.fields.get("step_label") or "")
    if step_label:
        indent_width += len(step_label) + 1
    return indent_width


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


def _planned_step_description(step_index: int, step_descriptions: tuple[str, ...]) -> str:
    """
    Return a planned step description, falling back to a numbered placeholder.

    Args:
        step_index: 1-based step number.
        step_descriptions: Full workflow step plan.

    Returns:
        Description for the row.
    """
    if 1 <= step_index <= len(step_descriptions):
        description = step_descriptions[step_index - 1].strip()
        if description:
            return description
    return f"Step {step_index}"


def _task_state_label(task: Task) -> tuple[str, str] | None:
    """
    Return the compact state label embedded in the description column.

    Args:
        task: Rich progress task.

    Returns:
        ``(label, style)`` or None.
    """
    if task.fields.get("row_kind") == "total":
        return ("T", "bold cyan")
    state = str(task.fields.get("step_state") or "")
    if state == "done":
        return ("ok", "green")
    if state == "pending":
        return (".", "dim")
    if state == "active":
        return (">", "bold cyan")
    return None


def _is_workflow_event(event: CliProgressEvent) -> bool:
    """
    Return whether an event should switch the renderer into workflow mode.

    Args:
        event: Progress event.

    Returns:
        True for numbered multi-step workflow events.
    """
    return event.step_index is not None and event.step_total is not None and event.step_total > 1


def _progress_log_line(event: CliProgressEvent) -> str:
    """
    Format a structured progress line for durable logs.

    Args:
        event: Progress event to format.

    Returns:
        Human-readable single line, or empty string when the event is not a log event.
    """
    if not event.log_kind:
        return ""
    parts = [f"[{event.timestamp or '-'}]", f"{event.log_kind}={event.stage or '-'}"]
    if event.project_id:
        parts.append(f"project_id={event.project_id}")
    if event.project_path:
        parts.append(f"project={event.project_path}")
    if event.input_file:
        parts.append(f"input={event.input_file}")
    if event.elapsed_seconds is not None:
        parts.append(f"elapsed={_format_elapsed_seconds(event.elapsed_seconds)}")
    if event.last_success:
        parts.append(f"last={_safe_log_value('last_success', event.last_success)}")
    if event.next_action:
        parts.append(f"next={_safe_log_value('next_action', event.next_action)}")
    for key, value in event.log_fields:
        parts.append(f"{key}={_safe_log_value(key, value)}")
    return " ".join(parts)


def _safe_log_value(key: str, value: object) -> str:
    """
    Redact URL query strings and token-like values in progress logs.

    Args:
        key: Field name.
        value: Field value.

    Returns:
        Safe display string.
    """
    text = str(value)
    lowered = key.lower()
    if any(secret in lowered for secret in ("token", "secret", "signature", "access_key")):
        return "<redacted>"
    if "url" in lowered and "?" in text:
        return text.split("?", 1)[0] + "?<redacted>"
    return text


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
    return cli_console(stderr=True, width=_progress_console_width())


def _progress_console_width() -> int:
    """
    Return the capped console width used by live progress rendering.

    Returns:
        Terminal width capped to keep progress readable on wide screens.
    """
    terminal_width = shutil.get_terminal_size((PROGRESS_MAX_WIDTH, 24)).columns
    return max(1, min(terminal_width, PROGRESS_MAX_WIDTH))


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
