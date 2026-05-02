"""Tests for Rich-backed CLI progress helpers."""

from __future__ import annotations

import io

from rich.console import Console
from rich.progress import Progress

from app.presentation.cli import progress as cli_ui
from app.core.progress import CliProgressEvent


def test_reset_progress_event_clears_finished_unknown_total() -> None:
    """Resetting to an unknown total should not keep stale 100% task state."""
    progress = Progress(console=Console(file=io.StringIO()))
    task_id = progress.add_task(
        "submitted",
        total=1,
        step_label="[4/10]",
        step_started_at=10.0,
        workflow_started_at=1.0,
    )
    progress.update(task_id, completed=1)
    task = progress.tasks[0]
    assert task.finished

    cli_ui._apply_progress_event(
        progress,
        task_id,
        CliProgressEvent(
            "Waiting for DashScope transcription",
            step_index=5,
            step_total=10,
            reset_total=True,
        ),
    )
    task = progress.tasks[0]

    assert task.total is None
    assert task.completed == 0
    assert not task.finished
    assert task.description == "Waiting for DashScope transcription"
    assert "description" not in task.fields
    assert task.fields["step_label"] == "[5/10]"
    assert task.fields["workflow_started_at"] == 1.0


def test_long_progress_description_moves_metadata_to_detail_line() -> None:
    """Long poll metadata should not stay in the primary progress prefix."""
    progress = Progress(console=Console(file=io.StringIO()))
    task_id = progress.add_task("submitted", total=1, step_label="[4/10]", detail_label="")

    cli_ui._apply_progress_event(
        progress,
        task_id,
        CliProgressEvent(
            "Waiting for DashScope ASR (0204d7fb-f068-46b2-995c-540aa2f1e8d7) "
            "| RUNNING | baseline: collecting",
            step_index=5,
            step_total=10,
            reset_total=True,
        ),
    )
    task = progress.tasks[0]
    rendered = cli_ui._DescriptionColumn().render(task)

    assert task.description == "Waiting for DashScope ASR"
    assert task.fields["detail_label"] == "RUNNING | ETA collecting | task 0204d7fb"
    assert rendered.plain.splitlines() == [
        "[5/10] Waiting for DashScope ASR",
        "  RUNNING | ETA collecting | task 0204d7fb",
    ]


def test_progress_description_split_handles_eta_without_task_id() -> None:
    """ETA metadata should become detail even when there is no task id."""
    main_action, detail_label = cli_ui._split_progress_description("Uploading audio to OSS | ETA ~8s | medium n=3")

    assert main_action == "Uploading audio to OSS"
    assert detail_label == "ETA ~8s | medium n=3"


def test_workflow_renderer_keeps_step_rows_and_total_row(monkeypatch) -> None:
    """Multi-step workflows should keep completed steps visible with frozen duration."""
    now = 0.0
    monkeypatch.setattr(cli_ui.time, "monotonic", lambda: now)
    progress = Progress(console=Console(file=io.StringIO()))
    fallback_id = progress.add_task(
        "initial",
        row_kind="step",
        step_state="active",
        step_started_at=0.0,
        workflow_started_at=0.0,
    )
    renderer = cli_ui._RichProgressRenderer(progress, fallback_id, 0.0)

    renderer.report(CliProgressEvent("Step one", step_index=1, step_total=3, reset_total=True))
    now = 8.0
    renderer.report(CliProgressEvent("Step two", step_index=2, step_total=3, reset_total=True))
    now = 11.0

    step_one = progress._tasks[renderer.step_task_ids[1]]
    step_two = progress._tasks[renderer.step_task_ids[2]]
    step_three = progress._tasks[renderer.step_task_ids[3]]
    total = progress._tasks[renderer.total_task_id]

    assert not progress._tasks[fallback_id].visible
    assert step_one.fields["step_state"] == "done"
    assert step_two.fields["step_state"] == "active"
    assert step_three.fields["step_state"] == "pending"
    assert cli_ui._StepElapsedColumn().render(step_one).plain == "0:00:08"
    assert cli_ui._StepElapsedColumn().render(step_two).plain == "0:00:03"
    assert cli_ui._TotalElapsedColumn().render(total).plain == "0:00:11"


def test_total_elapsed_column_uses_workflow_clock(monkeypatch) -> None:
    """Total elapsed time should keep moving even when the Rich task is finished."""
    monkeypatch.setattr(cli_ui.time, "monotonic", lambda: 100.0)
    progress = Progress(console=Console(file=io.StringIO()))
    task_id = progress.add_task("done", total=1, completed=1, workflow_started_at=40.0, row_kind="total")
    task = progress.tasks[0]

    progress.update(task_id, completed=1)

    assert task.finished
    assert cli_ui._TotalElapsedColumn().render(task).plain == "0:01:00"
