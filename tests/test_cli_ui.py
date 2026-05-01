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


def test_total_elapsed_column_uses_workflow_clock(monkeypatch) -> None:
    """Total elapsed time should keep moving even when the Rich task is finished."""
    monkeypatch.setattr(cli_ui.time, "monotonic", lambda: 100.0)
    progress = Progress(console=Console(file=io.StringIO()))
    task_id = progress.add_task("done", total=1, completed=1, workflow_started_at=40.0)
    task = progress.tasks[0]

    progress.update(task_id, completed=1)

    assert task.finished
    assert cli_ui._TotalElapsedColumn().render(task).plain == "0:01:00"
