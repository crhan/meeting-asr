"""Artifact-gated stage pipeline for the ``project run`` workflow.

``project run`` is a converging pipeline, not a one-shot script: every stage
declares how many workflow steps it spans and, optionally, a *satisfaction
probe* that reports why the stage can be skipped (its outputs already exist
and are current). Re-running a project therefore resumes from what is missing
instead of re-paying for completed work, and the step numbering shown to the
user is derived from the plan instead of hand-maintained arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.core.progress import CliProgressReporter, emit_progress


@dataclass(frozen=True, slots=True)
class StageRun:
    """Step-numbering context handed to one executing stage."""

    step_index: int
    step_total: int


@dataclass(frozen=True, slots=True)
class RunStage:
    """One ``project run`` pipeline stage.

    Attributes:
        key: Stable machine key (also used in skip reporting).
        description: User-facing progress description.
        execute: Stage body; receives the step-numbering context.
        step_span: Number of workflow steps this stage occupies.
        satisfied: Optional probe returning a human-readable skip reason when
            the stage's outputs already exist, or ``None`` to run the stage.
        sub_descriptions: Step descriptions when the stage spans several steps.
    """

    key: str
    description: str
    execute: Callable[[StageRun], None]
    step_span: int = 1
    satisfied: Callable[[], str | None] | None = None
    sub_descriptions: tuple[str, ...] = field(default=())


def pipeline_step_descriptions(stages: Sequence[RunStage]) -> tuple[str, ...]:
    """Return the flattened per-step descriptions for a stage plan."""
    descriptions: list[str] = []
    for stage in stages:
        if stage.sub_descriptions:
            descriptions.extend(stage.sub_descriptions)
        else:
            descriptions.extend([stage.description] * stage.step_span)
    return tuple(descriptions)


def execute_run_pipeline(
    stages: Sequence[RunStage],
    progress: CliProgressReporter | None,
) -> dict[str, str]:
    """Execute a stage plan, skipping stages whose outputs already exist.

    Args:
        stages: Ordered stage plan.
        progress: Optional progress reporter.

    Returns:
        Mapping of skipped stage key to the skip reason.
    """
    step_total = sum(stage.step_span for stage in stages)
    skipped: dict[str, str] = {}
    cursor = 0
    for stage in stages:
        step_index = cursor + 1
        cursor += stage.step_span
        reason = stage.satisfied() if stage.satisfied is not None else None
        if reason is not None:
            skipped[stage.key] = reason
            emit_progress(
                progress,
                f"{stage.description} — skipped: {reason}",
                step_index=step_index,
                step_total=step_total,
                reset_total=True,
                completed=1,
                total=1,
            )
            continue
        stage.execute(StageRun(step_index=step_index, step_total=step_total))
    return skipped


__all__ = [
    "RunStage",
    "StageRun",
    "execute_run_pipeline",
    "pipeline_step_descriptions",
]
