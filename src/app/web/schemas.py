"""Pydantic response/request models for the web API.

These mirror the internal frozen dataclasses (``ProjectListItem``,
``ProjectWorkflowSummary``, ...) at the HTTP boundary. Keeping a dedicated schema layer
means internal model changes do not silently reshape the public API, and FastAPI gets
the OpenAPI/validation it expects.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.project_models import ProjectListItem
from app.core.project_workflow import ProjectWorkflowSummary


class WorkflowState(BaseModel):
    """User-facing workflow state derived from on-disk artifacts."""

    state: str
    state_key: str
    next_action: str
    next_command_short: str
    outputs: list[str]
    missing: list[str]

    @classmethod
    def from_summary(cls, summary: ProjectWorkflowSummary) -> "WorkflowState":
        """Build from a :class:`ProjectWorkflowSummary`."""
        return cls(
            state=summary.state,
            state_key=summary.state_key,
            next_action=summary.next_action,
            next_command_short=summary.next_command_short,
            outputs=list(summary.outputs),
            missing=list(summary.missing),
        )


class ProjectSummary(BaseModel):
    """One project row for the projects list."""

    project_id: str
    title: str
    status: str
    meeting_time: str | None
    created_at: str
    updated_at: str
    meeting_keywords: list[str]
    path: str
    workflow: WorkflowState | None = None

    @classmethod
    def from_item(
        cls,
        item: ProjectListItem,
        workflow: ProjectWorkflowSummary | None = None,
    ) -> "ProjectSummary":
        """Build from a :class:`ProjectListItem` and optional workflow summary."""
        return cls(
            project_id=item.project_id,
            title=item.title,
            status=item.status,
            meeting_time=item.meeting_time,
            created_at=item.created_at,
            updated_at=item.updated_at,
            meeting_keywords=list(item.meeting_keywords),
            path=str(item.project_dir),
            workflow=WorkflowState.from_summary(workflow) if workflow else None,
        )


class ProjectListResponse(BaseModel):
    """Projects discovered under the configured projects directory."""

    projects_dir: str
    projects: list[ProjectSummary]


class JobRef(BaseModel):
    """A reference to a submitted background job."""

    job_id: str
    kind: str
    status: str
