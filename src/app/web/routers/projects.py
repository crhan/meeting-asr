"""Project discovery and detail endpoints.

P0 scope is read-only: list projects and show one project's workflow state. Mutating
endpoints (create/update/delete, the intake pipeline) land in later phases. All reads
reuse the existing discovery functions in ``app.core.project_refs`` /
``app.core.project_workflow`` -- no logic is duplicated here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.project_models import ProjectListItem
from app.core.project_refs import list_projects
from app.core.project_workflow import load_project_workflow_summary
from app.project_manager import load_manifest
from app.web.deps import get_settings, require_auth, resolve_web_project_ref
from app.web.schemas import ProjectListResponse, ProjectSummary
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/projects", tags=["projects"], dependencies=[Depends(require_auth)]
)


@router.get("", response_model=ProjectListResponse)
def get_projects(settings: WebSettings = Depends(get_settings)) -> ProjectListResponse:
    """List projects under the configured projects directory, with workflow state."""
    result = list_projects(settings.projects_dir)
    summaries = [
        ProjectSummary.from_item(
            item, load_project_workflow_summary(item.project_dir, item.project_id)
        )
        for item in result.projects
    ]
    return ProjectListResponse(
        projects_dir=str(result.projects_dir), projects=summaries
    )


@router.get("/{project_ref}", response_model=ProjectSummary)
def get_project(
    project_ref: str, settings: WebSettings = Depends(get_settings)
) -> ProjectSummary:
    """Resolve one project by id/title/path and return its summary + workflow state."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    manifest = load_manifest(project_dir)
    item = ProjectListItem(
        project_dir,
        manifest.project_id,
        manifest.title,
        manifest.source.meeting_time,
        manifest.status,
        manifest.created_at,
        manifest.updated_at,
        tuple(manifest.meeting_keywords),
    )
    workflow = load_project_workflow_summary(project_dir, project_ref)
    return ProjectSummary.from_item(item, workflow)
