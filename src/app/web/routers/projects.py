"""Project discovery and detail endpoints.

P0 scope is read-only: list projects and show one project's workflow state. Mutating
endpoints (create/update/delete, the intake pipeline) land in later phases. All reads
reuse the existing discovery functions in ``app.core.project_refs`` /
``app.core.project_workflow`` -- no logic is duplicated here.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.core.project_artifacts import (
    list_project_artifacts,
    resolve_project_artifact,
)
from app.core.project_models import ProjectListItem
from app.core.project_refs import list_projects
from app.core.project_workflow import load_project_workflow_summary
from app.project_manager import load_manifest, update_project_metadata
from app.speaker_match_status import project_has_unresolved_match
from app.web.deps import (
    get_locks,
    get_settings,
    require_auth,
    resolve_web_project_ref,
)
from app.web.locks import LockRegistry, project_lock_key
from app.web.schemas import (
    ArtifactListOut,
    ArtifactOut,
    ProjectListResponse,
    ProjectSummary,
    ProjectUpdateIn,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/projects", tags=["projects"], dependencies=[Depends(require_auth)]
)


@router.get("", response_model=ProjectListResponse)
def get_projects(settings: WebSettings = Depends(get_settings)) -> ProjectListResponse:
    """List projects under the configured projects directory, with workflow state."""
    result = list_projects(settings.projects_dir, restrict_to_projects_dir=True)
    summaries = [
        ProjectSummary.from_item(
            item,
            load_project_workflow_summary(item.project_dir, item.project_id),
            has_unresolved_matches=project_has_unresolved_match(item.project_dir),
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
    return ProjectSummary.from_item(
        item,
        workflow,
        has_unresolved_matches=project_has_unresolved_match(project_dir),
    )


@router.patch("/{project_ref}", response_model=ProjectSummary)
async def patch_project(
    project_ref: str,
    payload: ProjectUpdateIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> ProjectSummary:
    """Edit a project's title / meeting time (reuses the CLI's update semantics).

    Runs under the per-project lock so a concurrent run/save cannot interleave with
    the manifest rewrite.
    """
    project_dir = resolve_web_project_ref(project_ref, settings)

    def work() -> ProjectSummary:
        update_project_metadata(
            project_dir, title=payload.title, meeting_time=payload.meeting_time
        )
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
        return ProjectSummary.from_item(
            item,
            workflow,
            has_unresolved_matches=project_has_unresolved_match(project_dir),
        )

    loop = asyncio.get_running_loop()
    async with locks.acquire(project_lock_key(str(project_dir))):
        return await loop.run_in_executor(None, work)


@router.get("/{project_ref}/artifacts", response_model=ArtifactListOut)
def get_artifacts(
    project_ref: str, settings: WebSettings = Depends(get_settings)
) -> ArtifactListOut:
    """List the project's existing exportable artifacts (final deliverables)."""
    project_dir = resolve_web_project_ref(project_ref, settings)
    manifest = load_manifest(project_dir)
    return ArtifactListOut(
        project_id=manifest.project_id,
        artifacts=[
            ArtifactOut(
                name=artifact.name,
                kind=artifact.kind,
                corrected=artifact.corrected,
                file_name=artifact.path.name,
                size_bytes=artifact.size_bytes,
                media_type=artifact.media_type,
            )
            for artifact in list_project_artifacts(project_dir)
        ],
    )


@router.get("/{project_ref}/artifacts/{name}")
def download_artifact(
    project_ref: str,
    name: str,
    download: bool = Query(default=False),
    settings: WebSettings = Depends(get_settings),
) -> FileResponse:
    """Serve one artifact by whitelist name -- inline for preview, attachment for download.

    ``resolve_project_artifact`` only serves registered, project-contained files
    (LookupError -> 404), so this cannot be steered to arbitrary paths.
    """
    project_dir = resolve_web_project_ref(project_ref, settings)
    artifact = resolve_project_artifact(project_dir, name)
    return FileResponse(
        artifact.path,
        # Serve text artifacts as UTF-8 text so inline preview renders in-browser;
        # .srt would otherwise default to a download prompt.
        media_type=(
            "text/plain; charset=utf-8"
            if artifact.media_type != "text/markdown"
            else "text/markdown; charset=utf-8"
        ),
        filename=artifact.path.name,
        content_disposition_type="attachment" if download else "inline",
    )
