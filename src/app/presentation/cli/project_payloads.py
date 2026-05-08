"""Machine-readable payloads for project CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.project_models import ProjectListItem, ProjectManifest, ProjectPaths
from app.core.project_workflow import load_project_workflow_summary, project_workflow_summary, workflow_payload


def project_list_payload(projects_dir: Path, projects: list[ProjectListItem]) -> dict[str, Any]:
    """
    Build the JSON payload for ``meeting-asr project list``.

    Args:
        projects_dir: Resolved projects parent directory.
        projects: Project rows.

    Returns:
        Stable JSON-ready project list.
    """
    return {
        "projects_dir": projects_dir,
        "count": len(projects),
        "projects": [_project_item_payload(project) for project in projects],
    }


def project_status_payload(paths: ProjectPaths, manifest: ProjectManifest) -> dict[str, Any]:
    """
    Build the JSON payload for ``meeting-asr project status``.

    Args:
        paths: Resolved project paths.
        manifest: Loaded project manifest.

    Returns:
        Stable JSON-ready project status.
    """
    workflow = project_workflow_summary(paths.root, manifest)
    return {
        "project": paths.root,
        "project_id": manifest.project_id,
        "title": manifest.title,
        "title_source": manifest.title_source,
        "title_model": manifest.title_model,
        "meeting_time": manifest.source.meeting_time,
        "status": manifest.status,
        "workflow": workflow_payload(workflow),
        "source": manifest.source.path,
        "original_source": manifest.source.original_path,
        "audio": manifest.audio.get("path"),
        "task_id": manifest.asr.get("task_id"),
        "runtime": manifest.runtime,
        "detected_speakers": manifest.speakers.get("detected_ids", []),
        "outputs": manifest.outputs,
    }


def _project_item_payload(project: ProjectListItem) -> dict[str, Any]:
    """Return a JSON-ready payload for one project list item."""
    workflow = load_project_workflow_summary(project.project_dir, project_ref=project.project_id)
    return {
        "project_id": project.project_id,
        "title": project.title,
        "meeting_time": project.meeting_time,
        "status": project.status,
        "workflow": workflow_payload(workflow),
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "project_dir": project.project_dir,
        "directory": project.project_dir.name,
    }
