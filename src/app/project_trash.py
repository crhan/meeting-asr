"""Project trash lifecycle helpers."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import get_data_dir, get_default_projects_dir
from app.core.project_models import (
    ProjectManifest,
    ProjectPurgeSummary,
    ProjectRestoreSummary,
    ProjectTrashCleanupSummary,
    ProjectTrashListResult,
    TrashedProjectListItem,
)
from app.utils import ensure_directory, safe_write_json

TRASH_METADATA_FILENAME = "trash.json"
TRASH_SCHEMA_VERSION = 1
TRASH_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def move_project_to_trash(project_dir: Path) -> Path:
    """
    Move a project directory into Meeting-ASR trash.

    Args:
        project_dir: Existing project directory.

    Returns:
        Destination trash directory.
    """
    source = project_dir.expanduser().resolve()
    trashed_at = _now_utc()
    destination = _unique_trash_project_path(source, trashed_at)
    ensure_directory(destination.parent)
    shutil.move(str(source), str(destination))
    _write_trash_metadata(destination, source, trashed_at)
    return destination


def list_trashed_projects() -> ProjectTrashListResult:
    """
    List projects currently stored in Meeting-ASR trash.

    Returns:
        Trash project list.
    """
    trash_dir = get_project_trash_dir()
    if not trash_dir.exists():
        return ProjectTrashListResult(trash_dir, [])
    if not trash_dir.is_dir():
        raise NotADirectoryError(f"Project trash path is not a directory: {trash_dir}")
    projects = [
        _trashed_project_item(child)
        for child in trash_dir.iterdir()
        if _is_project_dir(child)
    ]
    projects = [project for project in projects if project is not None]
    projects.sort(
        key=lambda project: (
            project.trashed_at,
            project.updated_at,
            project.project_id,
        ),
        reverse=True,
    )
    return ProjectTrashListResult(trash_dir, projects)


def restore_trashed_project(
    trash_ref: str | Path,
    *,
    projects_dir: Path | None,
    project_dir: Path | None,
) -> ProjectRestoreSummary:
    """
    Restore one trashed project.

    Args:
        trash_ref: Trash path, project id, title, or trash directory name.
        projects_dir: Optional destination projects parent.
        project_dir: Optional exact destination directory.

    Returns:
        Restore summary.
    """
    item = resolve_trashed_project_ref(trash_ref)
    manifest = _load_manifest(item.trash_dir)
    destination = _restore_destination(item, projects_dir, project_dir)
    if destination.exists():
        raise FileExistsError(
            f"Restore destination already exists: {destination}. "
            "Pass --project-dir to restore to a different directory."
        )
    ensure_directory(destination.parent)
    shutil.move(str(item.trash_dir), str(destination))
    return ProjectRestoreSummary(item.trash_dir, destination, manifest)


def purge_trashed_project(trash_ref: str | Path) -> ProjectPurgeSummary:
    """
    Permanently delete one trashed project.

    Args:
        trash_ref: Trash path, project id, title, or trash directory name.

    Returns:
        Purge summary.
    """
    item = resolve_trashed_project_ref(trash_ref)
    manifest = _load_manifest(item.trash_dir)
    shutil.rmtree(item.trash_dir)
    return ProjectPurgeSummary(item.trash_dir, manifest)


def cleanup_project_trash(*, older_than_days: int) -> ProjectTrashCleanupSummary:
    """
    Permanently delete trashed projects older than the configured age.

    Args:
        older_than_days: Minimum trash age in days. Zero means all trashed projects.

    Returns:
        Cleanup summary.
    """
    if older_than_days < 0:
        raise ValueError("--older-than-days must be greater than or equal to 0.")
    result = list_trashed_projects()
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    removed = []
    for item in result.projects:
        if _parse_trashed_at(item.trashed_at) <= cutoff:
            manifest = _load_manifest(item.trash_dir)
            shutil.rmtree(item.trash_dir)
            removed.append(ProjectPurgeSummary(item.trash_dir, manifest))
    return ProjectTrashCleanupSummary(result.trash_dir, removed)


def resolve_trashed_project_ref(trash_ref: str | Path) -> TrashedProjectListItem:
    """
    Resolve a trash reference into one trashed project.

    Args:
        trash_ref: Trash path, project id, title, or trash directory name.

    Returns:
        Matching trashed project.
    """
    ref_text = str(trash_ref).strip()
    if not ref_text:
        raise ValueError("Trash project reference must not be empty.")
    ref_path = Path(ref_text).expanduser()
    if _looks_like_path(ref_text, ref_path):
        return _trashed_project_path_match(ref_path)
    projects = list_trashed_projects().projects
    exact = [
        project
        for project in projects
        if _matches_trash_ref(project, ref_text, partial=False)
    ]
    if exact:
        return _single_trash_match(ref_text, exact)
    partial = [
        project
        for project in projects
        if _matches_trash_ref(project, ref_text, partial=True)
    ]
    if partial:
        return _single_trash_match(ref_text, partial)
    raise FileNotFoundError(
        f"Trashed project not found: {ref_text}. Run `meeting-asr project trash list`."
    )


def get_project_trash_dir() -> Path:
    """
    Return the XDG trash directory for projects.

    Returns:
        Global project trash directory.
    """
    return get_data_dir() / "trash" / "projects"


def _write_trash_metadata(
    trash_dir: Path, original_project_dir: Path, trashed_at: datetime
) -> None:
    """Persist restore metadata next to the trashed project."""
    payload = {
        "schema_version": TRASH_SCHEMA_VERSION,
        "original_project_dir": str(original_project_dir),
        "original_project_name": original_project_dir.name,
        "trashed_at": trashed_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    safe_write_json(trash_dir / TRASH_METADATA_FILENAME, payload)


def _unique_trash_project_path(project_dir: Path, trashed_at: datetime) -> Path:
    """Return a non-existing trash path for a project directory."""
    trash_dir = get_project_trash_dir()
    stamp = trashed_at.strftime(TRASH_STAMP_FORMAT)
    base = trash_dir / f"{stamp}_{project_dir.name}"
    candidate = base
    index = 2
    while candidate.exists():
        candidate = trash_dir / f"{base.name}_{index}"
        index += 1
    return candidate


def _trashed_project_item(trash_dir: Path) -> TrashedProjectListItem | None:
    """Build one trash list item, ignoring invalid trash entries."""
    try:
        manifest = _load_manifest(trash_dir)
        metadata = _load_trash_metadata(trash_dir)
        trashed_at = _metadata_trashed_at(trash_dir, metadata)
        restore_dir = _metadata_restore_dir(trash_dir, metadata)
        return TrashedProjectListItem(
            trash_dir.resolve(),
            restore_dir,
            manifest.project_id,
            manifest.title,
            manifest.status,
            manifest.created_at,
            manifest.updated_at,
            trashed_at,
        )
    except Exception:  # noqa: BLE001
        return None


def _load_manifest(project_dir: Path) -> ProjectManifest:
    """Load the manifest from one project directory."""
    payload = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Project manifest must contain a JSON object: {project_dir / 'project.json'}"
        )
    return ProjectManifest.from_dict(payload)


def _load_trash_metadata(trash_dir: Path) -> dict[str, Any]:
    """Load trash metadata, returning an empty dict for legacy trash entries."""
    metadata_path = trash_dir / TRASH_METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _metadata_trashed_at(trash_dir: Path, metadata: dict[str, Any]) -> str:
    """Return the trash timestamp as an ISO string."""
    raw_value = metadata.get("trashed_at")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    parsed = _trash_stamp_from_name(trash_dir.name)
    if parsed is not None:
        return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    return (
        datetime.fromtimestamp(trash_dir.stat().st_mtime, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _metadata_restore_dir(trash_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the default restore destination."""
    raw_value = metadata.get("original_project_dir")
    if isinstance(raw_value, str) and raw_value.strip():
        return Path(raw_value).expanduser().resolve()
    return (
        get_default_projects_dir().resolve() / _original_project_name(trash_dir.name)
    ).resolve()


def _restore_destination(
    item: TrashedProjectListItem,
    projects_dir: Path | None,
    project_dir: Path | None,
) -> Path:
    """Resolve the destination for restoring a trashed project."""
    if project_dir is not None:
        return project_dir.expanduser().resolve()
    if projects_dir is not None:
        return (
            projects_dir.expanduser().resolve() / item.restore_project_dir.name
        ).resolve()
    return item.restore_project_dir


def _is_project_dir(path: Path) -> bool:
    """Return whether a path looks like a project directory."""
    return path.is_dir() and (path / "project.json").is_file()


def _looks_like_path(ref_text: str, path: Path) -> bool:
    """Return whether a reference should be treated as a filesystem path."""
    return (
        path.exists()
        or path.is_absolute()
        or "/" in ref_text
        or ref_text in {".", ".."}
    )


def _trashed_project_path_match(path: Path) -> TrashedProjectListItem:
    """Resolve a trash project by path."""
    resolved = path.resolve()
    if _is_project_dir(resolved):
        item = _trashed_project_item(resolved)
        if item is not None:
            return item
    raise FileNotFoundError(
        f"Trashed project manifest does not exist: {resolved / 'project.json'}"
    )


def _matches_trash_ref(
    project: TrashedProjectListItem, ref_text: str, *, partial: bool
) -> bool:
    """Return whether a trashed project matches a text reference."""
    targets = (
        project.project_id,
        project.title,
        project.trash_dir.name,
        project.restore_project_dir.name,
    )
    normalized_ref = ref_text.casefold()
    if partial:
        return any(normalized_ref in target.casefold() for target in targets)
    return any(normalized_ref == target.casefold() for target in targets)


def _single_trash_match(
    ref_text: str, projects: list[TrashedProjectListItem]
) -> TrashedProjectListItem:
    """Resolve one non-path trash reference."""
    if len(projects) == 1:
        return projects[0]
    choices = ", ".join(
        f"{project.project_id} ({project.title})" for project in projects[:5]
    )
    raise ValueError(
        f"Trash project reference is ambiguous: {ref_text}. Matches: {choices}"
    )


def _original_project_name(trash_name: str) -> str:
    """Return the original project directory name encoded in a trash directory name."""
    if len(trash_name) > 17 and trash_name[16] == "_":
        return trash_name[17:]
    return trash_name


def _trash_stamp_from_name(trash_name: str) -> datetime | None:
    """Parse the timestamp prefix from a trash directory name."""
    if len(trash_name) <= 16 or trash_name[16] != "_":
        return None
    try:
        return datetime.strptime(trash_name[:16], TRASH_STAMP_FORMAT).replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def _parse_trashed_at(value: str) -> datetime:
    """Parse a trash timestamp."""
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
