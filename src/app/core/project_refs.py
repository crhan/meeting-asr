"""Project discovery and reference resolution."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_default_projects_dir
from app.core.project_models import ProjectListItem, ProjectListResult, ProjectManifest


def list_projects(projects_dir: Path | None) -> ProjectListResult:
    """
    List known projects under a parent directory.

    Args:
        projects_dir: Optional projects parent directory.

    Returns:
        Project list result.
    """
    parent = _projects_parent_dir(projects_dir)
    if not parent.exists():
        return ProjectListResult(parent, [])
    if not parent.is_dir():
        raise NotADirectoryError(f"Projects directory is not a directory: {parent}")
    projects: list[ProjectListItem] = []
    for child in parent.iterdir():
        if not child.is_dir() or not (child / "project.json").is_file():
            continue
        manifest = _load_manifest_or_none(child)
        if manifest is None:
            continue
        projects.append(
            ProjectListItem(
                child.resolve(),
                manifest.project_id,
                manifest.title,
                manifest.status,
                manifest.created_at,
                manifest.updated_at,
            )
        )
    projects.sort(key=lambda project: (project.created_at, project.project_id), reverse=True)
    return ProjectListResult(parent, _number_project_list_items(projects))


def resolve_project_ref(project_ref: Path | str, projects_dir: Path | None = None) -> Path:
    """
    Resolve a project path, numeric project number, id, or title.

    Args:
        project_ref: Project reference.
        projects_dir: Optional projects parent directory.

    Returns:
        Resolved project path.
    """
    ref_text = str(project_ref).strip()
    if not ref_text:
        raise ValueError("Project reference must not be empty.")
    ref_path = Path(ref_text).expanduser()
    if _looks_like_path(ref_text, ref_path):
        return _resolve_project_path(ref_path)
    projects = list_projects(projects_dir).projects
    if _is_project_number_ref(ref_text):
        return _single_project_number_match(ref_text, projects)
    exact = [project for project in projects if _matches_project_ref(project, ref_text, partial=False)]
    if exact:
        return _single_project_match(ref_text, exact)
    partial = [project for project in projects if _matches_project_ref(project, ref_text, partial=True)]
    if partial:
        return _single_project_match(ref_text, partial)
    raise FileNotFoundError(f"Project not found by path, id, or title: {ref_text}")


def find_project_by_source(
    input_path: Path,
    projects_dir: Path | None,
    *,
    source_sha256: str | None = None,
) -> Path | None:
    """
    Find an existing project created from a source file.

    Args:
        input_path: Source media path.
        projects_dir: Optional projects parent directory.
        source_sha256: Optional source content hash.

    Returns:
        Matching project path or None.
    """
    source_path = input_path.expanduser().resolve()
    matches = []
    for project in list_projects(projects_dir).projects:
        manifest = _load_manifest_or_none(project.project_dir)
        if manifest and _source_manifest_matches(source_path, source_sha256, manifest):
            matches.append(project)
    if not matches:
        return None
    return max(matches, key=_project_reuse_rank).project_dir


def _projects_parent_dir(projects_dir: Path | None) -> Path:
    """Resolve the projects parent directory."""
    if projects_dir is not None:
        return projects_dir.expanduser().resolve()
    return get_default_projects_dir().resolve()


def _load_manifest_or_none(project_dir: Path) -> ProjectManifest | None:
    """Load a project manifest, ignoring non-project directories."""
    manifest_path = project_dir / "project.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return ProjectManifest.from_dict(payload)
    except Exception:  # noqa: BLE001
        return None


def _number_project_list_items(projects: list[ProjectListItem]) -> list[ProjectListItem]:
    """Assign stable short numbers for display and CLI references."""
    return [
        ProjectListItem(
            item.project_dir,
            item.project_id,
            item.title,
            item.status,
            item.created_at,
            item.updated_at,
            index,
        )
        for index, item in enumerate(projects, start=1)
    ]


def _looks_like_path(ref_text: str, path: Path) -> bool:
    """Return whether a reference should be treated as a filesystem path."""
    return path.exists() or path.is_absolute() or ref_text in {".", ".."} or "/" in ref_text


def _resolve_project_path(path: Path) -> Path:
    """Resolve and validate a project path."""
    resolved = path.resolve()
    manifest_path = resolved / "project.json"
    if manifest_path.is_file():
        return resolved
    raise FileNotFoundError(f"Project manifest does not exist: {manifest_path}")


def _source_manifest_matches(source: Path, source_sha256: str | None, manifest: ProjectManifest) -> bool:
    """Return whether a manifest belongs to a source file."""
    if _same_original_source_path(source, manifest.source.original_path):
        return True
    return bool(source_sha256 and manifest.source.sha256 and manifest.source.sha256 == source_sha256)


def _same_original_source_path(source: Path, original_path: str | None) -> bool:
    """Return whether two source paths resolve to the same file."""
    if not original_path:
        return False
    return Path(original_path).expanduser().resolve() == source


def _project_reuse_rank(project: ProjectListItem) -> tuple[int, str, str]:
    """Rank reusable project candidates."""
    status_rank = {
        "created": 0,
        "prepared": 1,
        "transcribed": 2,
        "named": 3,
        "corrected": 4,
        "voiceprinted": 5,
    }
    return status_rank.get(project.status, 0), project.updated_at, project.created_at


def _is_project_number_ref(ref_text: str) -> bool:
    """Return whether a reference is a short project number."""
    return ref_text.isdecimal() and int(ref_text) > 0


def _single_project_number_match(ref_text: str, projects: list[ProjectListItem]) -> Path:
    """Resolve a numeric project reference."""
    number = int(ref_text)
    for project in projects:
        if project.number == number:
            return project.project_dir
    raise FileNotFoundError(f"Project number not found: {ref_text}. Run `meeting-asr project list`.")


def _matches_project_ref(project: ProjectListItem, ref_text: str, *, partial: bool) -> bool:
    """Return whether a project matches a text reference."""
    targets = (project.project_id, project.title, project.project_dir.name)
    normalized_ref = ref_text.casefold()
    if partial:
        return any(normalized_ref in target.casefold() for target in targets)
    return any(normalized_ref == target.casefold() for target in targets)


def _single_project_match(ref_text: str, projects: list[ProjectListItem]) -> Path:
    """Resolve a non-path project reference."""
    if len(projects) == 1:
        return projects[0].project_dir
    choices = ", ".join(f"{project.project_id} ({project.title})" for project in projects[:5])
    raise ValueError(f"Project reference is ambiguous: {ref_text}. Matches: {choices}")
