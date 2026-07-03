"""Enumerate and resolve a project's exportable artifacts.

Shared by the web artifacts endpoints (list/preview/download) and any CLI callers.
The registry mirrors ``_artifact_flags`` in ``core/project_workflow.py``: each artifact
resolves through its manifest ``outputs`` key first, then well-known ``exports/``
fallbacks. Resolution is a strict whitelist and, for the web, refuses paths that
escape the project root -- a tampered manifest must not turn the download endpoint
into an arbitrary-file server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.project_models import ProjectManifest


@dataclass(frozen=True)
class ProjectArtifact:
    """One downloadable artifact of a project."""

    name: str  # stable identifier, e.g. "transcript_named"
    kind: str  # "transcript" | "subtitle" | "summary"
    corrected: bool
    path: Path
    size_bytes: int
    media_type: str


@dataclass(frozen=True)
class _ArtifactSpec:
    name: str
    kind: str
    corrected: bool
    manifest_keys: tuple[str, ...]
    candidates: tuple[str, ...]
    media_type: str


# Order is the display order: final deliverables first.
_SPECS: tuple[_ArtifactSpec, ...] = (
    _ArtifactSpec(
        "meeting_summary",
        "summary",
        False,
        ("meeting_summary",),
        ("exports/meeting_summary.md",),
        "text/markdown",
    ),
    _ArtifactSpec(
        "transcript_named_corrected",
        "transcript",
        True,
        ("corrected_named_transcript", "corrected_transcript"),
        ("exports/transcript_named_corrected.txt", "exports/transcript_corrected.txt"),
        "text/plain",
    ),
    _ArtifactSpec(
        "subtitle_named_corrected",
        "subtitle",
        True,
        ("corrected_named_subtitle",),
        ("exports/subtitle_named_corrected.srt", "exports/subtitle_corrected.srt"),
        "application/x-subrip",
    ),
    _ArtifactSpec(
        "transcript_named",
        "transcript",
        False,
        ("named_transcript",),
        ("exports/transcript_named.txt",),
        "text/plain",
    ),
    _ArtifactSpec(
        "subtitle_named",
        "subtitle",
        False,
        ("named_subtitle",),
        ("exports/subtitle_named.srt",),
        "application/x-subrip",
    ),
    _ArtifactSpec(
        "transcript_speakers",
        "transcript",
        False,
        ("anonymous_transcript",),
        ("exports/transcript_speakers.txt",),
        "text/plain",
    ),
    _ArtifactSpec(
        "transcript_plain",
        "transcript",
        False,
        ("plain_transcript",),
        ("exports/transcript.txt",),
        "text/plain",
    ),
    _ArtifactSpec(
        "subtitle_plain",
        "subtitle",
        False,
        ("subtitle",),
        ("exports/subtitle.srt",),
        "application/x-subrip",
    ),
)


def list_project_artifacts(project_dir: Path) -> list[ProjectArtifact]:
    """List the project's existing exportable artifacts, deliverables first."""
    root = project_dir.resolve()
    # Parse project.json directly (same as core/project_workflow.py): core must not
    # import app.project_manager, which itself imports app.core.*.
    payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
    manifest = ProjectManifest.from_dict(payload)
    artifacts: list[ProjectArtifact] = []
    for spec in _SPECS:
        path = _resolve_spec(root, manifest, spec)
        if path is not None:
            artifacts.append(
                ProjectArtifact(
                    name=spec.name,
                    kind=spec.kind,
                    corrected=spec.corrected,
                    path=path,
                    size_bytes=path.stat().st_size,
                    media_type=spec.media_type,
                )
            )
    return artifacts


def resolve_project_artifact(project_dir: Path, name: str) -> ProjectArtifact:
    """Resolve one artifact by its whitelist name.

    Raises:
        LookupError: Unknown name or the artifact file does not exist (web maps
            LookupError to 404).
    """
    for artifact in list_project_artifacts(project_dir):
        if artifact.name == name:
            return artifact
    raise LookupError(f"Unknown or missing artifact: {name}")


def _resolve_spec(
    root: Path, manifest: ProjectManifest, spec: _ArtifactSpec
) -> Path | None:
    """Resolve a spec to an existing, project-contained file."""
    for key in spec.manifest_keys:
        value = manifest.outputs.get(key)
        if isinstance(value, str) and value:
            path = _contained_file(root, Path(value).expanduser())
            if path is not None:
                return path
    for candidate in spec.candidates:
        path = _contained_file(root, Path(candidate))
        if path is not None:
            return path
    return None


def _contained_file(root: Path, path: Path) -> Path | None:
    """Return the resolved path when it is an existing file under the project root."""
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved if resolved.is_file() else None
