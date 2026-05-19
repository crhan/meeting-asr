"""Project context and summary helpers for the unified voiceprint review TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.project_manager import load_manifest, project_paths, resolve_project_audio_path


@dataclass(frozen=True, slots=True)
class ProjectVoiceprintContext:
    """Project metadata shown in voiceprint review.

    Attributes:
        source_path: Resolved ASR-aligned audio path.
        title: Human-readable project title.
        status: Current project processing status.
        source_name: Original source media file name.
        meeting_time: Optional meeting time supplied by the user.
    """

    source_path: Path
    title: str
    status: str
    source_name: str
    meeting_time: str | None


def load_project_voiceprint_context(project_dir: Path) -> ProjectVoiceprintContext:
    """Load project metadata for the voiceprint review landing page.

    Args:
        project_dir: Project root.

    Returns:
        Metadata and source media path for voiceprint review.
    """
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    source_path = resolve_project_audio_path(paths.root, manifest)
    return ProjectVoiceprintContext(
        source_path=source_path,
        title=manifest.title,
        status=manifest.status,
        source_name=manifest.source.filename or source_path.name,
        meeting_time=manifest.source.meeting_time,
    )


def render_voiceprint_review_summary(session: Any) -> str:
    """Render a non-interactive summary for the unified voiceprint review session.

    Args:
        session: Unified voiceprint review inputs.

    Returns:
        Plain terminal text.
    """
    lines = ["Voiceprint review"]
    if session.capture is None:
        lines.append("Project candidates: unavailable")
    else:
        _append_project_summary(lines, session.capture)
    _append_library_summary(lines, session.library)
    return "\n".join(lines)


def _append_project_summary(lines: list[str], capture: Any) -> None:
    """Append project candidate information to summary lines."""
    sample_total = sum(len(speaker.clips) for speaker in capture.speakers)
    lines.extend(
        [
            f"Project candidates: {capture.project_title or capture.project_id}",
            f"Project ID: {capture.project_id}",
            f"Status: {capture.project_status or '-'}",
            f"Meeting time: {capture.meeting_time or '-'}",
            f"Source: {capture.source_name or capture.source_path.name}",
            f"Candidate speakers: {len(capture.speakers)} | samples: {sample_total}",
        ]
    )
    for speaker in capture.speakers:
        lines.append(f"  {speaker.name} speaker={speaker.speaker_id} samples={len(speaker.clips)}")


def _append_library_summary(lines: list[str], library: Any) -> None:
    """Append global library information to summary lines."""
    sample_total = sum(speaker.sample_count for speaker in library.speakers)
    embedded_total = sum(speaker.embedded_sample_count for speaker in library.speakers)
    lines.extend(
        [
            f"Global library: {library.db_path}",
            f"Library people: {len(library.speakers)} | samples: {sample_total} | embedded: {embedded_total}/{sample_total}",
        ]
    )
    for speaker in library.speakers:
        lines.append(f"  {speaker.name} id={speaker.public_id} samples={speaker.sample_count}")
