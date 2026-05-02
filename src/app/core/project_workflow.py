"""Infer user-facing project workflow state from project artifacts."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.core.project_models import ProjectManifest


@dataclass(frozen=True, slots=True)
class ProjectWorkflowSummary:
    """User-facing workflow state for one project."""

    state_key: str
    state: str
    next_action: str
    next_command: str
    next_command_short: str
    outputs: tuple[str, ...]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ArtifactFlags:
    """Artifact existence flags used to derive workflow state."""

    audio: bool
    sentences: bool
    plain_transcript: bool
    speaker_transcript: bool
    subtitle: bool
    summary: bool
    named_transcript: bool
    named_subtitle: bool
    corrected_transcript: bool
    corrected_subtitle: bool


def load_project_workflow_summary(project_dir: Path, project_ref: str | None = None) -> ProjectWorkflowSummary:
    """
    Load a project manifest and infer the user-facing workflow state.

    Args:
        project_dir: Project root.
        project_ref: Optional command reference to use in next-step commands.

    Returns:
        Inferred project workflow summary.
    """
    manifest_path = project_dir / "project.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ProjectManifest.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        ref = str(project_ref or project_dir)
        return _workflow_summary("broken", ref, (), (f"project.json:{exc}",))
    return project_workflow_summary(project_dir, manifest, project_ref=project_ref)


def project_workflow_summary(
    project_dir: Path,
    manifest: ProjectManifest,
    *,
    project_ref: str | None = None,
) -> ProjectWorkflowSummary:
    """
    Infer a project's current workflow state from files that actually exist.

    Args:
        project_dir: Project root.
        manifest: Loaded project manifest.
        project_ref: Optional command reference to use in next-step commands.

    Returns:
        User-facing workflow summary.
    """
    root = project_dir.expanduser().resolve()
    ref = str(project_ref or manifest.project_id)
    flags = _artifact_flags(root, manifest)
    outputs = _key_outputs(flags)
    missing = _missing_declared_paths(root, manifest)
    if missing:
        return _workflow_summary("broken", ref, outputs, missing)
    if flags.corrected_transcript:
        return _workflow_summary("corrected", ref, outputs)
    if flags.named_transcript:
        return _workflow_summary("completed", ref, outputs)
    if flags.sentences:
        return _workflow_summary("transcribed", ref, outputs)
    if flags.audio:
        return _workflow_summary("prepared", ref, outputs)
    return _workflow_summary("created", ref, outputs)


def project_outputs_text(outputs: Iterable[str]) -> str:
    """
    Render workflow output labels in one compact cell.

    Args:
        outputs: Output labels to display.

    Returns:
        Comma-separated output labels, or ``-`` when none exist.
    """
    values = tuple(outputs)
    return ", ".join(values) if values else "-"


def workflow_payload(summary: ProjectWorkflowSummary) -> dict[str, object]:
    """
    Convert a workflow summary to stable JSON fields.

    Args:
        summary: Workflow summary to serialize.

    Returns:
        JSON-ready workflow payload.
    """
    return {
        "state": summary.state,
        "state_key": summary.state_key,
        "next_action": summary.next_action,
        "next_command": summary.next_command,
        "next_command_short": summary.next_command_short,
        "artifacts": list(summary.outputs),
        "outputs": list(summary.outputs),
        "missing": list(summary.missing),
    }


def _artifact_flags(root: Path, manifest: ProjectManifest) -> _ArtifactFlags:
    """Return existence flags for workflow-relevant artifacts."""
    return _ArtifactFlags(
        audio=_audio_exists(root, manifest),
        sentences=_artifact_exists(root, manifest, (), ("asr/sentences.json",)),
        plain_transcript=_artifact_exists(root, manifest, ("plain_transcript",), ("exports/transcript.txt",)),
        speaker_transcript=_artifact_exists(
            root,
            manifest,
            ("anonymous_transcript",),
            ("exports/transcript_speakers.txt",),
        ),
        subtitle=_artifact_exists(root, manifest, ("subtitle",), ("exports/subtitle.srt",)),
        summary=_artifact_exists(root, manifest, ("meeting_summary",), ("exports/meeting_summary.md",)),
        named_transcript=_artifact_exists(root, manifest, ("named_transcript",), ("exports/transcript_named.txt",)),
        named_subtitle=_artifact_exists(root, manifest, ("named_subtitle",), ("exports/subtitle_named.srt",)),
        corrected_transcript=_artifact_exists(
            root,
            manifest,
            ("corrected_named_transcript", "corrected_transcript"),
            ("exports/transcript_named_corrected.txt", "exports/transcript_corrected.txt"),
        ),
        corrected_subtitle=_artifact_exists(
            root,
            manifest,
            ("corrected_named_subtitle",),
            ("exports/subtitle_named_corrected.srt", "exports/subtitle_corrected.srt"),
        ),
    )


def _key_outputs(flags: _ArtifactFlags) -> tuple[str, ...]:
    """Return compact, user-facing output labels."""
    outputs: list[str] = []
    if flags.summary:
        outputs.append("summary")
    if flags.corrected_transcript or flags.corrected_subtitle:
        _append_present(outputs, flags.corrected_transcript, "corrected-txt")
        _append_present(outputs, flags.corrected_subtitle, "corrected-srt")
        return tuple(outputs)
    if flags.named_transcript or flags.named_subtitle:
        _append_present(outputs, flags.named_transcript, "named-txt")
        _append_present(outputs, flags.named_subtitle, "named-srt")
        return tuple(outputs)
    if flags.sentences:
        _append_present(outputs, flags.plain_transcript, "plain-txt")
        _append_present(outputs, flags.speaker_transcript, "speaker-txt")
        _append_present(outputs, flags.subtitle, "srt")
        return tuple(outputs)
    _append_present(outputs, flags.audio, "audio")
    return tuple(outputs)


def _append_present(outputs: list[str], exists: bool, label: str) -> None:
    """Append one output label when the artifact exists."""
    if exists:
        outputs.append(label)


def _workflow_summary(
    state_key: str,
    project_ref: str,
    outputs: tuple[str, ...],
    missing: tuple[str, ...] = (),
) -> ProjectWorkflowSummary:
    """Build a workflow summary for one state key."""
    labels = {
        "created": ("Created", "Transcribe", ("transcribe", project_ref)),
        "prepared": ("Prepared", "Transcribe", ("transcribe", project_ref)),
        "transcribed": ("Transcribed", "Resolve speakers", ("review", project_ref)),
        "completed": ("Completed", "Correct vocabulary", ("correct", "edit", project_ref)),
        "corrected": (
            "Corrected",
            "View corrected transcript",
            ("transcript", "show", project_ref, "--kind", "corrected"),
        ),
        "broken": ("Broken", "Inspect status", ("status", project_ref)),
    }
    state, action, command_parts = labels[state_key]
    short_command = " ".join(shlex.quote(part) for part in command_parts)
    return ProjectWorkflowSummary(
        state_key=state_key,
        state=state,
        next_action=action,
        next_command=f"meeting-asr project {short_command}",
        next_command_short=short_command,
        outputs=outputs,
        missing=missing,
    )


def _artifact_exists(
    root: Path,
    manifest: ProjectManifest,
    manifest_keys: tuple[str, ...],
    candidates: tuple[str, ...],
) -> bool:
    """Return whether any manifest path or fallback candidate exists."""
    for key in manifest_keys:
        value = manifest.outputs.get(key)
        if isinstance(value, str) and _stored_path(root, value).is_file():
            return True
    return any((root / candidate).is_file() for candidate in candidates)


def _audio_exists(root: Path, manifest: ProjectManifest) -> bool:
    """Return whether project audio exists."""
    audio_path = manifest.audio.get("path")
    if isinstance(audio_path, str) and _stored_path(root, audio_path).is_file():
        return True
    return any((root / "audio" / f"audio.{suffix}").is_file() for suffix in ("flac", "wav", "mp3", "m4a"))


def _missing_declared_paths(root: Path, manifest: ProjectManifest) -> tuple[str, ...]:
    """Return manifest output entries that point at missing files."""
    missing = []
    for key, value in sorted(manifest.outputs.items()):
        if isinstance(value, str) and value and not _stored_path(root, value).exists():
            missing.append(f"{key}:{value}")
    return tuple(missing)


def _stored_path(root: Path, value: str) -> Path:
    """Resolve a stored project path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path
