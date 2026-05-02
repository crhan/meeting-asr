"""Project manifest and workflow result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(slots=True)
class ProjectSource:
    """Source media metadata stored in project.json."""

    path: str
    filename: str
    size_bytes: int
    mtime: str
    sha256: str | None = None
    meeting_time: str | None = None
    original_path: str | None = None


@dataclass(slots=True)
class ProjectManifest:
    """Stable project manifest stored in project.json."""

    schema_version: int
    project_id: str
    title: str
    created_at: str
    updated_at: str
    status: str
    source: ProjectSource
    audio: dict[str, Any] = field(default_factory=dict)
    asr: dict[str, Any] = field(default_factory=dict)
    oss: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    speakers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest to a JSON-ready dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectManifest":
        """Build a manifest from JSON data."""
        source = ProjectSource(**payload["source"])
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            project_id=str(payload["project_id"]),
            title=str(payload["title"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            status=str(payload["status"]),
            source=source,
            audio=dict(payload.get("audio", {})),
            asr=dict(payload.get("asr", {})),
            oss=dict(payload.get("oss", {})),
            outputs=dict(payload.get("outputs", {})),
            speakers=dict(payload.get("speakers", {})),
        )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved project paths."""

    root: Path

    @property
    def manifest(self) -> Path:
        """Return the manifest path."""
        return self.root / "project.json"

    @property
    def audio_dir(self) -> Path:
        """Return the audio artifact directory."""
        return self.root / "audio"

    @property
    def asr_dir(self) -> Path:
        """Return the ASR artifact directory."""
        return self.root / "asr"

    @property
    def exports_dir(self) -> Path:
        """Return the export artifact directory."""
        return self.root / "exports"

    @property
    def speakers_dir(self) -> Path:
        """Return the speaker metadata directory."""
        return self.root / "speakers"


@dataclass(frozen=True, slots=True)
class ProjectListItem:
    """One project row shown by ``meeting-asr project list``."""

    project_dir: Path
    project_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    number: int = 0


@dataclass(frozen=True, slots=True)
class ProjectListResult:
    """Projects discovered below one projects parent directory."""

    projects_dir: Path
    projects: list[ProjectListItem]


@dataclass(frozen=True, slots=True)
class ProjectCreateSummary:
    """Result of creating or reusing a project."""

    project_dir: Path
    manifest: ProjectManifest
    created: bool


@dataclass(slots=True)
class ProjectTranscribeOptions:
    """Options for transcribing a project."""

    speaker_count: int | None
    language: str | None
    model: str
    oss_upload: str | bool
    file_url: str | None
    generate_srt: bool
    timestamp_alignment: bool
    disfluency_removal: bool
    audio_format: str
    asr_hotwords: str | None = "auto"


@dataclass(slots=True)
class ProjectTranscribeSummary:
    """Terminal summary for a project transcription."""

    project_dir: Path
    task_id: str
    file_url_source: str
    detected_speaker_count: int
    sentence_count: int


@dataclass(frozen=True, slots=True)
class ProjectMeetingSummary:
    """Terminal summary for generated meeting summary artifacts."""

    project_dir: Path
    title: str
    summary_path: Path
    json_path: Path
    model: str
    title_updated: bool


@dataclass(frozen=True, slots=True)
class ProjectUpdateSummary:
    """Result of updating project metadata."""

    project_dir: Path
    manifest: ProjectManifest


@dataclass(frozen=True, slots=True)
class ProjectDeleteSummary:
    """Result of deleting or trashing a project."""

    project_dir: Path
    destination: Path | None
    permanent: bool


@dataclass(frozen=True, slots=True)
class TrashedProjectListItem:
    """One project row stored in Meeting-ASR trash."""

    trash_dir: Path
    restore_project_dir: Path
    project_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    trashed_at: str
    number: int = 0


@dataclass(frozen=True, slots=True)
class ProjectTrashListResult:
    """Projects discovered below the Meeting-ASR trash directory."""

    trash_dir: Path
    projects: list[TrashedProjectListItem]


@dataclass(frozen=True, slots=True)
class ProjectRestoreSummary:
    """Result of restoring a project from Meeting-ASR trash."""

    trash_dir: Path
    project_dir: Path
    manifest: ProjectManifest


@dataclass(frozen=True, slots=True)
class ProjectPurgeSummary:
    """Result of permanently deleting a trashed project."""

    trash_dir: Path
    manifest: ProjectManifest


@dataclass(frozen=True, slots=True)
class ProjectTrashCleanupSummary:
    """Result of cleaning old trashed projects."""

    trash_dir: Path
    removed: list[ProjectPurgeSummary]
