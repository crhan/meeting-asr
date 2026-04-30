"""Project directory management for meeting transcription workflows."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from app.asr_client import download_transcription_json, submit_transcription, wait_transcription
from app.config import get_default_projects_dir, load_settings
from app.ffmpeg_utils import SUPPORTED_AUDIO_FORMATS, extract_audio_for_asr
from app.models import TranscriptResult
from app.postprocess import (
    merge_adjacent_sentences,
    parse_transcription_result,
    render_plain_text,
    render_speaker_text,
    speaker_id_to_label,
)
from app.speaker_labeling import (
    load_transcript_result,
    render_named_speaker_text,
    render_named_srt,
    write_speaker_mapping,
)
from app.srt_utils import build_srt
from app.uploader import SIGNED_URL_EXPIRES_SECONDS, upload_file_to_oss
from app.utils import ensure_directory, safe_write_json, safe_write_text

SCHEMA_VERSION = 1
PROJECT_DIRS = ("source", "audio", "asr", "speakers", "exports", "logs", "tmp")
PROJECT_GITIGNORE = """source/
audio/
logs/
tmp/
asr/raw_result.json
*.signed-url
"""


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


@dataclass(frozen=True, slots=True)
class ProjectListResult:
    """Projects discovered below one projects parent directory."""

    projects_dir: Path
    projects: list[ProjectListItem]


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


@dataclass(slots=True)
class ProjectTranscribeSummary:
    """Terminal summary for a project transcription."""

    project_dir: Path
    task_id: str
    file_url_source: str
    detected_speaker_count: int
    sentence_count: int


def create_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    hash_source: bool,
) -> ProjectManifest:
    """
    Create a project directory and copy the source media into it.

    Args:
        input_path: Local source media file.
        title: Optional human title.
        projects_dir: Optional parent directory used when project_dir is omitted.
        project_dir: Explicit project directory.
        meeting_time: Optional meeting start time string.
        hash_source: Whether to compute SHA-256.

    Returns:
        Created manifest.
    """
    source = input_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    resolved_title = title or source.stem
    created_at = _now_iso()
    root = _resolve_project_root(source, resolved_title, projects_dir, project_dir, created_at)
    if (root / "project.json").exists():
        raise FileExistsError(f"Project already exists: {root}")
    _create_project_dirs(root)
    project_source = _copy_source_into_project(source, root)
    manifest = _initial_manifest(project_source, source, root, resolved_title, created_at, meeting_time, hash_source)
    safe_write_text(root / "source" / "original.path", str(source) + "\n")
    safe_write_text(root / "notes.md", f"# {resolved_title}\n")
    save_manifest(root, manifest)
    return manifest


def load_manifest(project_dir: Path) -> ProjectManifest:
    """
    Load a project manifest from disk.

    Args:
        project_dir: Project root.

    Returns:
        Parsed manifest.
    """
    payload = json.loads(project_paths(project_dir).manifest.read_text(encoding="utf-8"))
    return ProjectManifest.from_dict(payload)


def save_manifest(project_dir: Path, manifest: ProjectManifest) -> Path:
    """
    Persist a project manifest.

    Args:
        project_dir: Project root.
        manifest: Manifest to write.

    Returns:
        Written manifest path.
    """
    manifest.updated_at = _now_iso()
    return safe_write_json(project_paths(project_dir).manifest, manifest.to_dict())


def project_paths(project_dir: Path) -> ProjectPaths:
    """
    Resolve project paths.

    Args:
        project_dir: Project root.

    Returns:
        Project paths.
    """
    return ProjectPaths(root=project_dir.expanduser().resolve())


def list_projects(projects_dir: Path | None) -> ProjectListResult:
    """
    List project manifests below a projects parent directory.

    Args:
        projects_dir: Optional projects parent directory. Defaults to XDG data.

    Returns:
        Resolved projects parent directory and project summaries sorted newest first.
    """
    root = _projects_parent_dir(projects_dir)
    if not root.exists():
        return ProjectListResult(root, [])
    if not root.is_dir():
        raise NotADirectoryError(f"Projects directory is not a directory: {root}")

    projects: list[ProjectListItem] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or not (candidate / "project.json").is_file():
            continue
        manifest = load_manifest(candidate)
        projects.append(
            ProjectListItem(
                project_dir=candidate.resolve(),
                project_id=manifest.project_id,
                title=manifest.title,
                status=manifest.status,
                created_at=manifest.created_at,
                updated_at=manifest.updated_at,
            )
        )
    projects.sort(key=lambda project: (project.created_at, project.project_id), reverse=True)
    return ProjectListResult(root, projects)


def ensure_project_dirs(project_dir: Path) -> ProjectPaths:
    """
    Ensure standard project directories exist.

    Args:
        project_dir: Project root.

    Returns:
        Project paths.
    """
    paths = project_paths(project_dir)
    _create_project_dirs(paths.root)
    return paths


def prepare_project_audio(project_dir: Path, *, audio_format: str) -> Path:
    """
    Extract mono 16kHz audio into the project audio directory.

    Args:
        project_dir: Project root.
        audio_format: wav or flac.

    Returns:
        Generated audio path.
    """
    normalized_format = _normalize_audio_format(audio_format)
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    audio_path = paths.audio_dir / f"audio.{normalized_format}"
    extract_audio_for_asr(resolve_project_source_path(paths.root, manifest), audio_path, audio_format=normalized_format)
    manifest.audio = _audio_metadata(audio_path, normalized_format)
    manifest.status = "prepared"
    save_manifest(paths.root, manifest)
    return audio_path


def transcribe_project(project_dir: Path, options: ProjectTranscribeOptions) -> ProjectTranscribeSummary:
    """
    Run DashScope transcription for a project.

    Args:
        project_dir: Project root.
        options: Transcription options.

    Returns:
        Transcription summary.
    """
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    audio_path = _ensure_project_audio(paths, manifest, options.audio_format)
    should_upload = _parse_project_oss_upload(options.oss_upload, options.file_url)
    settings = load_settings(require_oss=should_upload)
    file_url, file_url_source = _resolve_project_file_url(paths, manifest, audio_path, options, should_upload, settings)
    task_response = _submit_project_task(settings, file_url, options)
    task_id = _extract_task_id(task_response)
    raw_result = download_transcription_json(wait_transcription(settings=settings, task=task_response))
    parsed_result = parse_transcription_result(raw_result)
    _write_project_asr_outputs(paths, raw_result, parsed_result, options.generate_srt)
    _record_asr_metadata(manifest, task_id, file_url_source, options, parsed_result)
    manifest.status = "transcribed"
    save_manifest(paths.root, manifest)
    return ProjectTranscribeSummary(
        paths.root,
        task_id,
        file_url_source,
        len(parsed_result.detected_speakers),
        len(parsed_result.sentences),
    )


def apply_project_speakers(project_dir: Path, mappings: dict[int, str]) -> tuple[Path, Path, Path]:
    """
    Apply speaker names to project outputs.

    Args:
        project_dir: Project root.
        mappings: Explicit speaker mappings.

    Returns:
        Paths for map, transcript, and SRT.
    """
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    resolved_mapping = _merge_speaker_mapping(result, mappings)
    mapping_path = write_speaker_mapping(paths.speakers_dir / "speaker_map.json", resolved_mapping)
    transcript_path = safe_write_text(
        paths.exports_dir / "transcript_named.txt",
        render_named_speaker_text(result, resolved_mapping),
    )
    srt_path = safe_write_text(paths.exports_dir / "subtitle_named.srt", render_named_srt(result, resolved_mapping))
    manifest.speakers["mapped"] = {str(key): value for key, value in sorted(resolved_mapping.items())}
    manifest.outputs["named_transcript"] = _relative_path(paths.root, transcript_path)
    manifest.outputs["named_subtitle"] = _relative_path(paths.root, srt_path)
    manifest.status = "named"
    save_manifest(paths.root, manifest)
    return mapping_path, transcript_path, srt_path


def init_project_git(project_dir: Path) -> Path:
    """
    Initialize optional Git tracking for edited project outputs.

    Args:
        project_dir: Project root.

    Returns:
        Written .gitignore path.
    """
    paths = ensure_project_dirs(project_dir)
    gitignore_path = safe_write_text(paths.root / ".gitignore", PROJECT_GITIGNORE)
    subprocess.run(["git", "init"], cwd=paths.root, check=True, capture_output=True, text=True)
    return gitignore_path


def resolve_project_source_path(project_root: Path, manifest: ProjectManifest) -> Path:
    """
    Resolve the project source media path.

    Args:
        project_root: Project root.
        manifest: Loaded manifest.

    Returns:
        Absolute source path.
    """
    source_path = Path(manifest.source.path)
    if source_path.is_absolute():
        return source_path
    return (project_root / source_path).resolve()


def parse_mapping_items(mappings: list[str], known_speakers: set[int]) -> dict[int, str]:
    """
    Parse ``speaker_id=name`` mapping CLI values.

    Args:
        mappings: Mapping strings.
        known_speakers: Speaker IDs present in the transcript.

    Returns:
        Parsed mapping.
    """
    resolved: dict[int, str] = {}
    for item in mappings:
        speaker_id, name = _parse_mapping_item(item)
        if speaker_id not in known_speakers:
            raise typer.BadParameter(f"speaker_id={speaker_id} is not present in the transcript.")
        resolved[speaker_id] = name
    return resolved


def _initial_manifest(
    source: Path,
    original_source: Path,
    project_root: Path,
    title: str,
    created_at: str,
    meeting_time: str | None,
    hash_source: bool,
) -> ProjectManifest:
    """Build the initial project manifest."""
    stat = source.stat()
    return ProjectManifest(
        schema_version=SCHEMA_VERSION,
        project_id=_build_project_id(title, created_at),
        title=title,
        created_at=created_at,
        updated_at=created_at,
        status="created",
        source=ProjectSource(
            path=_relative_path(project_root, source),
            filename=source.name,
            size_bytes=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            sha256=_sha256_file(source) if hash_source else None,
            meeting_time=meeting_time,
            original_path=str(original_source),
        ),
        speakers={"detected_ids": [], "mapped": {}},
    )


def _copy_source_into_project(source: Path, root: Path) -> Path:
    """Copy source media into the project directory."""
    target = root / "source" / source.name
    if source.resolve() == target.resolve():
        return target
    if target.exists():
        raise FileExistsError(f"Project source file already exists: {target}")
    shutil.copy2(source, target)
    return target


def _resolve_project_root(
    source: Path,
    title: str,
    projects_dir: Path | None,
    project_dir: Path | None,
    created_at: str,
) -> Path:
    """Resolve the directory for a new project."""
    if project_dir is not None:
        return project_dir.expanduser().resolve()
    base_dir = _projects_parent_dir(projects_dir)
    return (base_dir / f"{created_at[:10].replace('-', '')}_{_slugify(title or source.stem)}").resolve()


def _projects_parent_dir(projects_dir: Path | None) -> Path:
    """
    Resolve the parent directory used for project creation and listing.

    Args:
        projects_dir: Optional projects parent directory.

    Returns:
        Absolute projects parent directory.
    """
    if projects_dir is not None:
        return projects_dir.expanduser().resolve()
    return get_default_projects_dir().resolve()


def _create_project_dirs(root: Path) -> None:
    """Create the project root and standard child directories."""
    ensure_directory(root)
    for name in PROJECT_DIRS:
        ensure_directory(root / name)


def _build_project_id(title: str, created_at: str) -> str:
    """Build a stable human-readable project id."""
    return f"{created_at[:10].replace('-', '')}-{_slugify(title)}"


def _slugify(value: str) -> str:
    """Convert a title into a filesystem-safe slug while preserving CJK text."""
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    return re.sub(r"-+", "-", cleaned).strip("-") or "project"


def _now_iso() -> str:
    """Return the current local timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_audio_format(audio_format: str) -> str:
    """Validate a project audio format."""
    normalized = audio_format.strip().lower()
    if normalized in SUPPORTED_AUDIO_FORMATS:
        return normalized
    supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
    raise typer.BadParameter(f"Unsupported audio format: {audio_format}. Supported formats: {supported}.")


def _audio_metadata(audio_path: Path, audio_format: str) -> dict[str, Any]:
    """Build project audio metadata."""
    return {
        "path": _relative_path(audio_path.parents[1], audio_path),
        "format": audio_format,
        "sample_rate": 16000,
        "channels": 1,
        "size_bytes": audio_path.stat().st_size,
    }


def _ensure_project_audio(paths: ProjectPaths, manifest: ProjectManifest, audio_format: str) -> Path:
    """Return existing project audio or generate it."""
    normalized_format = _normalize_audio_format(audio_format)
    existing_path = _manifest_audio_path(paths.root, manifest, normalized_format)
    if existing_path and existing_path.exists():
        return existing_path
    return prepare_project_audio(paths.root, audio_format=normalized_format)


def _manifest_audio_path(root: Path, manifest: ProjectManifest, audio_format: str) -> Path | None:
    """Resolve audio path from manifest metadata."""
    audio_path = manifest.audio.get("path")
    if audio_path and manifest.audio.get("format") == audio_format:
        return (root / str(audio_path)).resolve()
    return None


def _parse_project_oss_upload(value: str | bool, file_url: str | None) -> bool:
    """Resolve OSS upload mode."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return file_url is None
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise typer.BadParameter("--oss-upload must be auto, true, or false.")


def _resolve_project_file_url(paths, manifest, audio_path, options, should_upload, settings) -> tuple[str, str]:
    """Resolve the HTTP URL submitted to DashScope without storing signed URLs."""
    if not should_upload:
        if not options.file_url:
            raise typer.BadParameter("--file-url is required when --oss-upload is false.")
        manifest.oss = {"mode": "provided_url"}
        return options.file_url, "provided_url"
    object_key = f"meeting-asr/projects/{manifest.project_id}/{audio_path.name}"
    file_url = upload_file_to_oss(audio_path, object_name=object_key, settings=settings)
    expires_at = datetime.now(UTC) + timedelta(seconds=SIGNED_URL_EXPIRES_SECONDS)
    manifest.oss = _oss_metadata(settings.oss_bucket_name, object_key, expires_at)
    save_manifest(paths.root, manifest)
    return file_url, "oss_signed_url"


def _oss_metadata(bucket_name: str | None, object_key: str, expires_at: datetime) -> dict[str, str | None]:
    """Build non-secret OSS metadata."""
    return {
        "mode": "oss_signed_url",
        "bucket": bucket_name,
        "object_key": object_key,
        "signed_url_expires_at": expires_at.isoformat(timespec="seconds"),
    }


def _submit_project_task(settings, file_url: str, options: ProjectTranscribeOptions):
    """Submit the DashScope project transcription task."""
    return submit_transcription(
        settings=settings,
        file_url=file_url,
        model=options.model,
        language_hints=_parse_languages(options.language),
        speaker_count=options.speaker_count,
        timestamp_alignment_enabled=options.timestamp_alignment,
        disfluency_removal_enabled=options.disfluency_removal,
    )


def _extract_task_id(task_response) -> str:
    """Extract task ID from a DashScope submission response."""
    output = getattr(task_response, "output", None)
    task_id = getattr(output, "task_id", None)
    if task_id is None and isinstance(output, dict):
        task_id = output.get("task_id")
    if not task_id:
        raise RuntimeError("DashScope task submission succeeded but task_id is missing.")
    return str(task_id)


def _write_project_asr_outputs(
    paths: ProjectPaths,
    raw_result: dict,
    parsed_result: TranscriptResult,
    generate_srt: bool,
) -> None:
    """Write structured project ASR and export outputs."""
    safe_write_json(paths.asr_dir / "raw_result.json", raw_result)
    safe_write_json(paths.asr_dir / "sentences.json", _sentences_payload(parsed_result))
    safe_write_text(paths.exports_dir / "transcript.txt", render_plain_text(parsed_result))
    safe_write_text(paths.exports_dir / "transcript_speakers.txt", _render_merged_speaker_text(parsed_result))
    if generate_srt:
        safe_write_text(paths.exports_dir / "subtitle.srt", build_srt(parsed_result.sentences))


def _record_asr_metadata(manifest, task_id, file_url_source, options, parsed_result) -> None:
    """Record non-secret ASR metadata into the manifest."""
    manifest.asr = {
        "provider": "dashscope",
        "model": options.model,
        "task_id": task_id,
        "language": _parse_languages(options.language),
        "speaker_count_hint": options.speaker_count,
        "timestamp_alignment": options.timestamp_alignment,
        "disfluency_removal": options.disfluency_removal,
        "file_url_source": file_url_source,
    }
    manifest.outputs.update(_default_output_paths())
    manifest.speakers["detected_ids"] = parsed_result.detected_speakers


def _default_output_paths() -> dict[str, str]:
    """Return standard project output paths."""
    return {
        "raw_result": "asr/raw_result.json",
        "sentences": "asr/sentences.json",
        "anonymous_transcript": "exports/transcript_speakers.txt",
        "plain_transcript": "exports/transcript.txt",
        "subtitle": "exports/subtitle.srt",
    }


def _sentences_payload(parsed_result: TranscriptResult) -> dict[str, Any]:
    """Build normalized sentence JSON payload."""
    return {
        "full_text": parsed_result.full_text,
        "detected_speakers": parsed_result.detected_speakers,
        "sentences": [sentence.to_dict() for sentence in parsed_result.sentences],
    }


def _render_merged_speaker_text(parsed_result: TranscriptResult) -> str:
    """Render merged anonymous speaker transcript text."""
    merged_result = TranscriptResult(
        parsed_result.full_text,
        merge_adjacent_sentences(parsed_result.sentences),
        parsed_result.detected_speakers,
    )
    return render_speaker_text(merged_result)


def _merge_speaker_mapping(result: TranscriptResult, mappings: dict[int, str]) -> dict[int, str]:
    """Merge user mappings with anonymous fallback names."""
    resolved = {speaker_id: speaker_id_to_label(speaker_id) for speaker_id in result.detected_speakers}
    resolved.update(mappings)
    return resolved


def _parse_mapping_item(item: str) -> tuple[int, str]:
    """Parse one speaker mapping value."""
    if "=" not in item:
        raise typer.BadParameter(f"Invalid --map value: {item}. Expected speaker_id=name.")
    raw_key, raw_value = item.split("=", 1)
    if not raw_key.strip() or not raw_value.strip():
        raise typer.BadParameter(f"Invalid --map value: {item}. Expected speaker_id=name.")
    try:
        return int(raw_key.strip()), raw_value.strip()
    except ValueError as exc:
        raise typer.BadParameter(f"Speaker id must be an integer in --map: {item}") from exc


def _parse_languages(language: str | None) -> list[str]:
    """Parse comma-separated language hints."""
    if not language:
        return []
    return [item.strip() for item in language.split(",") if item.strip()]


def _relative_path(root: Path, path: Path) -> str:
    """Return a project-relative POSIX path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
