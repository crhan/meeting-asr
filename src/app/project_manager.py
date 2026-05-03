"""Project directory management for meeting transcription workflows."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer

from app.core.asr_wait import (
    asr_wait_description,
    asr_wait_total,
    emit_dashscope_wait_poll,
    estimate_dashscope_wait,
    record_dashscope_wait,
)
from app.config import load_settings
from app.asr_hotwords import AsrHotwordResolution, resolve_asr_hotwords
from app.asr_pricing import AsrCostEstimate, estimate_asr_cost
from app.core.progress import CliProgressReporter, emit_progress
from app.core.project_models import (
    SCHEMA_VERSION,
    ProjectCreateSummary,
    ProjectDeleteSummary,
    ProjectManifest,
    ProjectMeetingSummary,
    ProjectPaths,
    ProjectSource,
    ProjectTranscribeOptions,
    ProjectTranscribeSummary,
    ProjectUpdateSummary,
)
from app.core.project_refs import (
    _projects_parent_dir,
    find_project_by_source,
)
from app.project_trash import move_project_to_trash
from app.core.oss_upload import (
    emit_oss_upload_progress,
    emit_oss_upload_start,
    estimate_oss_upload,
    record_oss_upload,
)
from app.infra.dashscope_asr import download_transcription_json, submit_transcription, wait_transcription
from app.infra.ffmpeg import SUPPORTED_AUDIO_FORMATS, extract_audio_for_asr, probe_media_duration_seconds
from app.meeting_summary import MeetingSummary, generate_meeting_summary, render_meeting_summary_markdown
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

PROJECT_DIRS = ("source", "audio", "asr", "speakers", "exports", "logs", "tmp")
PROJECT_GITIGNORE = """source/
audio/
logs/
tmp/
asr/raw_result.json
*.signed-url
"""
DOWNSTREAM_OUTPUT_KEYS = (
    "meeting_summary",
    "meeting_summary_json",
    "named_transcript",
    "named_subtitle",
    "corrected_sentences",
    "corrected_transcript",
    "corrected_named_transcript",
    "corrected_named_subtitle",
    "asr_hotwords",
    "vocabulary_corrections",
)
DOWNSTREAM_ARTIFACT_PATHS = (
    "exports/meeting_summary.md",
    "exports/meeting_summary.json",
    "exports/transcript_named.txt",
    "exports/subtitle_named.srt",
    "asr/sentences_corrected.json",
    "exports/transcript_corrected.txt",
    "exports/transcript_speakers_corrected.txt",
    "exports/transcript_named_corrected.txt",
    "exports/subtitle_corrected.srt",
    "exports/subtitle_named_corrected.srt",
    "corrections/asr_hotwords.json",
    "corrections/applied.json",
    "speakers/speaker_map.json",
    "speakers/speaker_matches.json",
)
DOWNSTREAM_SPEAKER_KEYS = ("mapped", "matches", "voiceprints")

LOGGER = logging.getLogger(__name__)


def create_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    hash_source: bool,
    source_sha256: str | None = None,
    progress: CliProgressReporter | None = None,
) -> ProjectManifest:
    """
    Create a project directory and copy the source media into it.

    Args:
        input_path: Local source media file.
        title: Optional human title.
        projects_dir: Optional parent directory used when project_dir is omitted.
        project_dir: Explicit project directory.
        meeting_time: Optional meeting start time string.
        hash_source: Deprecated compatibility flag. Source identity is always hashed.
        source_sha256: Optional precomputed source SHA-256.
        progress: Optional progress reporter.

    Returns:
        Created manifest.
    """
    source = input_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    resolved_title = title or source.stem
    created_at = _now_iso()
    resolved_sha256 = source_sha256 or _sha256_file(source, progress)
    root = _resolve_project_root(source, projects_dir, project_dir, resolved_sha256)
    if (root / "project.json").exists():
        raise FileExistsError(f"Project already exists: {root}")
    _create_project_dirs(root)
    project_source = _copy_source_into_project(source, root, progress)
    manifest = _initial_manifest(
        project_source,
        source,
        root,
        resolved_title,
        created_at,
        meeting_time,
        resolved_sha256,
        progress,
    )
    safe_write_text(root / "source" / "original.path", str(source) + "\n")
    safe_write_text(root / "notes.md", f"# {resolved_title}\n")
    save_manifest(root, manifest)
    return manifest

def create_or_reuse_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    hash_source: bool,
    progress: CliProgressReporter | None = None,
) -> ProjectCreateSummary:
    """
    Return the existing project for a source video, or create a new one.

    Args:
        input_path: Local source media file.
        title: Optional human title for new projects.
        projects_dir: Optional parent directory used when project_dir is omitted.
        project_dir: Explicit project directory. Explicit paths always create that path.
        meeting_time: Optional meeting start time string for new projects.
        hash_source: Deprecated compatibility flag. Source identity is always hashed.
        progress: Optional progress reporter.

    Returns:
        Project creation summary.
    """
    if project_dir is None:
        existing = find_project_by_source(input_path, projects_dir)
        if existing is not None:
            emit_progress(progress, "Using existing project", total=1, completed=1)
            return ProjectCreateSummary(existing, load_manifest(existing), False)
    source = input_path.expanduser().resolve()
    source_sha256 = _sha256_file(source, progress)
    if project_dir is None:
        existing = find_project_by_source(input_path, projects_dir, source_sha256=source_sha256)
        if existing is not None:
            emit_progress(progress, "Using existing project", total=1, completed=1)
            return ProjectCreateSummary(existing, load_manifest(existing), False)
    manifest = create_project(
        input_path,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=meeting_time,
        hash_source=hash_source,
        source_sha256=source_sha256,
        progress=progress,
    )
    project_root = _resolve_project_root(source, projects_dir, project_dir, source_sha256)
    return ProjectCreateSummary(project_root, manifest, True)

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

def update_project_metadata(
    project_dir: Path,
    *,
    title: str | None,
    meeting_time: str | None,
) -> ProjectUpdateSummary:
    """
    Update editable project metadata.

    Args:
        project_dir: Project root.
        title: Optional replacement title.
        meeting_time: Optional replacement meeting time.

    Returns:
        Updated project summary.
    """
    if title is None and meeting_time is None:
        raise ValueError("Nothing to update. Pass --title or --meeting-time.")
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    if title is not None:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Project title must not be empty.")
        manifest.title = cleaned_title
    if meeting_time is not None:
        manifest.source.meeting_time = meeting_time.strip() or None
    save_manifest(paths.root, manifest)
    return ProjectUpdateSummary(paths.root, manifest)

def delete_project(project_dir: Path, *, permanent: bool) -> ProjectDeleteSummary:
    """
    Delete a project, moving it to Meeting-ASR trash by default.

    Args:
        project_dir: Project root.
        permanent: Physically remove the project when true.

    Returns:
        Deletion summary.
    """
    paths = project_paths(project_dir)
    load_manifest(paths.root)
    if permanent:
        shutil.rmtree(paths.root)
        return ProjectDeleteSummary(paths.root, None, True)
    destination = move_project_to_trash(paths.root)
    return ProjectDeleteSummary(paths.root, destination, False)

def project_paths(project_dir: Path) -> ProjectPaths:
    """
    Resolve project paths.

    Args:
        project_dir: Project root.

    Returns:
        Project paths.
    """
    return ProjectPaths(root=project_dir.expanduser().resolve())

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

def prepare_project_audio(
    project_dir: Path,
    *,
    audio_format: str,
    progress: CliProgressReporter | None = None,
) -> Path:
    """
    Extract mono 16kHz audio into the project audio directory.

    Args:
        project_dir: Project root.
        audio_format: wav or flac.
        progress: Optional progress reporter.

    Returns:
        Generated audio path.
    """
    normalized_format = _normalize_audio_format(audio_format)
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    audio_path = paths.audio_dir / f"audio.{normalized_format}"
    emit_progress(progress, "Extracting 16 kHz mono audio")
    extract_audio_for_asr(resolve_project_source_path(paths.root, manifest), audio_path, audio_format=normalized_format)
    emit_progress(progress, "Audio ready", advance=1)
    manifest.audio = _audio_metadata(audio_path, normalized_format)
    manifest.status = "prepared"
    save_manifest(paths.root, manifest)
    return audio_path

def transcribe_project(
    project_dir: Path,
    options: ProjectTranscribeOptions,
    progress: CliProgressReporter | None = None,
    *,
    step_offset: int = 0,
    step_total: int | None = None,
) -> ProjectTranscribeSummary:
    """
    Run DashScope transcription for a project.

    Args:
        project_dir: Project root.
        options: Transcription options.
        progress: Optional progress reporter.
        step_offset: Number of workflow steps before transcription.
        step_total: Optional total workflow step count.

    Returns:
        Transcription summary.
    """
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    transcribe_steps = step_total or 6
    emit_progress(
        progress,
        "Preparing audio",
        step_index=step_offset + 1,
        step_total=transcribe_steps,
        reset_total=True,
    )
    audio_path = _ensure_project_audio(paths, manifest, options.audio_format, progress)
    should_upload = _parse_project_oss_upload(options.oss_upload, options.file_url)
    settings = load_settings(require_oss=should_upload)
    emit_progress(
        progress,
        "Resolving audio URL",
        step_index=step_offset + 2,
        step_total=transcribe_steps,
        reset_total=True,
    )
    file_url, file_url_source = _resolve_project_file_url(
        paths,
        manifest,
        audio_path,
        options,
        should_upload,
        settings,
        progress,
    )
    emit_progress(
        progress,
        "Submitting DashScope task",
        step_index=step_offset + 3,
        step_total=transcribe_steps,
        reset_total=True,
    )
    hotwords = _resolve_project_asr_hotwords(settings, options)
    task_response = _submit_project_task(settings, file_url, options, hotwords)
    task_id = _extract_task_id(task_response)
    emit_progress(progress, f"DashScope task submitted: {task_id}", completed=1, total=1)
    audio_duration_seconds = _audio_duration_seconds(paths.root, manifest, audio_path)
    cost = estimate_asr_cost(
        model=options.model,
        base_url=settings.dashscope_base_url,
        audio_duration_seconds=audio_duration_seconds,
    )
    wait_estimate = estimate_dashscope_wait(settings, model=options.model, audio_duration_seconds=audio_duration_seconds)
    emit_progress(
        progress,
        asr_wait_description(task_id, wait_estimate, status=None),
        step_index=step_offset + 4,
        step_total=transcribe_steps,
        total=asr_wait_total(wait_estimate),
        reset_total=True,
    )
    wait_started_at = time.monotonic()
    wait_status = "failed"
    try:
        wait_response = wait_transcription(
            settings=settings,
            task=task_response,
            poll_callback=lambda event: emit_dashscope_wait_poll(
                progress,
                task_id=task_id,
                estimate=wait_estimate,
                event=event,
            ),
        )
        wait_status = "succeeded"
    finally:
        wait_seconds = time.monotonic() - wait_started_at
        record_dashscope_wait(
            settings=settings,
            project_id=manifest.project_id,
            model=options.model,
            task_id=task_id,
            audio_duration_seconds=audio_duration_seconds,
            wait_seconds=wait_seconds,
            status=wait_status,
        )
    emit_progress(
        progress,
        "Downloading transcription result",
        step_index=step_offset + 5,
        step_total=transcribe_steps,
        reset_total=True,
    )
    raw_result = download_transcription_json(wait_response)
    emit_progress(progress, "Normalizing transcript")
    parsed_result = parse_transcription_result(raw_result)
    emit_progress(
        progress,
        "Writing transcript artifacts",
        step_index=step_offset + 6,
        step_total=transcribe_steps,
        reset_total=True,
    )
    _invalidate_downstream_artifacts(paths, manifest)
    _write_project_asr_outputs(paths, raw_result, parsed_result, options.generate_srt)
    emit_progress(progress, "Transcription complete", completed=1, total=1)
    _record_asr_metadata(manifest, task_id, file_url_source, options, parsed_result, hotwords, cost)
    manifest.status = "transcribed"
    save_manifest(paths.root, manifest)
    return ProjectTranscribeSummary(
        paths.root,
        task_id,
        file_url_source,
        len(parsed_result.detected_speakers),
        len(parsed_result.sentences),
        cost,
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

def summarize_project(
    project_dir: Path,
    *,
    model: str | None,
    update_title: bool,
    progress: CliProgressReporter | None = None,
) -> ProjectMeetingSummary:
    """
    Generate meeting summary artifacts from the project transcript.

    Args:
        project_dir: Project root.
        model: Optional DashScope text model override.
        update_title: Whether to replace the manifest title.
        progress: Optional progress reporter.

    Returns:
        Summary artifact paths.
    """
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    settings = load_settings(require_oss=False)
    emit_progress(progress, "Generating meeting summary")
    summary = generate_meeting_summary(result, settings=settings, model=model)
    json_path = safe_write_json(paths.exports_dir / "meeting_summary.json", summary.to_dict())
    summary_path = safe_write_text(paths.exports_dir / "meeting_summary.md", render_meeting_summary_markdown(summary))
    title_updated = bool(update_title and _can_replace_title(manifest, summary))
    if title_updated:
        manifest.title = summary.title
    _record_meeting_summary(manifest, paths, summary, summary_path, json_path)
    save_manifest(paths.root, manifest)
    emit_progress(progress, "Meeting summary ready", completed=1, total=1)
    return ProjectMeetingSummary(paths.root, summary.title, summary_path, json_path, summary.model, title_updated)

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
    source_sha256: str,
    progress: CliProgressReporter | None,
) -> ProjectManifest:
    """Build the initial project manifest."""
    stat = source.stat()
    return ProjectManifest(
        schema_version=SCHEMA_VERSION,
        project_id=_build_project_id(source_sha256),
        title=title,
        created_at=created_at,
        updated_at=created_at,
        status="created",
        source=ProjectSource(
            path=_relative_path(project_root, source),
            filename=source.name,
            size_bytes=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            sha256=source_sha256,
            meeting_time=meeting_time,
            original_path=str(original_source),
        ),
        speakers={"detected_ids": [], "mapped": {}},
    )

def _copy_source_into_project(source: Path, root: Path, progress: CliProgressReporter | None) -> Path:
    """Copy source media into the project directory."""
    target = root / "source" / source.name
    if source.resolve() == target.resolve():
        emit_progress(progress, "Source media already staged", total=1, completed=1)
        return target
    if target.exists():
        raise FileExistsError(f"Project source file already exists: {target}")
    _copy_file_with_progress(source, target, progress)
    return target

def _copy_file_with_progress(source: Path, target: Path, progress: CliProgressReporter | None) -> None:
    """
    Copy a file in chunks while reporting byte progress.

    Args:
        source: Existing source file.
        target: Destination path.
        progress: Optional progress reporter.

    Returns:
        None.
    """
    total_bytes = source.stat().st_size
    emit_progress(progress, "Copying source media", total=total_bytes, completed=0)
    try:
        with source.open("rb") as source_file, target.open("wb") as target_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                target_file.write(chunk)
                emit_progress(progress, advance=len(chunk))
        shutil.copystat(source, target)
    except Exception:
        target.unlink(missing_ok=True)
        raise

def _resolve_project_root(
    source: Path,
    projects_dir: Path | None,
    project_dir: Path | None,
    source_sha256: str,
) -> Path:
    """Resolve the directory for a new project."""
    if project_dir is not None:
        return project_dir.expanduser().resolve()
    base_dir = _projects_parent_dir(projects_dir)
    return (base_dir / _build_project_id(source_sha256)).resolve()

def _create_project_dirs(root: Path) -> None:
    """Create the project root and standard child directories."""
    ensure_directory(root)
    for name in PROJECT_DIRS:
        ensure_directory(root / name)

def _build_project_id(source_sha256: str) -> str:
    """Build a stable project id from source content."""
    return f"p-{source_sha256[:16]}"

def _slugify(value: str) -> str:
    """Convert a title into a filesystem-safe slug while preserving CJK text."""
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    return re.sub(r"-+", "-", cleaned).strip("-") or "project"

def _now_iso() -> str:
    """Return the current local timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _sha256_file(path: Path, progress: CliProgressReporter | None = None) -> str:
    """Compute SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    emit_progress(progress, "Hashing source media", total=path.stat().st_size, completed=0)
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
            emit_progress(progress, advance=len(chunk))
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
    metadata: dict[str, Any] = {
        "path": _relative_path(audio_path.parents[1], audio_path),
        "format": audio_format,
        "sample_rate": 16000,
        "channels": 1,
        "size_bytes": audio_path.stat().st_size,
    }
    duration_seconds = _probe_duration_safely(audio_path)
    if duration_seconds is not None:
        metadata["duration_seconds"] = duration_seconds
    return metadata

def _audio_duration_seconds(root: Path, manifest: ProjectManifest, audio_path: Path) -> float | None:
    """
    Return the project audio duration and backfill manifest metadata when possible.

    Args:
        root: Project root.
        manifest: Project manifest.
        audio_path: Local ASR audio file.

    Returns:
        Audio duration in seconds when available.
    """
    stored_duration = manifest.audio.get("duration_seconds")
    if isinstance(stored_duration, int | float) and stored_duration > 0:
        return float(stored_duration)
    duration_seconds = _probe_duration_safely(audio_path)
    if duration_seconds is None:
        return None
    manifest.audio["duration_seconds"] = duration_seconds
    save_manifest(root, manifest)
    return duration_seconds

def _probe_duration_safely(path: Path) -> float | None:
    """
    Probe media duration without failing the transcription workflow.

    Args:
        path: Local media path.

    Returns:
        Duration in seconds when ffprobe succeeds.
    """
    try:
        return probe_media_duration_seconds(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to probe media duration for %s: %s", path, exc)
        return None

def _ensure_project_audio(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    audio_format: str,
    progress: CliProgressReporter | None,
) -> Path:
    """Return existing project audio or generate it."""
    normalized_format = _normalize_audio_format(audio_format)
    existing_path = _manifest_audio_path(paths.root, manifest, normalized_format)
    if existing_path and existing_path.exists():
        emit_progress(progress, "Using existing project audio", advance=1)
        return existing_path
    return prepare_project_audio(paths.root, audio_format=normalized_format, progress=progress)

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

def _resolve_project_file_url(
    paths,
    manifest,
    audio_path,
    options,
    should_upload,
    settings,
    progress: CliProgressReporter | None,
) -> tuple[str, str]:
    """Resolve the HTTP URL submitted to DashScope without storing signed URLs."""
    if not should_upload:
        if not options.file_url:
            raise typer.BadParameter("--file-url is required when --oss-upload is false.")
        manifest.oss = {"mode": "provided_url"}
        emit_progress(progress, "Using provided file URL", advance=1)
        return options.file_url, "provided_url"
    object_key = f"meeting-asr/projects/{manifest.project_id}/{audio_path.name}"
    size_bytes = audio_path.stat().st_size
    upload_estimate = estimate_oss_upload(settings, size_bytes=size_bytes)
    emit_oss_upload_start(progress, estimate=upload_estimate, size_bytes=size_bytes)
    upload_started_at = time.monotonic()
    upload_status = "failed"
    try:
        file_url = upload_file_to_oss(
            audio_path,
            object_name=object_key,
            settings=settings,
            progress_callback=lambda consumed, total: emit_oss_upload_progress(
                progress,
                estimate=upload_estimate,
                consumed_bytes=consumed,
                total_bytes=total or size_bytes,
            ),
        )
        upload_status = "succeeded"
    finally:
        record_oss_upload(
            settings,
            project_id=manifest.project_id,
            object_key=object_key,
            size_bytes=size_bytes,
            upload_seconds=time.monotonic() - upload_started_at,
            status=upload_status,
        )
    expires_at = datetime.now(UTC) + timedelta(seconds=SIGNED_URL_EXPIRES_SECONDS)
    manifest.oss = _oss_metadata(settings.oss_bucket_name, object_key, expires_at)
    save_manifest(paths.root, manifest)
    emit_progress(progress, "Audio uploaded to OSS", total=size_bytes, completed=size_bytes)
    return file_url, "oss_signed_url"

def _oss_metadata(bucket_name: str | None, object_key: str, expires_at: datetime) -> dict[str, str | None]:
    """Build non-secret OSS metadata."""
    return {
        "mode": "oss_signed_url",
        "bucket": bucket_name,
        "object_key": object_key,
        "signed_url_expires_at": expires_at.isoformat(timespec="seconds"),
    }

def _submit_project_task(
    settings,
    file_url: str,
    options: ProjectTranscribeOptions,
    hotwords: AsrHotwordResolution,
):
    """Submit the DashScope project transcription task."""
    return submit_transcription(
        settings=settings,
        file_url=file_url,
        model=options.model,
        language_hints=_parse_languages(options.language),
        speaker_count=options.speaker_count,
        vocabulary_id=hotwords.vocabulary_id,
        timestamp_alignment_enabled=options.timestamp_alignment,
        disfluency_removal_enabled=options.disfluency_removal,
    )

def _resolve_project_asr_hotwords(settings, options: ProjectTranscribeOptions) -> AsrHotwordResolution:
    """Resolve ASR hotwords for one project transcription."""
    return resolve_asr_hotwords(mode=options.asr_hotwords, settings=settings, target_model=options.model)

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

def _invalidate_downstream_artifacts(paths: ProjectPaths, manifest: ProjectManifest) -> None:
    """
    Remove artifacts that depend on a previous ASR result.

    Args:
        paths: Project paths.
        manifest: Manifest to mutate.

    Returns:
        None.
    """
    for key in DOWNSTREAM_OUTPUT_KEYS:
        value = manifest.outputs.pop(key, None)
        if isinstance(value, str):
            _unlink_project_file(paths.root, value)
    for relative_path in DOWNSTREAM_ARTIFACT_PATHS:
        _unlink_project_file(paths.root, relative_path)
    for key in DOWNSTREAM_SPEAKER_KEYS:
        manifest.speakers.pop(key, None)
    manifest.asr.pop("summary_model", None)

def _unlink_project_file(project_root: Path, stored_path: str) -> None:
    """
    Delete one project-local file when it exists.

    Args:
        project_root: Project root boundary.
        stored_path: Relative or absolute stored path.

    Returns:
        None.
    """
    path = _project_local_file(project_root, stored_path)
    if path is not None and (path.is_file() or path.is_symlink()):
        path.unlink()

def _project_local_file(project_root: Path, stored_path: str) -> Path | None:
    """
    Resolve a stored artifact path without escaping the project root.

    Args:
        project_root: Project root boundary.
        stored_path: Relative or absolute stored path.

    Returns:
        Project-local path, or ``None`` when the path escapes the project.
    """
    root = project_root.expanduser().resolve()
    path = Path(stored_path).expanduser()
    candidate = path if path.is_absolute() else root / path
    try:
        candidate.resolve().relative_to(root)
    except ValueError:
        return None
    return candidate

def _record_asr_metadata(
    manifest,
    task_id,
    file_url_source,
    options,
    parsed_result,
    hotwords,
    cost: AsrCostEstimate | None,
) -> None:
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
        "hotwords": options.asr_hotwords,
        "vocabulary_id": hotwords.vocabulary_id,
        "vocabulary_source": hotwords.source,
        "hotword_count": hotwords.hotword_count,
        "hotword_hash": hotwords.vocabulary_hash,
        "hotword_error": hotwords.error,
    }
    if cost is not None:
        manifest.asr["cost"] = cost.to_dict()
    manifest.outputs.update(_default_output_paths())
    manifest.speakers["detected_ids"] = parsed_result.detected_speakers

def _record_meeting_summary(
    manifest: ProjectManifest,
    paths: ProjectPaths,
    summary: MeetingSummary,
    summary_path: Path,
    json_path: Path,
) -> None:
    """
    Record meeting summary metadata in the manifest.

    Args:
        manifest: Project manifest.
        paths: Project paths.
        summary: Generated summary.
        summary_path: Markdown summary path.
        json_path: JSON summary path.

    Returns:
        None.
    """
    manifest.asr["summary_model"] = summary.model
    manifest.outputs["meeting_summary"] = _relative_path(paths.root, summary_path)
    manifest.outputs["meeting_summary_json"] = _relative_path(paths.root, json_path)

def _can_replace_title(manifest: ProjectManifest, summary: MeetingSummary) -> bool:
    """
    Return whether an auto-generated summary title should replace the manifest title.

    Args:
        manifest: Project manifest.
        summary: Generated summary.

    Returns:
        True when the current title is still the source filename stem.
    """
    current_title = manifest.title.strip()
    source_stem = Path(manifest.source.filename).stem.strip()
    return bool(summary.title.strip()) and current_title == source_stem

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
