"""Project directory management for meeting transcription workflows."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
    TITLE_SOURCE_LLM,
    TITLE_SOURCE_MANUAL,
    TITLE_SOURCE_SOURCE,
    TITLE_SOURCE_UNKNOWN,
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
    load_speaker_person_mapping,
    load_transcript_result,
    render_named_speaker_text,
    render_named_srt,
    write_ignored_speakers,
    write_speaker_person_mapping,
    write_speaker_mapping,
)
from app.srt_utils import build_srt
from app.uploader import SIGNED_URL_EXPIRES_SECONDS, presign_oss_object, upload_file_to_oss
from app.utils import ensure_directory, safe_write_json, safe_write_text

PROJECT_DIRS = ("source", "audio", "asr", "speakers", "exports", "logs", "tmp")
PROJECT_HEARTBEAT_INTERVAL_SECONDS = 30.0
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
    "speakers/speaker_person_map.json",
    "speakers/speaker_matches.json",
)
DOWNSTREAM_SPEAKER_KEYS = ("mapped", "person_map", "matches", "voiceprints")

LOGGER = logging.getLogger(__name__)
_TITLE_TIME_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b\s*")


def create_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    hash_source: bool,
    variant: str | None = None,
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
        variant: Optional explicit experiment variant.
        source_sha256: Optional precomputed source SHA-256.
        progress: Optional progress reporter.

    Returns:
        Created manifest.
    """
    source = input_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    resolved_title, title_source = _resolve_initial_title(source, title)
    resolved_title = _title_with_meeting_time(resolved_title, meeting_time)
    created_at = _now_iso()
    resolved_sha256 = source_sha256 or _sha256_file(source, progress)
    resolved_variant = _normalize_project_variant(variant)
    root = _resolve_project_root(source, projects_dir, project_dir, resolved_sha256, resolved_variant)
    if (root / "project.json").exists():
        raise FileExistsError(f"Project already exists: {root}")
    _create_project_dirs(root)
    project_source = _copy_source_into_project(source, root, progress)
    manifest = _initial_manifest(
        project_source,
        source,
        root,
        resolved_title,
        title_source,
        created_at,
        meeting_time,
        resolved_sha256,
        resolved_variant,
        progress,
    )
    safe_write_text(root / "source" / "original.path", str(source) + "\n")
    safe_write_text(root / "notes.md", f"# {resolved_title}\n")
    save_manifest(root, manifest)
    return manifest


def _resolve_initial_title(source: Path, title: str | None) -> tuple[str, str]:
    """Return the initial project title and its provenance."""
    if title is None:
        return source.stem, TITLE_SOURCE_SOURCE
    cleaned_title = title.strip()
    if not cleaned_title:
        raise ValueError("Project title must not be empty.")
    return cleaned_title, TITLE_SOURCE_MANUAL


def _title_with_meeting_time(title: str, meeting_time: str | None) -> str:
    """Return a project title prefixed with ``YYYY-MM-DD HH:MM`` when meeting time is known."""
    cleaned_title = title.strip()
    prefix = _meeting_title_prefix(meeting_time)
    if prefix is None:
        return cleaned_title
    body = _TITLE_TIME_PREFIX_RE.sub("", cleaned_title, count=1).strip()
    return f"{prefix} {body}" if body else prefix


def _meeting_title_prefix(meeting_time: str | None) -> str | None:
    """Parse a meeting timestamp into the title prefix format."""
    if meeting_time is None:
        return None
    cleaned_time = meeting_time.strip()
    if not cleaned_time:
        return None
    try:
        return datetime.fromisoformat(cleaned_time).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})", cleaned_time)
        if match:
            return f"{match.group(1)} {match.group(2)}"
    return None


def create_or_reuse_project(
    input_path: Path,
    *,
    title: str | None,
    projects_dir: Path | None,
    project_dir: Path | None,
    meeting_time: str | None,
    hash_source: bool,
    variant: str | None = None,
    progress: CliProgressReporter | None = None,
) -> ProjectCreateSummary:
    """
    Return the existing project for a source video, or create a new one.

    Args:
        input_path: Local source media file.
        title: Optional human title for new projects.
        projects_dir: Optional parent directory used when project_dir is omitted.
        project_dir: Explicit project directory for a new project.
        meeting_time: Optional meeting start time string for new projects.
        hash_source: Deprecated compatibility flag. Source identity is always hashed.
        variant: Optional explicit experiment variant. Variants get distinct project ids.
        progress: Optional progress reporter.

    Returns:
        Project creation summary.
    """
    resolved_variant = _normalize_project_variant(variant)
    existing = _find_existing_project_for_create(
        input_path,
        projects_dir,
        project_dir,
        source_sha256=None,
        variant=resolved_variant,
    )
    if existing is not None:
        emit_progress(progress, "Using existing project", total=1, completed=1)
        return ProjectCreateSummary(existing, _load_reused_manifest(existing, title), False)
    source = input_path.expanduser().resolve()
    source_sha256 = _sha256_file(source, progress)
    existing = _find_existing_project_for_create(
        input_path,
        projects_dir,
        project_dir,
        source_sha256=source_sha256,
        variant=resolved_variant,
    )
    if existing is not None:
        emit_progress(progress, "Using existing project", total=1, completed=1)
        return ProjectCreateSummary(existing, _load_reused_manifest(existing, title), False)
    manifest = create_project(
        input_path,
        title=title,
        projects_dir=projects_dir,
        project_dir=project_dir,
        meeting_time=meeting_time,
        hash_source=hash_source,
        variant=resolved_variant,
        source_sha256=source_sha256,
        progress=progress,
    )
    project_root = _resolve_project_root(source, projects_dir, project_dir, source_sha256, resolved_variant)
    return ProjectCreateSummary(project_root, manifest, True)


def _find_existing_project_for_create(
    input_path: Path,
    projects_dir: Path | None,
    project_dir: Path | None,
    *,
    source_sha256: str | None,
    variant: str | None,
) -> Path | None:
    """Find an existing project across the normal parent and explicit path parent."""
    for parent in _project_reuse_parents(projects_dir, project_dir):
        existing = find_project_by_source(input_path, parent, source_sha256=source_sha256, variant=variant)
        if existing is not None:
            return existing
    return None


def _project_reuse_parents(projects_dir: Path | None, project_dir: Path | None) -> tuple[Path | None, ...]:
    """Return project parent directories that define the reuse namespace."""
    parents: list[Path | None] = [projects_dir]
    if project_dir is not None:
        explicit_parent = project_dir.expanduser().resolve().parent
        if projects_dir is None or explicit_parent != projects_dir.expanduser().resolve():
            parents.append(explicit_parent)
    return tuple(parents)


def _load_reused_manifest(project_dir: Path, title: str | None) -> ProjectManifest:
    """Load a reused project and apply an explicit manual title when provided."""
    manifest = load_manifest(project_dir)
    if title is None:
        return manifest
    cleaned_title = title.strip()
    if not cleaned_title:
        raise ValueError("Project title must not be empty.")
    normalized_title = _title_with_meeting_time(cleaned_title, manifest.source.meeting_time)
    if manifest.title == normalized_title and manifest.title_source == TITLE_SOURCE_MANUAL:
        return manifest
    manifest.title = normalized_title
    manifest.title_source = TITLE_SOURCE_MANUAL
    manifest.title_model = None
    save_manifest(project_dir, manifest)
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


def record_project_stage(
    project_dir: Path,
    *,
    stage: str,
    input_file: str | Path | None = None,
    external_ids: dict[str, object] | None = None,
    last_success: str | None = None,
    last_error: str | None = None,
    heartbeat: bool = False,
) -> ProjectManifest:
    """
    Persist the current long-running project stage.

    Args:
        project_dir: Project root.
        stage: Stable stage name.
        input_file: Optional source input path.
        external_ids: Non-secret external task identifiers.
        last_success: Last successful operation in this stage.
        last_error: Last recoverable or fatal error.
        heartbeat: Whether this update is a heartbeat instead of a stage transition.

    Returns:
        Updated manifest.
    """
    manifest = load_manifest(project_dir)
    _update_runtime_stage(
        manifest,
        stage=stage,
        input_file=input_file,
        external_ids=external_ids,
        last_success=last_success,
        last_error=last_error,
        heartbeat=heartbeat,
    )
    save_manifest(project_dir, manifest)
    return manifest


def _update_runtime_stage(
    manifest: ProjectManifest,
    *,
    stage: str,
    input_file: str | Path | None,
    external_ids: dict[str, object] | None,
    last_success: str | None,
    last_error: str | None,
    heartbeat: bool,
) -> None:
    """Mutate manifest runtime metadata for one stage or heartbeat."""
    now = _now_iso()
    runtime = dict(manifest.runtime)
    if not heartbeat or runtime.get("current_stage") != stage:
        runtime["current_stage"] = stage
        runtime["stage_started_at"] = now
        runtime.pop("last_error", None)
    runtime["last_heartbeat_at"] = now
    if input_file is not None:
        runtime["input_file"] = str(input_file)
    if last_success:
        runtime["last_success"] = last_success
    if last_error:
        runtime["last_error"] = {"at": now, "stage": stage, "message": last_error}
    if external_ids:
        merged = dict(runtime.get("external_ids") or {})
        merged.update(_safe_external_ids(external_ids))
        runtime["external_ids"] = merged
    manifest.runtime = runtime


def _safe_external_ids(values: dict[str, object]) -> dict[str, object]:
    """Return external identifiers with URL query strings redacted."""
    safe: dict[str, object] = {}
    for key, value in values.items():
        text = str(value)
        if "url" in key.lower() and "?" in text:
            safe[key] = text.split("?", 1)[0] + "?<redacted>"
        else:
            safe[key] = value
    return safe


def _emit_stage_event(
    progress: CliProgressReporter | None,
    *,
    manifest: ProjectManifest,
    project_dir: Path,
    stage: str,
    input_file: str | Path | None,
    external_ids: dict[str, object] | None = None,
    last_success: str | None = None,
    description: str | None = None,
    step_index: int | None = None,
    step_total: int | None = None,
    reset_total: bool = False,
    total: int | None = None,
    completed: int | None = None,
) -> None:
    """Emit one human-facing stage log and optional progress update."""
    emit_progress(
        progress,
        description or stage,
        step_index=step_index,
        step_total=step_total,
        reset_total=reset_total,
        total=total,
        completed=completed,
        log_kind="stage",
        stage=stage,
        project_id=manifest.project_id,
        project_path=str(project_dir),
        input_file=str(input_file) if input_file is not None else None,
        timestamp=_now_iso(),
        last_success=last_success,
        log_fields=tuple((external_ids or {}).items()),
    )


def _emit_heartbeat_event(
    progress: CliProgressReporter | None,
    *,
    manifest: ProjectManifest,
    project_dir: Path,
    stage: str,
    input_file: str | Path | None,
    elapsed_seconds: float,
    last_success: str,
    next_action: str,
    external_ids: dict[str, object] | None = None,
) -> None:
    """Emit one structured heartbeat without changing the Rich step row."""
    emit_progress(
        progress,
        None,
        log_kind="heartbeat",
        stage=stage,
        project_id=manifest.project_id,
        project_path=str(project_dir),
        input_file=str(input_file) if input_file is not None else None,
        timestamp=_now_iso(),
        elapsed_seconds=elapsed_seconds,
        last_success=last_success,
        next_action=next_action,
        log_fields=tuple((external_ids or {}).items()),
    )


def _run_with_stage_heartbeat(
    operation: Callable[[], Any],
    *,
    progress: CliProgressReporter | None,
    manifest: ProjectManifest,
    project_dir: Path,
    stage: str,
    input_file: str | Path | None,
    last_success: str,
    next_action: Callable[[int], str],
    external_ids: dict[str, object] | None = None,
    interval_seconds: float = PROJECT_HEARTBEAT_INTERVAL_SECONDS,
) -> Any:
    """Run a blocking operation while emitting periodic heartbeats."""
    if progress is None:
        return operation()
    started_at = time.monotonic()
    heartbeat_index = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        while True:
            try:
                return future.result(timeout=interval_seconds)
            except FutureTimeoutError:
                heartbeat_index += 1
                elapsed = time.monotonic() - started_at
                record_project_stage(
                    project_dir,
                    stage=stage,
                    input_file=input_file,
                    external_ids=external_ids,
                    last_success=last_success,
                    heartbeat=True,
                )
                _emit_heartbeat_event(
                    progress,
                    manifest=manifest,
                    project_dir=project_dir,
                    stage=stage,
                    input_file=input_file,
                    elapsed_seconds=elapsed,
                    last_success=last_success,
                    next_action=next_action(heartbeat_index),
                    external_ids=external_ids,
                )

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
    updated_meeting_time = manifest.source.meeting_time
    if meeting_time is not None:
        updated_meeting_time = meeting_time.strip() or None
        manifest.source.meeting_time = updated_meeting_time
    if title is not None:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Project title must not be empty.")
        manifest.title = _title_with_meeting_time(cleaned_title, updated_meeting_time)
        manifest.title_source = TITLE_SOURCE_MANUAL
        manifest.title_model = None
    elif meeting_time is not None:
        manifest.title = _title_with_meeting_time(manifest.title, updated_meeting_time)
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
    input_file = _project_input_file(paths.root, manifest)
    transcribe_steps = step_total or 6
    manifest = record_project_stage(paths.root, stage="audio extraction", input_file=input_file)
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="audio extraction",
        input_file=input_file,
        description="Preparing audio",
        step_index=step_offset + 1,
        step_total=transcribe_steps,
        reset_total=True,
    )
    audio_path = _ensure_project_audio(paths, manifest, options.audio_format, progress)
    should_upload = _parse_project_oss_upload(options.oss_upload, options.file_url)
    settings = load_settings(require_oss=should_upload)
    oss_external = _oss_stage_external_ids(manifest, audio_path, should_upload)
    manifest = record_project_stage(
        paths.root,
        stage="OSS upload/sign",
        input_file=input_file,
        external_ids=oss_external,
    )
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="OSS upload/sign",
        input_file=input_file,
        external_ids=oss_external,
        description="Resolving audio URL",
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
    manifest = load_manifest(paths.root)
    signed_external = _signed_url_external_ids(manifest, file_url_source)
    record_project_stage(
        paths.root,
        stage="OSS upload/sign",
        input_file=input_file,
        external_ids=signed_external,
        last_success="signed URL ready" if file_url_source == "oss_signed_url" else "file URL ready",
        heartbeat=True,
    )
    _emit_heartbeat_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="OSS upload/sign",
        input_file=input_file,
        elapsed_seconds=0.0,
        last_success="signed URL ready" if file_url_source == "oss_signed_url" else "file URL ready",
        next_action="submit ASR task",
        external_ids=signed_external,
    )
    manifest = record_project_stage(paths.root, stage="ASR submit", input_file=input_file)
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="ASR submit",
        input_file=input_file,
        description="Submitting DashScope task",
        step_index=step_offset + 3,
        step_total=transcribe_steps,
        reset_total=True,
    )
    hotwords = _resolve_project_asr_hotwords(settings, options)
    task_response = _submit_project_task(settings, file_url, options, hotwords)
    task_id = _extract_task_id(task_response)
    record_project_stage(
        paths.root,
        stage="ASR submit",
        input_file=input_file,
        external_ids={"dashscope_task_id": task_id, "model": options.model},
        last_success="DashScope task submitted",
        heartbeat=True,
    )
    emit_progress(progress, f"DashScope task submitted: {task_id}", completed=1, total=1)
    audio_duration_seconds = _audio_duration_seconds(paths.root, manifest, audio_path)
    cost = estimate_asr_cost(
        model=options.model,
        base_url=settings.dashscope_base_url,
        audio_duration_seconds=audio_duration_seconds,
    )
    wait_estimate = estimate_dashscope_wait(settings, model=options.model, audio_duration_seconds=audio_duration_seconds)
    manifest = load_manifest(paths.root)
    manifest = record_project_stage(
        paths.root,
        stage="ASR polling",
        input_file=input_file,
        external_ids={"dashscope_task_id": task_id, "model": options.model},
    )
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="ASR polling",
        input_file=input_file,
        external_ids={"dashscope_task_id": task_id, "model": options.model},
        description=asr_wait_description(task_id, wait_estimate, status=None),
        step_index=step_offset + 4,
        step_total=transcribe_steps,
        total=asr_wait_total(wait_estimate),
        reset_total=True,
    )
    wait_started_at = time.monotonic()
    wait_status = "failed"
    heartbeat = _HeartbeatThrottle(PROJECT_HEARTBEAT_INTERVAL_SECONDS)
    try:
        wait_response = wait_transcription(
            settings=settings,
            task=task_response,
            poll_callback=lambda event: _handle_asr_poll_event(
                progress=progress,
                paths=paths,
                manifest=manifest,
                input_file=input_file,
                task_id=task_id,
                model=options.model,
                estimate=wait_estimate,
                event=event,
                heartbeat=heartbeat,
            ),
        )
        wait_status = "succeeded"
    except Exception as exc:
        message = _asr_recovery_message(manifest.project_id, task_id, exc)
        record_project_stage(
            paths.root,
            stage="ASR polling",
            input_file=input_file,
            external_ids={"dashscope_task_id": task_id, "model": options.model},
            last_error=message,
            heartbeat=True,
        )
        raise RuntimeError(message) from exc
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
    manifest = record_project_stage(
        paths.root,
        stage="transcript materialized",
        input_file=input_file,
        external_ids={"dashscope_task_id": task_id},
        last_success="ASR task succeeded",
    )
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="transcript materialized",
        input_file=input_file,
        external_ids={"dashscope_task_id": task_id},
        last_success="ASR task succeeded",
        description="Downloading transcription result",
        step_index=step_offset + 5,
        step_total=transcribe_steps,
        reset_total=True,
    )
    raw_result = download_transcription_json(wait_response)
    emit_progress(progress, "Normalizing transcript")
    parsed_result = parse_transcription_result(raw_result)
    manifest = record_project_stage(
        paths.root,
        stage="final artifact write",
        input_file=input_file,
        last_success="transcript normalized",
    )
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="final artifact write",
        input_file=input_file,
        last_success="transcript normalized",
        description="Writing transcript artifacts",
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

def apply_project_speakers(
    project_dir: Path,
    mappings: dict[int, str],
    *,
    person_mapping: dict[int, int] | None = None,
    person_public_mapping: dict[int, str] | None = None,
    ignored_speaker_ids: Collection[int] | None = None,
    replace_existing: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Apply speaker names to project outputs.

    Args:
        project_dir: Project root.
        mappings: Explicit speaker display-name mappings.
        person_mapping: Optional project speaker to internal voiceprint person id mapping.
        person_public_mapping: Optional project speaker to voiceprint person public id mapping.
        ignored_speaker_ids: Optional explicit speaker ids to keep anonymous.
        replace_existing: Replace stored speaker mappings instead of merging into them.

    Returns:
        Paths for map, transcript, and SRT.
    """
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    existing_mapping = {} if replace_existing else _load_existing_speaker_mapping(paths.speakers_dir / "speaker_map.json")
    explicit_mapping = _merge_speaker_mapping(result, mappings)
    resolved_mapping = _merge_speaker_mapping(result, existing_mapping | explicit_mapping)
    mapping_path = write_speaker_mapping(paths.speakers_dir / "speaker_map.json", resolved_mapping)
    if ignored_speaker_ids is not None:
        _write_project_ignored_speakers(paths, manifest, result, ignored_speaker_ids)
    person_map_path = paths.speakers_dir / "speaker_person_map.json"
    existing_person_mapping = {} if replace_existing else load_speaker_person_mapping(person_map_path)
    stored_person_mapping = _resolve_speaker_person_mapping(
        resolved_mapping,
        existing_mapping,
        existing_person_mapping,
        person_mapping or {},
        person_public_mapping or {},
    )
    if stored_person_mapping:
        write_speaker_person_mapping(person_map_path, stored_person_mapping)
        manifest.speakers["person_map"] = {str(key): value for key, value in sorted(stored_person_mapping.items())}
    else:
        if person_map_path.exists():
            person_map_path.unlink()
        manifest.speakers.pop("person_map", None)
    transcript_path = safe_write_text(
        paths.exports_dir / "transcript_named.txt",
        render_named_speaker_text(result, resolved_mapping),
    )
    srt_path = safe_write_text(paths.exports_dir / "subtitle_named.srt", render_named_srt(result, resolved_mapping))
    corrected_result = _load_corrected_result(paths)
    if corrected_result is not None:
        corrected_transcript_path = safe_write_text(
            paths.exports_dir / "transcript_named_corrected.txt",
            render_named_speaker_text(corrected_result, resolved_mapping),
        )
        corrected_srt_path = safe_write_text(
            paths.exports_dir / "subtitle_named_corrected.srt",
            render_named_srt(corrected_result, resolved_mapping),
        )
        manifest.outputs["corrected_named_transcript"] = _relative_path(paths.root, corrected_transcript_path)
        manifest.outputs["corrected_named_subtitle"] = _relative_path(paths.root, corrected_srt_path)
    manifest.speakers["detected_ids"] = result.detected_speakers
    manifest.speakers["mapped"] = {str(key): value for key, value in sorted(resolved_mapping.items())}
    manifest.outputs["named_transcript"] = _relative_path(paths.root, transcript_path)
    manifest.outputs["named_subtitle"] = _relative_path(paths.root, srt_path)
    manifest.status = "corrected" if corrected_result is not None else "named"
    save_manifest(paths.root, manifest)
    return mapping_path, transcript_path, srt_path


def _write_project_ignored_speakers(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_ids: Collection[int],
) -> None:
    """Persist explicit ignored-speaker state for project review."""
    active_speakers = set(result.detected_speakers)
    ignored = {int(speaker_id) for speaker_id in speaker_ids if int(speaker_id) in active_speakers}
    ignore_path = paths.speakers_dir / "speaker_ignore.json"
    if ignored:
        write_ignored_speakers(ignore_path, ignored)
        manifest.speakers["ignored"] = sorted(ignored)
        return
    if ignore_path.exists():
        ignore_path.unlink()
    manifest.speakers.pop("ignored", None)


def _load_corrected_result(paths: ProjectPaths) -> TranscriptResult | None:
    """Load corrected transcript artifacts when vocabulary correction exists."""
    corrected_path = paths.asr_dir / "sentences_corrected.json"
    if not corrected_path.exists():
        return None
    return load_transcript_result(corrected_path)


def summarize_project(
    project_dir: Path,
    *,
    model: str | None,
    update_title: bool,
    progress: CliProgressReporter | None = None,
) -> ProjectMeetingSummary:
    """
    Generate meeting memory-index artifacts from the project transcript.

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
    input_file = _project_input_file(paths.root, manifest)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    settings = load_settings(require_oss=False)
    model_label = model or getattr(settings, "dashscope_summary_model", "configured-default")
    manifest = record_project_stage(
        paths.root,
        stage="summary",
        input_file=input_file,
        external_ids={"model": model_label},
    )
    _emit_stage_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="summary",
        input_file=input_file,
        external_ids={"model": model_label},
        description="Generating meeting memory index",
    )
    try:
        summary = _run_with_stage_heartbeat(
            lambda: generate_meeting_summary(result, settings=settings, model=model),
            progress=progress,
            manifest=manifest,
            project_dir=paths.root,
            stage="summary",
            input_file=input_file,
            last_success="summary request submitted",
            next_action=lambda index: f"waiting for summary response heartbeat {index}",
            external_ids={"model": model_label},
        )
    except Exception as exc:
        message = _project_stage_recovery_message(manifest.project_id, "summary", exc)
        record_project_stage(paths.root, stage="summary", input_file=input_file, last_error=message, heartbeat=True)
        raise RuntimeError(message) from exc
    json_path = safe_write_json(paths.exports_dir / "meeting_summary.json", summary.to_dict())
    summary_path = safe_write_text(paths.exports_dir / "meeting_summary.md", render_meeting_summary_markdown(summary))
    title_updated = bool(update_title and _can_replace_title(manifest, summary))
    if title_updated:
        manifest.title = _title_with_meeting_time(summary.title, manifest.source.meeting_time)
        manifest.title_source = TITLE_SOURCE_LLM
        manifest.title_model = summary.model
    elif _looks_like_legacy_custom_title(manifest):
        manifest.title_source = TITLE_SOURCE_MANUAL
        manifest.title_model = None
    _record_meeting_summary(manifest, paths, summary, summary_path, json_path)
    save_manifest(paths.root, manifest)
    emit_progress(progress, "Meeting memory index ready", completed=1, total=1)
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


def resolve_project_audio_path(project_root: Path, manifest: ProjectManifest) -> Path:
    """
    Resolve the media path that matches ASR sentence timestamps.

    Args:
        project_root: Project root.
        manifest: Loaded manifest.

    Returns:
        Project ASR audio when available, otherwise the original source path.
    """
    audio_path = manifest.audio.get("path")
    if isinstance(audio_path, str) and audio_path.strip():
        resolved = Path(audio_path)
        if not resolved.is_absolute():
            resolved = project_root / resolved
        if resolved.exists():
            return resolved.resolve()
    for name in ("audio.flac", "audio.wav", "audio.mp3", "audio.m4a"):
        candidate = project_root / "audio" / name
        if candidate.exists():
            return candidate.resolve()
    return resolve_project_source_path(project_root, manifest)


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
    title_source: str,
    created_at: str,
    meeting_time: str | None,
    source_sha256: str,
    variant: str | None,
    progress: CliProgressReporter | None,
) -> ProjectManifest:
    """Build the initial project manifest."""
    stat = source.stat()
    return ProjectManifest(
        schema_version=SCHEMA_VERSION,
        project_id=_build_project_id(source_sha256, variant),
        title=title,
        title_source=title_source,
        title_model=None,
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
            variant=variant,
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
    variant: str | None,
) -> Path:
    """Resolve the directory for a new project."""
    if project_dir is not None:
        return project_dir.expanduser().resolve()
    base_dir = _projects_parent_dir(projects_dir)
    return (base_dir / _build_project_id(source_sha256, variant)).resolve()

def _create_project_dirs(root: Path) -> None:
    """Create the project root and standard child directories."""
    ensure_directory(root)
    for name in PROJECT_DIRS:
        ensure_directory(root / name)

def _build_project_id(source_sha256: str, variant: str | None = None) -> str:
    """Build a stable project id from source content."""
    base_id = f"p-{source_sha256[:16]}"
    return base_id if variant is None else f"{base_id}-v-{variant}"


def _normalize_project_variant(variant: str | None) -> str | None:
    """Normalize an optional experiment variant into a project-id-safe suffix."""
    if variant is None:
        return None
    cleaned = variant.strip()
    if not cleaned:
        raise ValueError("Project variant must not be empty.")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", cleaned.lower()).strip("-._")
    if not slug:
        raise ValueError("Project variant must contain ASCII letters or numbers.")
    return slug[:40]

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


class _HeartbeatThrottle:
    """Decide when a long-running polling loop should emit a heartbeat."""

    def __init__(self, interval_seconds: float) -> None:
        """
        Create a throttle.

        Args:
            interval_seconds: Minimum elapsed seconds between heartbeats.
        """
        self._interval_seconds = interval_seconds
        self._last_elapsed = 0.0

    def should_emit(self, elapsed_seconds: float) -> bool:
        """
        Return whether a heartbeat is due.

        Args:
            elapsed_seconds: Elapsed seconds reported by the poll loop.

        Returns:
            True when at least one interval has passed since the last heartbeat.
        """
        if elapsed_seconds < self._interval_seconds:
            return False
        if elapsed_seconds - self._last_elapsed < self._interval_seconds:
            return False
        self._last_elapsed = elapsed_seconds
        return True


def _handle_asr_poll_event(
    *,
    progress: CliProgressReporter | None,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    input_file: str,
    task_id: str,
    model: str,
    estimate,
    event,
    heartbeat: _HeartbeatThrottle,
) -> None:
    """Update ASR progress and emit durable heartbeat lines when due."""
    emit_dashscope_wait_poll(progress, task_id=task_id, estimate=estimate, event=event)
    if not heartbeat.should_emit(event.elapsed_seconds):
        return
    external_ids = {"dashscope_task_id": task_id, "model": model, "status": event.status or "unknown"}
    record_project_stage(
        paths.root,
        stage="ASR polling",
        input_file=input_file,
        external_ids=external_ids,
        last_success=f"status={event.status or 'unknown'}",
        heartbeat=True,
    )
    _emit_heartbeat_event(
        progress,
        manifest=manifest,
        project_dir=paths.root,
        stage="ASR polling",
        input_file=input_file,
        elapsed_seconds=event.elapsed_seconds,
        last_success=f"status={event.status or 'unknown'}",
        next_action=f"next poll in {event.wait_seconds:.0f}s",
        external_ids=external_ids,
    )


def _project_input_file(root: Path, manifest: ProjectManifest) -> str:
    """Return the best source input path for progress logs."""
    return manifest.source.original_path or str(resolve_project_source_path(root, manifest))


def _oss_stage_external_ids(manifest: ProjectManifest, audio_path: Path, should_upload: bool) -> dict[str, object]:
    """Return non-secret OSS stage identifiers."""
    if not should_upload:
        return {"file_url_source": "provided_url"}
    return {"oss_object_key": _project_oss_object_key(manifest, audio_path)}


def _signed_url_external_ids(manifest: ProjectManifest, file_url_source: str) -> dict[str, object]:
    """Return non-secret signed URL status fields."""
    if file_url_source != "oss_signed_url":
        return {"file_url_source": file_url_source}
    return {
        "oss_object_key": manifest.oss.get("object_key", "-"),
        "signed_url_ready": "true",
        "signed_url_expires_at": manifest.oss.get("signed_url_expires_at", "-"),
    }


def _project_oss_object_key(manifest: ProjectManifest, audio_path: Path) -> str:
    """Return the stable OSS object key for one project audio object."""
    return f"meeting-asr/projects/{manifest.project_id}/{audio_path.name}"


def _asr_recovery_message(project_id: str, task_id: str, error: Exception) -> str:
    """Return an ASR failure message with actionable recovery commands."""
    return (
        f"Project run failed: project_id={project_id} stage=ASR polling "
        f"dashscope_task_id={task_id} error={error}. "
        f"Inspect: meeting-asr project show {project_id}. "
        f"Review: meeting-asr project review {project_id}. "
        f"Retry ASR: meeting-asr project transcribe {project_id}."
    )


def _project_stage_recovery_message(project_id: str, stage: str, error: Exception) -> str:
    """Return a generic project stage failure with recovery commands."""
    return (
        f"Project run failed: project_id={project_id} stage={stage} error={error}. "
        f"Show: meeting-asr project show {project_id}. "
        f"Review: meeting-asr project review {project_id}."
    )

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
    object_key = _project_oss_object_key(manifest, audio_path)
    if _can_reuse_project_oss_object(manifest, object_key):
        try:
            return _presign_project_oss_object(paths, manifest, settings, object_key, progress)
        except Exception as error:
            emit_progress(
                progress,
                f"Existing OSS object presign failed; uploading audio instead ({type(error).__name__}: {error})",
                advance=1,
            )
    return _upload_project_audio_to_oss(paths, manifest, audio_path, settings, object_key, progress)


def _can_reuse_project_oss_object(manifest: ProjectManifest, object_key: str) -> bool:
    """Return true when manifest points at this project's current audio object."""
    return manifest.oss.get("mode") == "oss_signed_url" and manifest.oss.get("object_key") == object_key


def _presign_project_oss_object(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    settings,
    object_key: str,
    progress: CliProgressReporter | None,
) -> tuple[str, str]:
    """Presign an already uploaded project audio object and refresh manifest metadata."""
    file_url = presign_oss_object(
        object_key,
        settings=settings,
        expires_seconds=SIGNED_URL_EXPIRES_SECONDS,
    )
    expires_at = datetime.now(UTC) + timedelta(seconds=SIGNED_URL_EXPIRES_SECONDS)
    manifest.oss = _oss_metadata(settings.oss_bucket_name, object_key, expires_at)
    save_manifest(paths.root, manifest)
    emit_progress(progress, "Reused existing OSS object", advance=1)
    return file_url, "oss_signed_url"


def _upload_project_audio_to_oss(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    audio_path: Path,
    settings,
    object_key: str,
    progress: CliProgressReporter | None,
) -> tuple[str, str]:
    """Upload project audio to OSS and persist non-secret signing metadata."""
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
    # Keywords are derived from the transcript; once the transcript is
    # invalidated the keyword list is also stale and must not survive
    # into the next project list render.
    manifest.meeting_keywords = []

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
    Record meeting memory-index metadata in the manifest.

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
    manifest.meeting_keywords = list(summary.keywords)

def _can_replace_title(manifest: ProjectManifest, summary: MeetingSummary) -> bool:
    """
    Return whether an auto-generated summary title should replace the manifest title.

    Args:
        manifest: Project manifest.
        summary: Generated summary.

    Returns:
        True when the current title is automatic rather than manually edited.
    """
    if not summary.title.strip():
        return False
    if manifest.title_source in {TITLE_SOURCE_SOURCE, TITLE_SOURCE_LLM}:
        return True
    if manifest.title_source == TITLE_SOURCE_MANUAL:
        return False
    return _looks_like_legacy_source_title(manifest)


def _looks_like_legacy_source_title(manifest: ProjectManifest) -> bool:
    """Infer whether an old manifest title still came from the source filename."""
    current_title = manifest.title.strip()
    source_stem = Path(manifest.source.filename).stem.strip()
    return manifest.title_source == TITLE_SOURCE_UNKNOWN and current_title == source_stem


def _looks_like_legacy_custom_title(manifest: ProjectManifest) -> bool:
    """Infer whether an old unknown title should be treated as manually preserved."""
    current_title = manifest.title.strip()
    source_stem = Path(manifest.source.filename).stem.strip()
    return manifest.title_source == TITLE_SOURCE_UNKNOWN and bool(current_title) and current_title != source_stem

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
    """Return explicit mappings for active speakers only."""
    active_speakers = set(result.detected_speakers)
    resolved: dict[int, str] = {}
    for speaker_id, name in mappings.items():
        name_text = name.strip()
        if speaker_id not in active_speakers or name_text == speaker_id_to_label(speaker_id):
            continue
        resolved[speaker_id] = name_text
    return resolved


def _load_existing_speaker_mapping(path: Path) -> dict[int, str]:
    """Load persisted speaker names from a project speaker map file."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


def _resolve_speaker_person_mapping(
    speaker_mapping: dict[int, str],
    existing_speaker_mapping: dict[int, str],
    existing_person_mapping: dict[int, int | str],
    explicit_person_mapping: dict[int, int],
    explicit_public_mapping: dict[int, str],
) -> dict[int, int | str]:
    """Resolve project speaker to voiceprint person refs without keeping stale names."""
    resolved: dict[int, int | str] = {}
    for speaker_id, speaker_name in sorted(speaker_mapping.items()):
        public_id = explicit_public_mapping.get(speaker_id, "").strip()
        if public_id:
            resolved[speaker_id] = public_id
            continue
        person_id = explicit_person_mapping.get(speaker_id, 0)
        if person_id > 0:
            resolved[speaker_id] = person_id
            continue
        unchanged_existing_name = existing_speaker_mapping.get(speaker_id) == speaker_name
        if unchanged_existing_name and speaker_id in existing_person_mapping:
            resolved[speaker_id] = existing_person_mapping[speaker_id]
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
