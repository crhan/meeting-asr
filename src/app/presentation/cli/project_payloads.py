"""Machine-readable payloads for project CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.project_models import ProjectListItem, ProjectManifest, ProjectPaths
from app.core.project_workflow import load_project_workflow_summary, project_workflow_summary, workflow_payload
from app.postprocess import speaker_id_to_label
from app.speaker_labeling import build_speaker_summaries, load_project_ignored_speakers, load_transcript_result
from app.speaker_match_status import (
    MATCH_STATUS_IGNORED,
    MATCH_STATUS_MATCHED,
    accepted_match_name,
    best_candidate_name,
    best_candidate_score,
    effective_match_status,
    match_threshold,
    speaker_id_from_match,
)


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
    ignored_speakers = sorted(load_project_ignored_speakers(paths.root))
    speakers = _speakers_payload(paths, manifest, ignored_speakers)
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
        "ignored_speakers": ignored_speakers,
        "speakers": speakers,
        "outputs": manifest.outputs,
    }


def _speakers_payload(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    ignored_speakers: list[int],
) -> list[dict[str, Any]]:
    """
    Build the per-speaker status rows for ``project show --json``.

    Args:
        paths: Resolved project paths.
        manifest: Loaded project manifest.
        ignored_speakers: Sorted ignored project speaker ids.

    Returns:
        One row per detected speaker, ordered by speaker id.
    """
    ignored_set = set(ignored_speakers)
    matches_by_id, default_threshold = _load_match_rows_by_id(paths.speakers_dir / "speaker_matches.json")
    mapped = _mapped_names(manifest.speakers.get("mapped"))
    sentences_path = paths.asr_dir / "sentences.json"
    summaries: list[Any] = []
    if sentences_path.exists():
        try:
            summaries = build_speaker_summaries(load_transcript_result(sentences_path), sample_count=1)
        except (OSError, ValueError, TypeError, KeyError):
            summaries = []
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for summary in summaries:
        speaker_id = int(summary.speaker_id)
        seen_ids.add(speaker_id)
        rows.append(
            _speaker_payload_row(
                speaker_id=speaker_id,
                label=summary.anonymous_label,
                sample_count=summary.segment_count,
                ignored_set=ignored_set,
                mapped=mapped,
                matches_by_id=matches_by_id,
                default_threshold=default_threshold,
            )
        )
    detected_ids = manifest.speakers.get("detected_ids") or []
    for raw_id in detected_ids:
        try:
            speaker_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if speaker_id in seen_ids:
            continue
        seen_ids.add(speaker_id)
        rows.append(
            _speaker_payload_row(
                speaker_id=speaker_id,
                label=speaker_id_to_label(speaker_id),
                sample_count=0,
                ignored_set=ignored_set,
                mapped=mapped,
                matches_by_id=matches_by_id,
                default_threshold=default_threshold,
            )
        )
    rows.sort(key=lambda row: row["speaker_id"])
    return rows


def _speaker_payload_row(
    *,
    speaker_id: int,
    label: str,
    sample_count: int,
    ignored_set: set[int],
    mapped: dict[int, str],
    matches_by_id: dict[int, dict[str, Any]],
    default_threshold: float | None,
) -> dict[str, Any]:
    """Build one entry for the ``speakers`` payload list."""
    name = mapped.get(speaker_id)
    match = matches_by_id.get(speaker_id)
    status = _speaker_status(speaker_id, match, name, ignored_set, label)
    row: dict[str, Any] = {
        "speaker_id": speaker_id,
        "label": label,
        "name": name,
        "status": status,
        "sample_count": sample_count,
        "ignored": speaker_id in ignored_set,
    }
    if match is not None:
        threshold = match_threshold(match, default_threshold)
        if status == MATCH_STATUS_MATCHED:
            candidate = accepted_match_name(match)
        else:
            candidate = best_candidate_name(match)
        score = best_candidate_score(match)
        row["match"] = {
            "candidate": candidate,
            "score": score,
            "threshold": threshold,
        }
    return row


def _speaker_status(
    speaker_id: int,
    match: dict[str, Any] | None,
    name: str | None,
    ignored_set: set[int],
    label: str,
) -> str:
    """Return the user-facing speaker status string."""
    if speaker_id in ignored_set:
        return MATCH_STATUS_IGNORED
    if match is not None:
        status = effective_match_status(match, ignored_speaker_ids=ignored_set)
        if status != "no-candidate":
            return status
    if name and name != label:
        return "matched"
    return "unnamed"


def _load_match_rows_by_id(match_path: Path) -> tuple[dict[int, dict[str, Any]], float | None]:
    """Return speaker_matches.json rows keyed by speaker_id."""
    if not match_path.exists():
        return {}, None
    try:
        payload = json.loads(match_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if not isinstance(payload, dict):
        return {}, None
    default_threshold = _safe_float(payload.get("threshold"))
    rows: dict[int, dict[str, Any]] = {}
    for item in payload.get("matches", []) or []:
        if not isinstance(item, dict):
            continue
        speaker_id = speaker_id_from_match(item)
        if speaker_id is None:
            continue
        if default_threshold is not None and item.get("threshold") is None:
            item = {**item, "threshold": default_threshold}
        rows[speaker_id] = item
    return rows, default_threshold


def _mapped_names(value: object) -> dict[int, str]:
    """Coerce manifest.speakers.mapped into an int-keyed dict."""
    if not isinstance(value, dict):
        return {}
    mapped: dict[int, str] = {}
    for key, raw_name in value.items():
        try:
            speaker_id = int(key)
        except (TypeError, ValueError):
            continue
        name = str(raw_name).strip()
        if name:
            mapped[speaker_id] = name
    return mapped


def _safe_float(value: object) -> float | None:
    """Return a float value when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "meeting_keywords": list(project.meeting_keywords),
    }
