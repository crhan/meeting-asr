"""Load project state for the speaker review TUI."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.project_manager import ProjectManifest, load_manifest, project_paths, resolve_project_source_path
from app.speaker_labeling import load_ignored_speakers, load_transcript_result
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    voiceprint_match_status,
)
from app.presentation.tui.speaker_matches import (
    SpeakerMatchCandidate,
    accepted_review_name,
    accepted_review_person_id,
    accepted_review_person_public_id,
    load_match_candidates,
)
from app.presentation.tui.speaker_models import (
    ReviewSpeaker,
    SpeakerClusterDiagnostic,
    SpeakerClusterSampleScore,
    SpeakerReviewSession,
)
from app.presentation.tui.speaker_people import load_existing_person_mapping, load_people
from app.presentation.tui.speaker_status import SpeakerReviewOverview, VoiceprintReviewProgress
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_embedded_sample_ids,
    list_voiceprint_samples_for_project,
)


def load_speaker_review_session(
    project_dir: Path,
    *,
    page_size: int | None = None,
    store_dir: Path | None = None,
    allow_correction: bool = False,
) -> SpeakerReviewSession:
    """
    Load all data needed by the speaker review TUI.

    Args:
        project_dir: Project root.
        page_size: Optional samples-per-page override.
        store_dir: Optional voiceprint store directory.
        allow_correction: Whether the TUI may launch transcript correction.

    Returns:
        Speaker review session.
    """
    paths = project_paths(project_dir)
    result = _load_review_transcript_result(paths.asr_dir)
    segments_by_speaker = _segments_by_speaker(result.sentences)
    if not segments_by_speaker:
        raise RuntimeError("No detected speakers found in the transcript.")
    manifest = load_manifest(paths.root)
    source_media = resolve_project_source_path(paths.root, manifest)
    mapping_path = paths.speakers_dir / "speaker_map.json"
    ignore_path = paths.speakers_dir / "speaker_ignore.json"
    match_path = paths.speakers_dir / "speaker_matches.json"
    cluster_path = paths.speakers_dir / "speaker_cluster_quality.json"
    mapping = _load_existing_mapping(mapping_path)
    ignored_speaker_ids = load_ignored_speakers(ignore_path)
    matches = load_match_candidates(match_path)
    people = load_people(store_dir)
    person_mapping, person_public_mapping = load_existing_person_mapping(
        paths.speakers_dir / "speaker_person_map.json",
        people,
    )
    speakers = _build_review_speakers(
        segments_by_speaker,
        mapping,
        ignored_speaker_ids,
        person_mapping,
        person_public_mapping,
        matches,
    )
    return SpeakerReviewSession(
        project_dir=paths.root,
        source_media=source_media,
        overview=_build_review_overview(
            manifest=manifest,
            source_media=source_media,
            sentences=result.sentences,
            match_file_exists=match_path.exists(),
            saved_names_by_speaker=mapping,
            saved_ignored_speaker_ids=ignored_speaker_ids,
            store_dir=store_dir,
        ),
        speakers=speakers,
        people_names=[person.name for person in people],
        page_size=page_size,
        allow_correction=allow_correction,
        people=tuple(people),
        store_dir=store_dir,
        projects_dir=paths.root.parent,
        cluster_diagnostics=_load_cluster_diagnostics(cluster_path),
    )


def load_voiceprint_review_progress(project_id: str, store_dir: Path | None) -> VoiceprintReviewProgress:
    """
    Load project-scoped voiceprint capture and embedding state.

    Args:
        project_id: Project id from the manifest.
        store_dir: Optional voiceprint store directory.

    Returns:
        Voiceprint progress for the current project.
    """
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_voiceprint_samples_for_project(project_id, db_path)
    names_by_speaker: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        names_by_speaker[sample.project_speaker_id].add(sample.speaker_name)
    model, embedded_ids, embed_error = _load_embedding_state(db_path)
    return VoiceprintReviewProgress(
        captured_names_by_speaker={
            speaker_id: frozenset(names)
            for speaker_id, names in names_by_speaker.items()
        },
        captured_sample_ids=frozenset(sample.sample_id for sample in samples),
        embed_model=model,
        embedded_sample_ids=embedded_ids,
        embed_error=embed_error,
    )


def _build_review_overview(
    *,
    manifest: ProjectManifest,
    source_media: Path,
    sentences: list[SentenceSegment],
    match_file_exists: bool,
    saved_names_by_speaker: dict[int, str],
    saved_ignored_speaker_ids: set[int],
    store_dir: Path | None,
) -> SpeakerReviewOverview:
    """Build the immutable project state shown by the TUI."""
    return SpeakerReviewOverview(
        project_id=manifest.project_id,
        title=manifest.title,
        project_status=manifest.status,
        source_name=manifest.source.filename or source_media.name,
        duration_ms=_project_duration_ms(sentences),
        match_file_exists=match_file_exists,
        saved_names_by_speaker=dict(saved_names_by_speaker),
        voiceprint=load_voiceprint_review_progress(manifest.project_id, store_dir),
        saved_ignored_speaker_ids=frozenset(saved_ignored_speaker_ids),
    )


def _load_embedding_state(db_path: Path) -> tuple[str | None, frozenset[int] | None, str | None]:
    """Load embedded sample ids for the configured voiceprint model."""
    try:
        _, model = resolve_voiceprint_embedding_options(provider=None, model=None)
        return model, frozenset(list_embedded_sample_ids(model, db_path)), None
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def _project_duration_ms(sentences: list[SentenceSegment]) -> int:
    """Return the transcript duration from the latest sentence end."""
    return max((sentence.end_time_ms for sentence in sentences), default=0)


def _load_review_transcript_result(asr_dir: Path) -> TranscriptResult:
    """Load the transcript version humans should review."""
    corrected = asr_dir / "sentences_corrected.json"
    if corrected.exists():
        return load_transcript_result(corrected, include_low_information=True)
    return load_transcript_result(asr_dir / "sentences.json", include_low_information=True)


def _segments_by_speaker(sentences: list[SentenceSegment]) -> dict[int, list[SentenceSegment]]:
    """Group non-empty transcript segments by speaker id."""
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for sentence in sentences:
        if sentence.speaker_id is not None and sentence.text.strip():
            grouped[sentence.speaker_id].append(sentence)
    return dict(grouped)


def _load_existing_mapping(path: Path) -> dict[int, str]:
    """Load the current project speaker map if it exists."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


def _load_cluster_diagnostics(path: Path) -> dict[int, SpeakerClusterDiagnostic]:
    """Load optional project speaker cluster diagnostics for review display."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    speakers = payload.get("speakers") if isinstance(payload, dict) else None
    if not isinstance(speakers, list):
        return {}
    diagnostics: dict[int, SpeakerClusterDiagnostic] = {}
    for item in speakers:
        diagnostic = _cluster_diagnostic(item)
        if diagnostic is not None:
            diagnostics[diagnostic.speaker_id] = diagnostic
    return diagnostics


def _cluster_diagnostic(item: object) -> SpeakerClusterDiagnostic | None:
    """Convert one serialized cluster report row into a TUI diagnostic."""
    if not isinstance(item, dict):
        return None
    try:
        speaker_id = int(item["speaker_id"])
    except (KeyError, TypeError, ValueError):
        return None
    samples = _cluster_sample_scores(item.get("samples"))
    return SpeakerClusterDiagnostic(
        speaker_id=speaker_id,
        status=str(item.get("status") or "unknown"),
        centroid_mean=_optional_float(item.get("centroid_mean")),
        centroid_min=_optional_float(item.get("centroid_min")),
        clip_count=_optional_int(item.get("clip_count")),
        segment_count=_optional_int(item.get("segment_count")),
        warning_clip_count=_optional_int(item.get("warning_clip_count")),
        critical_clip_count=_optional_int(item.get("critical_clip_count")),
        component_count=_optional_int(item.get("component_count")),
        component_sizes=tuple(_optional_int(value) for value in _list_values(item.get("component_sizes"))),
        warnings=tuple(str(value) for value in _list_values(item.get("warnings"))),
        samples={sample.key: sample for sample in samples},
    )


def _cluster_sample_scores(value: object) -> list[SpeakerClusterSampleScore]:
    """Convert serialized sample score rows into TUI rows."""
    rows: list[SpeakerClusterSampleScore] = []
    for item in _list_values(value):
        if not isinstance(item, dict):
            continue
        try:
            begin = int(item["begin_time_ms"])
            end = int(item["end_time_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        sentence_id = item.get("sentence_id")
        rows.append(
            SpeakerClusterSampleScore(
                sentence_id=None if sentence_id is None else _optional_int(sentence_id),
                begin_time_ms=begin,
                end_time_ms=end,
                score=_optional_float(item.get("centroid_score")),
                status=str(item.get("status") or "unknown"),
                text=str(item.get("text") or ""),
                nearest_speaker_id=_optional_int_or_none(item.get("nearest_speaker_id")),
                nearest_score=_optional_float(item.get("nearest_score")),
                margin_score=_optional_float(item.get("margin_score")),
            )
        )
    return rows


def _list_values(value: object) -> list[object]:
    """Return a JSON list payload as a list, or an empty list."""
    return value if isinstance(value, list) else []


def _optional_float(value: object) -> float | None:
    """Convert a JSON scalar to float, preserving missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int:
    """Convert a JSON scalar to int, defaulting to zero."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int_or_none(value: object) -> int | None:
    """Convert a JSON scalar to int, preserving missing values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_review_speakers(
    segments_by_speaker: dict[int, list[SentenceSegment]],
    mapping: dict[int, str],
    ignored_speaker_ids: set[int],
    person_mapping: dict[int, int],
    person_public_mapping: dict[int, str],
    matches: dict[int, SpeakerMatchCandidate],
) -> list[ReviewSpeaker]:
    """Build mutable speaker review rows."""
    return [
        _review_speaker(
            speaker_id,
            segments,
            mapping,
            ignored_speaker_ids,
            person_mapping,
            person_public_mapping,
            matches,
        )
        for speaker_id, segments in sorted(segments_by_speaker.items())
    ]


def _review_speaker(
    speaker_id: int,
    segments: list[SentenceSegment],
    mapping: dict[int, str],
    ignored_speaker_ids: set[int],
    person_mapping: dict[int, int],
    person_public_mapping: dict[int, str],
    matches: dict[int, SpeakerMatchCandidate],
) -> ReviewSpeaker:
    """Build one speaker review row."""
    label = speaker_id_to_label(speaker_id)
    match = matches.get(speaker_id)
    current_name = mapping.get(speaker_id) or accepted_review_name(match) or label
    person_id = person_mapping.get(speaker_id) or accepted_review_person_id(match)
    person_public_id = person_public_mapping.get(speaker_id) or accepted_review_person_public_id(match)
    legacy_ignored = (
        speaker_id in mapping
        and current_name == label
        and (match is None or voiceprint_match_status(match) != MATCH_STATUS_BELOW_THRESHOLD)
    )
    ignored = speaker_id in ignored_speaker_ids or legacy_ignored
    if ignored:
        current_name = label
        person_id = None
        person_public_id = None
    return ReviewSpeaker(
        speaker_id,
        label,
        segments,
        current_name,
        match,
        ignored=ignored,
        person_id=person_id,
        person_public_id=person_public_id,
    )
