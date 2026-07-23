"""Match project speakers against the cross-project voiceprint library."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import (
    ProjectManifest,
    ensure_project_dirs,
    load_manifest,
    resolve_project_audio_path,
    save_manifest,
)
from app.speaker_crosstalk import (
    DEFAULT_CROSSTALK_CONCENTRATION_MARGIN,
    DEFAULT_CROSSTALK_MAX_SAMPLES,
    DEFAULT_CROSSTALK_SCORE_FLOOR,
    CrosstalkParams,
    is_crosstalk,
)
from app.speaker_clip_embeddings import (
    clip_embedding_cache_key,
    read_clip_embedding_cache,
    write_clip_embedding_cache,
)
from app.speaker_match_status import voiceprint_match_status
from app.speaker_pipeline_params import (
    STRONG_MARGIN_ACCEPT_MARGIN,
    STRONG_MARGIN_ACCEPT_SCORE,
)
from app.speaker_labeling import load_transcript_result
from app.utils import safe_write_json
from app.voiceprint_audio import (
    VOICEPRINT_AUDIO_PREPROCESS_VERSION,
    trim_embedding_audio_silence,
)
from app.voiceprint_embedding import (
    embed_audio_file,
    resolve_voiceprint_embedding_options,
)
from app.voiceprint_segment_selection import (
    ScoredVoiceprintSegment,
    select_voiceprint_segments,
)
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings


@dataclass(frozen=True, slots=True)
class VoiceprintCandidate:
    """One ranked voiceprint candidate."""

    person_id: int
    name: str
    score: float
    person_public_id: str = ""
    score_source: str = "person-centroid"
    sample_count: int = 0
    project_count: int = 0


@dataclass(frozen=True, slots=True)
class SpeakerMatch:
    """One speaker match result."""

    speaker_id: int
    label: str
    name: str | None
    score: float
    accepted: bool
    sample_count: int
    best_name: str | None = None
    best_score: float | None = None
    accepted_name: str | None = None
    threshold: float | None = None
    best_person_id: int | None = None
    best_person_public_id: str | None = None
    accepted_person_id: int | None = None
    accepted_person_public_id: str | None = None
    candidates: tuple[VoiceprintCandidate, ...] = ()
    crosstalk: bool = False
    margin_score: float | None = None
    accept_reason: str | None = None
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SpeakerMatchSummary:
    """Project speaker match summary."""

    match_path: Path
    provider: str
    model: str
    threshold: float
    matches: list[SpeakerMatch]

    @property
    def accepted_mapping(self) -> dict[int, str]:
        """Return accepted speaker id to name mapping."""
        return {
            item.speaker_id: item.name
            for item in self.matches
            if item.accepted and item.name
        }

    @property
    def accepted_person_mapping(self) -> dict[int, int]:
        """Return accepted speaker id to voiceprint person id mapping."""
        return {
            item.speaker_id: item.accepted_person_id
            for item in self.matches
            if item.accepted and item.accepted_person_id is not None
        }

    @property
    def accepted_person_public_mapping(self) -> dict[int, str]:
        """Return accepted speaker id to voiceprint person public id mapping."""
        return {
            item.speaker_id: item.accepted_person_public_id
            for item in self.matches
            if item.accepted and item.accepted_person_public_id
        }


@dataclass(frozen=True, slots=True)
class _KnownProjectVector:
    """Averaged voiceprint vector for one person's samples in one project."""

    project_id: str
    vector: list[float]
    sample_count: int


@dataclass(frozen=True, slots=True)
class _KnownSpeakerVector:
    """Averaged voiceprint vector for one stable person id."""

    person_id: int
    name: str
    vector: list[float]
    person_public_id: str = ""
    project_vectors: tuple[_KnownProjectVector, ...] = ()
    sample_count: int = 0
    project_count: int = 0


@dataclass(frozen=True, slots=True)
class _ProbeVector:
    """Averaged project-speaker probe vector plus sampling diagnostics."""

    vector: list[float]
    segments: tuple[ScoredVoiceprintSegment, ...]
    cached: bool = False


@dataclass(frozen=True, slots=True)
class _MatchContext:
    """Shared project and voiceprint context."""

    project_root: Path
    manifest: ProjectManifest
    source: Path
    segments: list[SentenceSegment]
    known: dict[int, _KnownSpeakerVector]
    provider: str
    model: str


def match_project_speakers(
    project_dir: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    crosstalk_params: CrosstalkParams | None = None,
    progress: CliProgressReporter | None = None,
) -> SpeakerMatchSummary:
    """
    Match project speakers against stored voiceprint embeddings.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        model: Embedding model key.
        threshold: Minimum accepted cosine score.
        sample_count: Maximum probe clips per speaker.
        max_seconds: Maximum seconds per probe clip.
        crosstalk_params: Crosstalk/noise tier thresholds (defaults: enabled).
        progress: Optional progress reporter.

    Returns:
        Match summary.
    """
    context = _match_context(project_dir, store_dir, provider, model)
    # The crosstalk tier is a project-level decision. Persist it when given
    # explicitly so later rematch callers (stabilization / resplit / review),
    # which call this function without crosstalk_params, honor the run's
    # --crosstalk/--no-crosstalk choice instead of silently re-enabling it.
    resolved_crosstalk = _resolve_crosstalk_params(context.manifest, crosstalk_params)
    if crosstalk_params is not None:
        context.manifest.speakers["crosstalk"] = _crosstalk_params_payload(
            crosstalk_params
        )
    summary = _build_match_summary(
        context,
        threshold=threshold,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        crosstalk_params=resolved_crosstalk,
        progress=progress,
    )
    emit_progress(progress, "Writing speaker match suggestions")
    safe_write_json(
        summary.match_path,
        _matches_payload(context.provider, context.model, threshold, summary.matches),
    )
    context.manifest.speakers["matches"] = "speakers/speaker_matches.json"
    save_manifest(context.project_root, context.manifest)
    emit_progress(progress, "Speaker matching complete")
    return summary


def _resolve_crosstalk_params(
    manifest: ProjectManifest, explicit: CrosstalkParams | None
) -> CrosstalkParams:
    """Return explicit params, else the project's persisted crosstalk setting."""
    if explicit is not None:
        return explicit
    stored = manifest.speakers.get("crosstalk")
    if isinstance(stored, dict):
        return CrosstalkParams(
            enabled=bool(stored.get("enabled", True)),
            max_samples=int(stored.get("max_samples", DEFAULT_CROSSTALK_MAX_SAMPLES)),
            score_floor=float(
                stored.get("score_floor", DEFAULT_CROSSTALK_SCORE_FLOOR)
            ),
            concentration_margin=float(
                stored.get(
                    "concentration_margin", DEFAULT_CROSSTALK_CONCENTRATION_MARGIN
                )
            ),
        )
    return CrosstalkParams()


def _crosstalk_params_payload(params: CrosstalkParams) -> dict[str, object]:
    """Serialize crosstalk params for the project manifest."""
    return {
        "enabled": params.enabled,
        "max_samples": params.max_samples,
        "score_floor": params.score_floor,
        "concentration_margin": params.concentration_margin,
    }


def preview_project_speaker_matches(
    project_dir: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    crosstalk_params: CrosstalkParams | None = None,
    progress: CliProgressReporter | None = None,
) -> SpeakerMatchSummary:
    """
    Compute speaker matches without writing project metadata.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        model: Embedding model key.
        threshold: Minimum accepted cosine score.
        sample_count: Maximum probe clips per speaker.
        max_seconds: Maximum seconds per probe clip.
        padding_seconds: Context padding around each segment.
        progress: Optional progress reporter.

    Returns:
        Match summary that has not been persisted to ``speaker_matches.json``.
    """
    context = _match_context(project_dir, store_dir, provider, model)
    resolved_crosstalk = _resolve_crosstalk_params(context.manifest, crosstalk_params)
    return _build_match_summary(
        context,
        threshold=threshold,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        crosstalk_params=resolved_crosstalk,
        progress=progress,
    )


def _build_match_summary(
    context: _MatchContext,
    *,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    crosstalk_params: CrosstalkParams | None = None,
    progress: CliProgressReporter | None,
) -> SpeakerMatchSummary:
    """Build a match summary from a resolved project context."""
    matches = _match_speaker_groups(
        context.project_root,
        context.source,
        context.segments,
        context.known,
        context.provider,
        context.model,
        threshold,
        sample_count,
        max_seconds,
        padding_seconds,
        progress,
    )
    matches = _flag_crosstalk_matches(matches, crosstalk_params)
    match_path = context.project_root / "speakers" / "speaker_matches.json"
    return SpeakerMatchSummary(
        match_path, context.provider, context.model, threshold, matches
    )


def _flag_crosstalk_matches(
    matches: list[SpeakerMatch], params: CrosstalkParams | None
) -> list[SpeakerMatch]:
    """Mark low-confidence crosstalk/noise clusters without moving any speaker."""
    resolved = params if params is not None else CrosstalkParams()
    if not resolved.enabled:
        return matches
    return [
        replace(match, crosstalk=True) if is_crosstalk(match, resolved) else match
        for match in matches
    ]


def _match_context(
    project_dir: Path,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
) -> _MatchContext:
    """
    Build shared project and voiceprint context.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        model: Optional embedding model key.

    Returns:
        Context for matching.
    """
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    return _MatchContext(
        paths.root,
        manifest,
        resolve_project_audio_path(paths.root, manifest),
        result.sentences,
        _known_speaker_vectors(store_dir, resolved_model),
        resolved_provider,
        resolved_model,
    )


def _known_speaker_vectors(
    store_dir: Path | None, model: str
) -> dict[int, _KnownSpeakerVector]:
    """Load averaged known speaker vectors."""
    embeddings = list_voiceprint_embeddings(model, get_voiceprint_db_path(store_dir))
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    grouped_by_project: dict[int, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    names: dict[int, str] = {}
    public_ids: dict[int, str] = {}
    for row in embeddings:
        grouped[row.speaker_id].append(row.vector)
        grouped_by_project[row.speaker_id][row.project_id].append(row.vector)
        names[row.speaker_id] = row.speaker_name
        public_ids[row.speaker_id] = row.speaker_public_id
    return {
        person_id: _KnownSpeakerVector(
            person_id,
            names[person_id],
            _normalize(_mean_vector(vectors)),
            public_ids[person_id],
            _known_project_vectors(grouped_by_project[person_id]),
            len(vectors),
            len(grouped_by_project[person_id]),
        )
        for person_id, vectors in grouped.items()
    }


def _known_project_vectors(
    grouped_by_project: dict[str, list[list[float]]],
) -> tuple[_KnownProjectVector, ...]:
    """Return per-project centroids for one voiceprint person."""
    return tuple(
        _KnownProjectVector(project_id, _normalize(_mean_vector(vectors)), len(vectors))
        for project_id, vectors in sorted(grouped_by_project.items())
        if vectors
    )


def _match_speaker_groups(
    project_root: Path,
    source: Path,
    segments: list[SentenceSegment],
    known: dict[int, _KnownSpeakerVector],
    provider: str | None,
    model: str,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    progress: CliProgressReporter | None,
) -> list[SpeakerMatch]:
    """Match all speakers in a project transcript."""
    speaker_groups = sorted(_segments_by_speaker(segments).items())
    emit_progress(
        progress, "Matching project speakers", total=len(speaker_groups), completed=0
    )
    if not known:
        emit_progress(
            progress, "No voiceprint embeddings found; writing review-only matches"
        )
        return _unknown_speaker_matches(speaker_groups, threshold)
    if len(speaker_groups) == 1:
        speaker_id, speaker_segments = speaker_groups[0]
        emit_progress(progress, f"Matching {speaker_id_to_label(speaker_id)}")
        match = _match_one_speaker_group(
            project_root,
            source,
            speaker_id,
            speaker_segments,
            segments,
            known,
            provider,
            model,
            threshold,
            sample_count,
            max_seconds,
            padding_seconds,
            Lock(),
        )
        emit_progress(progress, f"Matched {speaker_id_to_label(speaker_id)}", advance=1)
        return [match]

    cache_lock = Lock()
    workers = min(4, len(speaker_groups))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _match_one_speaker_group,
                project_root,
                source,
                speaker_id,
                speaker_segments,
                segments,
                known,
                provider,
                model,
                threshold,
                sample_count,
                max_seconds,
                padding_seconds,
                cache_lock,
            ): speaker_id
            for speaker_id, speaker_segments in speaker_groups
        }
        matched_by_id: dict[int, SpeakerMatch] = {}
        for future in as_completed(futures):
            speaker_id = futures[future]
            matched_by_id[speaker_id] = future.result()
            emit_progress(
                progress, f"Matched {speaker_id_to_label(speaker_id)}", advance=1
            )
    return [matched_by_id[speaker_id] for speaker_id, _segments in speaker_groups]


def _match_one_speaker_group(
    project_root: Path,
    source: Path,
    speaker_id: int,
    speaker_segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    known: dict[int, _KnownSpeakerVector],
    provider: str | None,
    model: str,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    cache_lock: Lock,
) -> SpeakerMatch:
    """Match one project speaker against known voiceprint vectors."""
    probe = _probe_speaker_vector(
        project_root,
        source,
        speaker_id,
        speaker_segments,
        all_segments,
        provider,
        model,
        sample_count,
        max_seconds,
        padding_seconds,
        cache_lock,
    )
    candidates = _ranked_matches(probe.vector, known, limit=3)
    best = candidates[0] if candidates else None
    accepted, accept_reason = _acceptance_decision(best, tuple(candidates), threshold)
    return _speaker_match_from_best(
        speaker_id,
        speaker_segments,
        best,
        tuple(candidates),
        accepted,
        threshold,
        accept_reason,
        diagnostics=_match_diagnostics(probe, tuple(candidates), accept_reason),
    )


def _acceptance_decision(
    best: VoiceprintCandidate | None,
    candidates: tuple[VoiceprintCandidate, ...],
    threshold: float,
) -> tuple[bool, str | None]:
    """Return whether to auto-accept a candidate and why."""
    if best is None:
        return False, None
    if best.score >= threshold:
        return True, "threshold"
    margin = _candidate_margin(candidates)
    if (
        margin is not None
        and best.score >= STRONG_MARGIN_ACCEPT_SCORE
        and margin >= STRONG_MARGIN_ACCEPT_MARGIN
    ):
        return True, "strong-margin"
    return False, None


def _candidate_margin(candidates: tuple[VoiceprintCandidate, ...]) -> float | None:
    """Return top-1 minus top-2 score when a competitor exists."""
    if len(candidates) < 2:
        return None
    return candidates[0].score - candidates[1].score


def _speaker_match_from_best(
    speaker_id: int,
    speaker_segments: list[SentenceSegment],
    best: VoiceprintCandidate | None,
    candidates: tuple[VoiceprintCandidate, ...],
    accepted: bool,
    threshold: float,
    accept_reason: str | None = None,
    diagnostics: dict[str, object] | None = None,
) -> SpeakerMatch:
    """Build the persisted match row from a best candidate."""
    accepted_name = best.name if accepted and best is not None else None
    accepted_person_id = best.person_id if accepted and best is not None else None
    accepted_person_public_id = (
        best.person_public_id if accepted and best is not None else None
    )
    return SpeakerMatch(
        speaker_id,
        speaker_id_to_label(speaker_id),
        accepted_name,
        best.score if best is not None else 0.0,
        accepted,
        len(speaker_segments),
        best.name if best is not None else None,
        best.score if best is not None else None,
        accepted_name,
        threshold,
        best.person_id if best is not None else None,
        best.person_public_id if best is not None else None,
        accepted_person_id,
        accepted_person_public_id,
        candidates,
        False,
        _candidate_margin(candidates),
        accept_reason if accepted else None,
        diagnostics,
    )


def _unknown_speaker_matches(
    speaker_groups: list[tuple[int, list[SentenceSegment]]],
    threshold: float,
) -> list[SpeakerMatch]:
    """
    Build review-only match rows when the voiceprint library is empty.

    Args:
        speaker_groups: Speaker id and transcript segments.
        threshold: Configured auto-accept threshold.

    Returns:
        Unknown, non-accepted match rows.
    """
    return [
        SpeakerMatch(
            speaker_id,
            speaker_id_to_label(speaker_id),
            None,
            0.0,
            False,
            len(speaker_segments),
            None,
            None,
            None,
            threshold,
            None,
            None,
            None,
            None,
            (),
        )
        for speaker_id, speaker_segments in speaker_groups
    ]


def _segments_by_speaker(
    segments: list[SentenceSegment],
) -> dict[int, list[SentenceSegment]]:
    """Group usable segments by speaker id."""
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for segment in segments:
        if segment.speaker_id is None or not segment.text.strip():
            continue
        if segment.end_time_ms > segment.begin_time_ms:
            grouped[segment.speaker_id].append(segment)
    return grouped


def _probe_speaker_vector(
    project_root: Path,
    source: Path,
    speaker_id: int,
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    provider: str | None,
    model: str,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    cache_lock: Lock,
) -> _ProbeVector:
    """Build an averaged probe vector for one project speaker."""
    selected_segments = _select_probe_segments(segments, all_segments, sample_count)
    cache_key = _probe_cache_key(
        speaker_id=speaker_id,
        segments=selected_segments,
        provider=provider,
        model=model,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
    )
    cached_vector = _read_probe_cache(project_root, cache_key)
    if cached_vector is not None:
        return _ProbeVector(cached_vector, tuple(selected_segments), cached=True)
    # Raw per-clip vectors are shared with cluster/sample diagnostics, so a
    # rematch after stabilization reuses their embeddings instead of paying
    # for the model again; averaging raw vectors keeps scores unchanged.
    clip_cache = read_clip_embedding_cache(project_root)
    new_clip_vectors: dict[str, list[float]] = {}
    vectors: list[list[float]] = []
    for index, candidate in enumerate(selected_segments, start=1):
        segment = candidate.segment
        clip_key = clip_embedding_cache_key(
            provider=provider,
            model=model,
            speaker_id=speaker_id,
            segment=segment,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
        )
        raw_vector = clip_cache.get(clip_key)
        if raw_vector is None:
            clip_path = _probe_clip_path(project_root, speaker_id, index)
            _write_probe_clip(source, clip_path, segment, max_seconds, padding_seconds)
            embedding_path = _probe_embedding_clip_path(
                project_root, speaker_id, index
            )
            trim_embedding_audio_silence(clip_path, embedding_path)
            raw_vector = embed_audio_file(embedding_path, provider=provider)
            new_clip_vectors[clip_key] = raw_vector
        vectors.append(raw_vector)
    vector = _normalize(_mean_vector(vectors))
    with cache_lock:
        if new_clip_vectors:
            write_clip_embedding_cache(project_root, new_clip_vectors)
        _write_probe_cache(project_root, cache_key, vector)
    return _ProbeVector(vector, tuple(selected_segments))


def _probe_cache_key(
    *,
    speaker_id: int,
    segments: list[ScoredVoiceprintSegment],
    provider: str | None,
    model: str,
    max_seconds: float,
    padding_seconds: float,
) -> str:
    """Return a stable cache key for one project speaker probe embedding."""
    payload = {
        "version": 4,
        "speaker_id": speaker_id,
        "provider": provider,
        "model": model,
        "audio_preprocess": VOICEPRINT_AUDIO_PREPROCESS_VERSION,
        "max_seconds": max_seconds,
        "padding_seconds": padding_seconds,
        "segments": [
            {
                "sentence_id": item.segment.sentence_id,
                "begin_time_ms": item.segment.begin_time_ms,
                "end_time_ms": item.segment.end_time_ms,
                "text": item.segment.text,
                "selection_score": item.score,
                "selection_reason": item.reason,
            }
            for item in segments
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _probe_cache_path(project_root: Path) -> Path:
    """Return the project-local probe embedding cache path."""
    return project_root / "tmp" / "voiceprint_match" / "probe_embeddings.json"


def _read_probe_cache(project_root: Path, cache_key: str) -> list[float] | None:
    """Read one cached probe embedding vector."""
    cache_path = _probe_cache_path(project_root)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    vector = payload.get(cache_key) if isinstance(payload, dict) else None
    if not isinstance(vector, list) or not vector:
        return None
    try:
        return [float(item) for item in vector]
    except TypeError, ValueError:
        return None


def _write_probe_cache(project_root: Path, cache_key: str, vector: list[float]) -> None:
    """Write one cached probe embedding vector."""
    cache_path = _probe_cache_path(project_root)
    payload: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            payload = _valid_probe_cache_payload(loaded)
    payload[cache_key] = vector
    safe_write_json(cache_path, payload)


def _valid_probe_cache_payload(payload: object) -> dict[str, list[float]]:
    """Return only valid cached vectors from a loaded JSON payload."""
    if not isinstance(payload, dict):
        return {}
    valid: dict[str, list[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            continue
        try:
            valid[str(key)] = [float(item) for item in value]
        except TypeError, ValueError:
            continue
    return valid


def _select_probe_segments(
    segments: list[SentenceSegment],
    all_segments: list[SentenceSegment],
    sample_count: int,
) -> list[ScoredVoiceprintSegment]:
    """Select quality-scored, time-spread probe segments."""
    candidates = select_voiceprint_segments(
        segments,
        all_segments,
        sample_count,
        candidate_count=max(sample_count * 4, sample_count, 12),
    )
    selected = [item for item in candidates if item.recommended]
    if len(selected) < sample_count:
        selected.extend(item for item in candidates if not item.recommended)
    if selected:
        return selected[:sample_count]
    return _select_longest_segments(segments, sample_count)


def _select_longest_segments(
    segments: list[SentenceSegment], sample_count: int
) -> list[ScoredVoiceprintSegment]:
    """Select longest segments in timeline order as a compatibility fallback."""
    longest = sorted(
        segments, key=lambda item: item.end_time_ms - item.begin_time_ms, reverse=True
    )[:sample_count]
    return [
        ScoredVoiceprintSegment(segment, 0.0, "fallback-longest", recommended=True)
        for segment in sorted(longest, key=lambda item: item.begin_time_ms)
    ]


def _probe_clip_path(project_root: Path, speaker_id: int, index: int) -> Path:
    """Return a deterministic temporary probe clip path."""
    return (
        project_root
        / "tmp"
        / "voiceprint_match"
        / f"speaker_{speaker_id}"
        / f"clip_{index:03d}.wav"
    )


def _probe_embedding_clip_path(project_root: Path, speaker_id: int, index: int) -> Path:
    """Return the preprocessed probe clip path used for embedding."""
    return (
        project_root
        / "tmp"
        / "voiceprint_match"
        / f"speaker_{speaker_id}"
        / f"clip_{index:03d}_embedding.wav"
    )


def _write_probe_clip(
    source: Path,
    output: Path,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
) -> None:
    """Extract one probe clip."""
    padding_ms = int(round(padding_seconds * 1000))
    max_ms = int(round(max_seconds * 1000))
    start_ms = max(0, segment.begin_time_ms - padding_ms)
    end_ms = min(segment.end_time_ms + padding_ms, start_ms + max_ms)
    extract_audio_clip(
        source,
        output,
        start_seconds=start_ms / 1000,
        duration_seconds=(end_ms - start_ms) / 1000,
    )


def _ranked_matches(
    vector: list[float],
    known: dict[int, _KnownSpeakerVector],
    *,
    limit: int,
) -> list[VoiceprintCandidate]:
    """
    Return ranked matching known speakers.

    Args:
        vector: Probe speaker embedding.
        known: Known speaker id to embedding mapping.
        limit: Maximum candidate count.

    Returns:
        Candidates sorted by descending cosine score.
    """
    candidates = [
        _candidate_from_known_vector(vector, item)
        for item in known.values()
    ]
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def _match_diagnostics(
    probe: _ProbeVector,
    candidates: tuple[VoiceprintCandidate, ...],
    accept_reason: str | None,
) -> dict[str, object]:
    """Build persisted diagnostics for one speaker match."""
    best = candidates[0] if candidates else None
    return {
        "probe_cached": probe.cached,
        "probe_sample_count": len(probe.segments),
        "probe_segments": [_probe_segment_payload(item) for item in probe.segments],
        "candidate_count": len(candidates),
        "margin_score": _candidate_margin(candidates),
        "accept_reason": accept_reason,
        "best_score_source": best.score_source if best is not None else None,
        "best_sample_count": best.sample_count if best is not None else None,
        "best_project_count": best.project_count if best is not None else None,
    }


def _probe_segment_payload(segment: ScoredVoiceprintSegment) -> dict[str, object]:
    """Serialize one selected probe segment for diagnostics."""
    item = segment.segment
    return {
        "sentence_id": item.sentence_id,
        "begin_time_ms": item.begin_time_ms,
        "end_time_ms": item.end_time_ms,
        "duration_seconds": round((item.end_time_ms - item.begin_time_ms) / 1000, 3),
        "selection_score": segment.score,
        "selection_reason": segment.reason,
        "recommended": segment.recommended,
        "text": item.text.strip(),
    }


def _candidate_from_known_vector(
    vector: list[float], known: _KnownSpeakerVector
) -> VoiceprintCandidate:
    """Build a ranked candidate using robust person/project centroid scoring."""
    score, score_source = _score_known_vector(vector, known)
    return VoiceprintCandidate(
        known.person_id,
        known.name,
        score,
        known.person_public_id,
        score_source,
        known.sample_count,
        known.project_count,
    )


def _score_known_vector(
    vector: list[float], known: _KnownSpeakerVector
) -> tuple[float, str]:
    """Score a known person by whole-person centroid plus stable project centroids."""
    person_score = _cosine(vector, known.vector)
    project_scores = [
        _cosine(vector, project.vector)
        for project in known.project_vectors
        if project.sample_count >= 2
    ]
    if not project_scores:
        return person_score, "person-centroid"
    project_score = max(project_scores)
    if project_score > person_score:
        return project_score, "project-centroid"
    return person_score, "person-centroid"


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return element-wise mean vector."""
    if not vectors:
        raise ValueError("Cannot average empty vectors.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding vectors must have the same dimension.")
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]


def _normalize(vector: list[float]) -> list[float]:
    """Return an L2-normalized vector."""
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        raise ValueError("Embedding vector norm must not be zero.")
    return [item / norm for item in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    return sum(left[index] * right[index] for index in range(len(left)))


def _matches_payload(
    provider: str, model: str, threshold: float, matches: list[SpeakerMatch]
) -> dict[str, object]:
    """Build JSON payload for match results."""
    return {
        "provider": provider,
        "model": model,
        "threshold": threshold,
        "matches": [
            {
                "speaker_id": item.speaker_id,
                "label": item.label,
                "name": item.name,
                "score": item.score,
                "accepted": item.accepted,
                "best_name": item.best_name,
                "best_score": item.best_score,
                "accepted_name": item.accepted_name,
                "best_person_id": item.best_person_id,
                "best_person_public_id": item.best_person_public_id,
                "accepted_person_id": item.accepted_person_id,
                "accepted_person_public_id": item.accepted_person_public_id,
                "threshold": item.threshold,
                "crosstalk": item.crosstalk,
                "margin_score": item.margin_score,
                "accept_reason": item.accept_reason,
                "status": voiceprint_match_status(item),
                "candidates": [
                    {
                        "person_id": candidate.person_id,
                        "person_public_id": candidate.person_public_id,
                        "name": candidate.name,
                        "score": candidate.score,
                        "score_source": candidate.score_source,
                        "sample_count": candidate.sample_count,
                        "project_count": candidate.project_count,
                    }
                    for candidate in item.candidates
                ],
                "sample_count": item.sample_count,
                "diagnostics": item.diagnostics or {},
            }
            for item in matches
        ],
    }
