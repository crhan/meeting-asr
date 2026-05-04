"""Match project speakers against the cross-project voiceprint library."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import (
    ProjectManifest,
    ensure_project_dirs,
    load_manifest,
    resolve_project_source_path,
    save_manifest,
)
from app.speaker_match_status import voiceprint_match_status
from app.speaker_labeling import load_transcript_result
from app.utils import safe_write_json
from app.voiceprint_embedding import embed_audio_file, resolve_voiceprint_embedding_options
from app.voiceprint_store import get_voiceprint_db_path, list_voiceprint_embeddings


@dataclass(frozen=True, slots=True)
class VoiceprintCandidate:
    """One ranked voiceprint candidate."""

    person_id: int
    name: str
    score: float
    person_public_id: str = ""


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
        return {item.speaker_id: item.name for item in self.matches if item.accepted and item.name}

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
class _KnownSpeakerVector:
    """Averaged voiceprint vector for one stable person id."""

    person_id: int
    name: str
    vector: list[float]
    person_public_id: str = ""


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
    endpoint: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    progress: CliProgressReporter | None = None,
) -> SpeakerMatchSummary:
    """
    Match project speakers against stored voiceprint embeddings.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        endpoint: Optional provider endpoint.
        model: Embedding model key.
        threshold: Minimum accepted cosine score.
        sample_count: Maximum probe clips per speaker.
        max_seconds: Maximum seconds per probe clip.
        padding_seconds: Context padding around each segment.
        progress: Optional progress reporter.

    Returns:
        Match summary.
    """
    context = _match_context(project_dir, store_dir, provider, model)
    matches = _match_speaker_groups(
        context.project_root,
        context.source,
        context.segments,
        context.known,
        context.provider,
        endpoint,
        threshold,
        sample_count,
        max_seconds,
        padding_seconds,
        progress,
    )
    emit_progress(progress, "Writing speaker match suggestions")
    match_path = context.project_root / "speakers" / "speaker_matches.json"
    safe_write_json(match_path, _matches_payload(context.provider, context.model, threshold, matches))
    context.manifest.speakers["matches"] = "speakers/speaker_matches.json"
    save_manifest(context.project_root, context.manifest)
    emit_progress(progress, "Speaker matching complete")
    return SpeakerMatchSummary(match_path, context.provider, context.model, threshold, matches)


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
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(provider=provider, model=model)
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    return _MatchContext(
        paths.root,
        manifest,
        resolve_project_source_path(paths.root, manifest),
        result.sentences,
        _known_speaker_vectors(store_dir, resolved_model),
        resolved_provider,
        resolved_model,
    )


def _known_speaker_vectors(store_dir: Path | None, model: str) -> dict[int, _KnownSpeakerVector]:
    """Load averaged known speaker vectors."""
    embeddings = list_voiceprint_embeddings(model, get_voiceprint_db_path(store_dir))
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    names: dict[int, str] = {}
    public_ids: dict[int, str] = {}
    for row in embeddings:
        grouped[row.speaker_id].append(row.vector)
        names[row.speaker_id] = row.speaker_name
        public_ids[row.speaker_id] = row.speaker_public_id
    return {
        person_id: _KnownSpeakerVector(
            person_id,
            names[person_id],
            _normalize(_mean_vector(vectors)),
            public_ids[person_id],
        )
        for person_id, vectors in grouped.items()
    }


def _match_speaker_groups(
    project_root: Path,
    source: Path,
    segments: list[SentenceSegment],
    known: dict[int, _KnownSpeakerVector],
    provider: str | None,
    endpoint: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    progress: CliProgressReporter | None,
) -> list[SpeakerMatch]:
    """Match all speakers in a project transcript."""
    matches: list[SpeakerMatch] = []
    speaker_groups = sorted(_segments_by_speaker(segments).items())
    emit_progress(progress, "Matching project speakers", total=len(speaker_groups), completed=0)
    if not known:
        emit_progress(progress, "No voiceprint embeddings found; writing review-only matches")
        return _unknown_speaker_matches(speaker_groups, threshold)
    for speaker_id, speaker_segments in speaker_groups:
        emit_progress(progress, f"Matching {speaker_id_to_label(speaker_id)}")
        vector = _probe_speaker_vector(
            project_root,
            source,
            speaker_id,
            speaker_segments,
            provider,
            endpoint,
            sample_count,
            max_seconds,
            padding_seconds,
        )
        candidates = _ranked_matches(vector, known, limit=3)
        best = candidates[0] if candidates else None
        accepted = best is not None and best.score >= threshold
        accepted_name = best.name if accepted and best is not None else None
        accepted_person_id = best.person_id if accepted and best is not None else None
        accepted_person_public_id = best.person_public_id if accepted and best is not None else None
        matches.append(
            SpeakerMatch(
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
                tuple(candidates),
            )
        )
        emit_progress(progress, f"Matched {speaker_id_to_label(speaker_id)}", advance=1)
    return matches


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


def _segments_by_speaker(segments: list[SentenceSegment]) -> dict[int, list[SentenceSegment]]:
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
    provider: str | None,
    endpoint: str | None,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
) -> list[float]:
    """Build an averaged probe vector for one project speaker."""
    vectors: list[list[float]] = []
    for index, segment in enumerate(_select_segments(segments, sample_count), start=1):
        clip_path = _probe_clip_path(project_root, speaker_id, index)
        _write_probe_clip(source, clip_path, segment, max_seconds, padding_seconds)
        vectors.append(embed_audio_file(clip_path, provider=provider, endpoint=endpoint))
    return _normalize(_mean_vector(vectors))


def _select_segments(segments: list[SentenceSegment], sample_count: int) -> list[SentenceSegment]:
    """Select longest segments in timeline order."""
    longest = sorted(segments, key=lambda item: item.end_time_ms - item.begin_time_ms, reverse=True)[:sample_count]
    return sorted(longest, key=lambda item: item.begin_time_ms)


def _probe_clip_path(project_root: Path, speaker_id: int, index: int) -> Path:
    """Return a deterministic temporary probe clip path."""
    return project_root / "tmp" / "voiceprint_match" / f"speaker_{speaker_id}" / f"clip_{index:03d}.wav"


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
    extract_audio_clip(source, output, start_seconds=start_ms / 1000, duration_seconds=(end_ms - start_ms) / 1000)


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
        VoiceprintCandidate(item.person_id, item.name, _cosine(vector, item.vector), item.person_public_id)
        for item in known.values()
    ]
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return element-wise mean vector."""
    if not vectors:
        raise ValueError("Cannot average empty vectors.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding vectors must have the same dimension.")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]


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


def _matches_payload(provider: str, model: str, threshold: float, matches: list[SpeakerMatch]) -> dict[str, object]:
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
                "status": voiceprint_match_status(item),
                "candidates": [
                    {
                        "person_id": candidate.person_id,
                        "person_public_id": candidate.person_public_id,
                        "name": candidate.name,
                        "score": candidate.score,
                    }
                    for candidate in item.candidates
                ],
                "sample_count": item.sample_count,
            }
            for item in matches
        ],
    }
