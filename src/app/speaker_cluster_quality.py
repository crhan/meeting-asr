"""Project speaker embedding cluster quality diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import load_manifest, project_paths, resolve_project_source_path
from app.speaker_labeling import load_transcript_result
from app.utils import safe_write_json
from app.voiceprint_embedding import embed_audio_file, resolve_voiceprint_embedding_options
from app.voiceprint_quality import DEFAULT_CRITICAL_SCORE, DEFAULT_WARNING_SCORE

MIN_ANCHOR_DURATION_MS = 1500
MIN_ANCHOR_TEXT_CHARS = 8
LOW_INFO_TEXT_CHARS = 4
TEXT_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)
FILLER_PHRASES = (
    "然后",
    "就是",
    "这个",
    "那个",
    "其实",
    "反正",
    "可能",
    "应该",
    "还是",
    "也是",
    "呃",
    "嗯",
    "啊",
    "哦",
    "好",
)


@dataclass(frozen=True, slots=True)
class SpeakerClusterClip:
    """One embedded probe clip from a project speaker."""

    speaker_id: int
    index: int
    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    text: str
    vector: list[float]


@dataclass(frozen=True, slots=True)
class SpeakerClusterReport:
    """Cluster quality report for one detected project speaker."""

    speaker_id: int
    label: str
    segment_count: int
    clip_count: int
    centroid_mean: float | None
    centroid_min: float | None
    warning_clip_count: int
    critical_clip_count: int
    intra_mean: float | None
    intra_min: float | None
    component_count: int
    component_sizes: list[int]
    nearest_speaker_id: int | None
    nearest_score: float | None
    status: str
    warnings: list[str]
    samples: list["SpeakerClusterSampleScore"]


@dataclass(frozen=True, slots=True)
class SpeakerClusterSampleScore:
    """One clip score against its speaker centroid."""

    index: int
    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    text: str
    centroid_score: float | None
    status: str


@dataclass(frozen=True, slots=True)
class SpeakerCentroidPair:
    """Cosine score between two project speaker centroids."""

    left_speaker_id: int
    right_speaker_id: int
    score: float


@dataclass(frozen=True, slots=True)
class SpeakerClusterQualitySummary:
    """Full project speaker cluster quality summary."""

    report_path: Path
    provider: str
    model: str
    same_speaker_threshold: float
    merge_speaker_threshold: float
    warning_score: float
    critical_score: float
    score_all_segments: bool
    reports: list[SpeakerClusterReport]
    close_pairs: list[SpeakerCentroidPair]
    verdict: str


def analyze_project_speaker_clusters(
    project_dir: Path,
    *,
    provider: str | None,
    model: str | None,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    score_all_segments: bool,
    same_speaker_threshold: float,
    merge_speaker_threshold: float,
    write_report: bool,
    warning_score: float = DEFAULT_WARNING_SCORE,
    critical_score: float = DEFAULT_CRITICAL_SCORE,
    progress: CliProgressReporter | None = None,
) -> SpeakerClusterQualitySummary:
    """
    Analyze speaker embedding cluster quality for one project.

    Args:
        project_dir: Project root directory.
        provider: Optional voiceprint embedding provider override.
        model: Optional model storage key override.
        sample_count: Maximum probe clips per speaker.
        max_seconds: Maximum clip duration.
        padding_seconds: Context padding around each segment.
        score_all_segments: Whether to score every transcript segment against the anchor centroid.
        same_speaker_threshold: Edge threshold for same-speaker components.
        merge_speaker_threshold: Speaker centroid score considered too close.
        warning_score: Clip-to-centroid score below which a clip is suspicious.
        critical_score: Clip-to-centroid score below which a clip is a strong outlier.
        write_report: Whether to write ``speakers/speaker_cluster_quality.json``.
        progress: Optional CLI progress reporter.

    Returns:
        Speaker cluster quality summary.
    """
    context = _load_cluster_context(project_dir, provider, model)
    clips_by_speaker = _embed_speaker_clips(
        context,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        score_all_segments=score_all_segments,
        progress=progress,
    )
    summary = _build_quality_summary(
        context,
        clips_by_speaker,
        score_all_segments=score_all_segments,
        same_speaker_threshold=same_speaker_threshold,
        merge_speaker_threshold=merge_speaker_threshold,
        warning_score=warning_score,
        critical_score=critical_score,
    )
    if write_report:
        emit_progress(progress, "Writing speaker cluster quality report")
        safe_write_json(summary.report_path, speaker_cluster_quality_payload(summary))
    return summary


def speaker_cluster_quality_payload(summary: SpeakerClusterQualitySummary) -> dict[str, object]:
    """Return a JSON-safe cluster quality summary payload."""
    return _summary_payload(summary)


@dataclass(frozen=True, slots=True)
class _ClusterContext:
    """Resolved project data needed for cluster diagnostics."""

    project_root: Path
    source: Path
    segments_by_speaker: dict[int, list[SentenceSegment]]
    provider: str
    model: str


def _load_cluster_context(project_dir: Path, provider: str | None, model: str | None) -> _ClusterContext:
    """Resolve project data and embedding options."""
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(provider=provider, model=model)
    paths = project_paths(project_dir)
    manifest = load_manifest(paths.root)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    return _ClusterContext(
        paths.root,
        resolve_project_source_path(paths.root, manifest),
        _segments_by_speaker(result.sentences),
        resolved_provider,
        resolved_model,
    )


def _segments_by_speaker(segments: list[SentenceSegment]) -> dict[int, list[SentenceSegment]]:
    """Group usable transcript segments by speaker id."""
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for segment in segments:
        if segment.speaker_id is None or not segment.text.strip():
            continue
        if segment.end_time_ms > segment.begin_time_ms:
            grouped[segment.speaker_id].append(segment)
    return dict(grouped)


def _embed_speaker_clips(
    context: _ClusterContext,
    *,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    score_all_segments: bool,
    progress: CliProgressReporter | None,
) -> dict[int, tuple[list[SpeakerClusterClip], list[SpeakerClusterClip]]]:
    """Extract and embed selected clips for each project speaker."""
    speaker_items = sorted(context.segments_by_speaker.items())
    emit_progress(progress, "Embedding speaker cluster probes", total=len(speaker_items), completed=0)
    clips_by_speaker: dict[int, tuple[list[SpeakerClusterClip], list[SpeakerClusterClip]]] = {}
    cache = _read_embedding_cache(context.project_root)
    for speaker_id, segments in speaker_items:
        selected = _select_segments(segments, sample_count)
        anchor_clips = _embed_selected_segments(
            context,
            speaker_id,
            selected,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            cache=cache,
        )
        score_segments = segments if score_all_segments else selected
        score_clips = anchor_clips
        if score_all_segments:
            score_clips = _embed_selected_segments(
                context,
                speaker_id,
                score_segments,
                max_seconds=max_seconds,
                padding_seconds=padding_seconds,
                cache=cache,
            )
        clips_by_speaker[speaker_id] = (anchor_clips, score_clips)
        emit_progress(progress, f"Embedded {speaker_id_to_label(speaker_id)} cluster probes", advance=1)
    _write_embedding_cache(context.project_root, cache)
    return clips_by_speaker


def _embed_selected_segments(
    context: _ClusterContext,
    speaker_id: int,
    segments: list[SentenceSegment],
    *,
    max_seconds: float,
    padding_seconds: float,
    cache: dict[str, list[float]],
) -> list[SpeakerClusterClip]:
    """Build embedded probe clips for selected segments."""
    clips: list[SpeakerClusterClip] = []
    for index, segment in enumerate(segments, start=1):
        key = _clip_cache_key(context, speaker_id, segment, max_seconds, padding_seconds)
        vector = cache.get(key)
        if vector is None:
            clip_path = _clip_path(context.project_root, speaker_id, index)
            _write_clip(context.source, clip_path, segment, max_seconds, padding_seconds)
            vector = _normalize(embed_audio_file(clip_path, provider=context.provider))
            cache[key] = vector
        clips.append(_cluster_clip(speaker_id, index, segment, vector))
    return clips


def _cluster_clip(
    speaker_id: int,
    index: int,
    segment: SentenceSegment,
    vector: list[float],
) -> SpeakerClusterClip:
    """Create one cluster clip row."""
    return SpeakerClusterClip(
        speaker_id,
        index,
        segment.sentence_id,
        segment.begin_time_ms,
        segment.end_time_ms,
        segment.text,
        vector,
    )


def _select_segments(segments: list[SentenceSegment], sample_count: int) -> list[SentenceSegment]:
    """Select high-information speaker anchors and restore timeline order."""
    ranked = sorted(segments, key=_segment_selection_score, reverse=True)
    selected = [segment for segment in ranked if not _is_low_information_segment(segment)][:sample_count]
    if len(selected) < min(sample_count, len(segments)):
        selected_keys = {_segment_identity(segment) for segment in selected}
        for segment in ranked:
            if _segment_identity(segment) in selected_keys:
                continue
            selected.append(segment)
            selected_keys.add(_segment_identity(segment))
            if len(selected) >= sample_count:
                break
    return sorted(selected, key=lambda item: item.begin_time_ms)


def _segment_selection_score(segment: SentenceSegment) -> tuple[float, int, int]:
    """Return a quality-first sort key for speaker anchor selection."""
    chars = _content_chars(segment.text)
    duration_ms = segment.end_time_ms - segment.begin_time_ms
    length_score = min(len(chars) / 24, 1.0)
    duration_score = min(duration_ms / 8000, 1.0)
    diversity_score = len(set(chars)) / len(chars) if chars else 0.0
    low_info_penalty = 1.0 if _is_low_information_segment(segment) else 0.0
    quality = length_score * 0.50 + duration_score * 0.35 + diversity_score * 0.15 - low_info_penalty
    return quality, duration_ms, len(chars)


def _is_low_information_segment(segment: SentenceSegment) -> bool:
    """Return whether a segment is too short to trust as a cluster anchor."""
    chars = _content_chars(segment.text)
    duration_ms = segment.end_time_ms - segment.begin_time_ms
    if len(chars) <= LOW_INFO_TEXT_CHARS:
        return True
    if len(chars) < MIN_ANCHOR_TEXT_CHARS and duration_ms < MIN_ANCHOR_DURATION_MS:
        return True
    return len(set(chars)) <= 2 and len(chars) < 16


def _content_chars(text: str) -> list[str]:
    """Return text characters that carry lexical information."""
    normalized = text.lower()
    for phrase in FILLER_PHRASES:
        normalized = normalized.replace(phrase, "")
    return TEXT_TOKEN_RE.findall(normalized)


def _segment_identity(segment: SentenceSegment) -> tuple[int | None, int, int]:
    """Return a stable identity for one transcript segment."""
    return (segment.sentence_id, segment.begin_time_ms, segment.end_time_ms)


def _clip_cache_key(
    context: _ClusterContext,
    speaker_id: int,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
) -> str:
    """Return a stable embedding cache key for one probe clip."""
    payload = {
        "version": 1,
        "provider": context.provider,
        "model": context.model,
        "speaker_id": speaker_id,
        "sentence_id": segment.sentence_id,
        "begin_time_ms": segment.begin_time_ms,
        "end_time_ms": segment.end_time_ms,
        "text": segment.text,
        "max_seconds": max_seconds,
        "padding_seconds": padding_seconds,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clip_path(project_root: Path, speaker_id: int, index: int) -> Path:
    """Return a deterministic cluster probe clip path."""
    return project_root / "tmp" / "speaker_cluster" / f"speaker_{speaker_id}" / f"clip_{index:03d}.wav"


def _write_clip(
    source: Path,
    output: Path,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
) -> None:
    """Extract one bounded probe clip from the project source media."""
    padding_ms = int(round(padding_seconds * 1000))
    max_ms = int(round(max_seconds * 1000))
    start_ms = max(0, segment.begin_time_ms - padding_ms)
    end_ms = min(segment.end_time_ms + padding_ms, start_ms + max_ms)
    extract_audio_clip(source, output, start_seconds=start_ms / 1000, duration_seconds=(end_ms - start_ms) / 1000)


def _cache_path(project_root: Path) -> Path:
    """Return the project-local cluster embedding cache path."""
    return project_root / "tmp" / "speaker_cluster" / "clip_embeddings.json"


def _read_embedding_cache(project_root: Path) -> dict[str, list[float]]:
    """Read valid cached cluster embeddings."""
    path = _cache_path(project_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _valid_embedding_cache(payload)


def _valid_embedding_cache(payload: object) -> dict[str, list[float]]:
    """Filter a JSON payload down to valid embedding vectors."""
    if not isinstance(payload, dict):
        return {}
    valid: dict[str, list[float]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            try:
                valid[str(key)] = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    return valid


def _write_embedding_cache(project_root: Path, cache: dict[str, list[float]]) -> None:
    """Persist cluster embedding cache."""
    safe_write_json(_cache_path(project_root), cache)


def _build_quality_summary(
    context: _ClusterContext,
    clips_by_speaker: dict[int, tuple[list[SpeakerClusterClip], list[SpeakerClusterClip]]],
    *,
    score_all_segments: bool,
    same_speaker_threshold: float,
    merge_speaker_threshold: float,
    warning_score: float,
    critical_score: float,
) -> SpeakerClusterQualitySummary:
    """Build the full cluster quality summary."""
    anchor_clips_by_speaker = {speaker_id: clips[0] for speaker_id, clips in clips_by_speaker.items()}
    centroids = _speaker_centroids(anchor_clips_by_speaker)
    close_pairs = _close_centroid_pairs(centroids, merge_speaker_threshold)
    reports = [
        _speaker_report(
            speaker_id,
            context.segments_by_speaker[speaker_id],
            anchor_clips,
            score_clips,
            centroids,
            same_speaker_threshold,
            warning_score,
            critical_score,
        )
        for speaker_id, (anchor_clips, score_clips) in sorted(clips_by_speaker.items())
    ]
    verdict = _summary_verdict(reports, close_pairs)
    return SpeakerClusterQualitySummary(
        context.project_root / "speakers" / "speaker_cluster_quality.json",
        context.provider,
        context.model,
        same_speaker_threshold,
        merge_speaker_threshold,
        warning_score,
        critical_score,
        score_all_segments,
        reports,
        close_pairs,
        verdict,
    )


def _speaker_centroids(clips_by_speaker: dict[int, list[SpeakerClusterClip]]) -> dict[int, list[float]]:
    """Return one normalized centroid per speaker with embedded clips."""
    return {
        speaker_id: _normalize(_mean_vector([clip.vector for clip in clips]))
        for speaker_id, clips in clips_by_speaker.items()
        if clips
    }


def _speaker_report(
    speaker_id: int,
    all_segments: list[SentenceSegment],
    anchor_clips: list[SpeakerClusterClip],
    score_clips: list[SpeakerClusterClip],
    centroids: dict[int, list[float]],
    same_speaker_threshold: float,
    warning_score: float,
    critical_score: float,
) -> SpeakerClusterReport:
    """Build one speaker quality report."""
    scores = _pairwise_scores([clip.vector for clip in anchor_clips])
    components = _connected_components(anchor_clips, same_speaker_threshold)
    nearest_id, nearest_score = _nearest_other_speaker(speaker_id, centroids)
    anchor_scores = _clip_centroid_score_rows(anchor_clips, centroids.get(speaker_id), warning_score, critical_score)
    sample_scores = _clip_centroid_score_rows(score_clips, centroids.get(speaker_id), warning_score, critical_score)
    centroid_scores = [score.centroid_score for score in anchor_scores if score.centroid_score is not None]
    warning_count = sum(1 for score in centroid_scores if critical_score <= score < warning_score)
    critical_count = sum(1 for score in centroid_scores if score < critical_score)
    warnings = _speaker_warnings(anchor_clips, scores, components, warning_count, critical_count)
    return SpeakerClusterReport(
        speaker_id,
        speaker_id_to_label(speaker_id),
        len(all_segments),
        len(anchor_clips),
        _mean(centroid_scores) if centroid_scores else None,
        min(centroid_scores) if centroid_scores else None,
        warning_count,
        critical_count,
        _mean(scores) if scores else None,
        min(scores) if scores else None,
        len(components),
        sorted((len(component) for component in components), reverse=True),
        nearest_id,
        nearest_score,
        _speaker_status(warnings),
        warnings,
        sample_scores,
    )


def _clip_centroid_score_rows(
    clips: list[SpeakerClusterClip],
    centroid: list[float] | None,
    warning_score: float,
    critical_score: float,
) -> list[SpeakerClusterSampleScore]:
    """Return each clip's score against its own speaker centroid."""
    rows: list[SpeakerClusterSampleScore] = []
    for clip in clips:
        score = None if centroid is None else _cosine(clip.vector, centroid)
        low_information = _is_low_information_clip(clip)
        rows.append(
            SpeakerClusterSampleScore(
                clip.index,
                clip.sentence_id,
                clip.begin_time_ms,
                clip.end_time_ms,
                clip.text,
                score,
                _sample_score_status(score, warning_score, critical_score, low_information=low_information),
            )
        )
    return rows


def _is_low_information_clip(clip: SpeakerClusterClip) -> bool:
    """Return whether a scored clip is too short for strong interpretation."""
    segment = SentenceSegment(clip.begin_time_ms, clip.end_time_ms, clip.text, clip.speaker_id, clip.sentence_id)
    return _is_low_information_segment(segment)


def _sample_score_status(
    score: float | None,
    warning_score: float,
    critical_score: float,
    *,
    low_information: bool,
) -> str:
    """Return a stable status for one sample-to-centroid score."""
    if low_information:
        return "low-info"
    if score is None:
        return "unknown"
    if score < critical_score:
        return "critical"
    if score < warning_score:
        return "warning"
    return "ok"


def _pairwise_scores(vectors: list[list[float]]) -> list[float]:
    """Return all unique pairwise cosine scores."""
    return [_cosine(left, right) for index, left in enumerate(vectors) for right in vectors[index + 1 :]]


def _connected_components(
    clips: list[SpeakerClusterClip],
    same_speaker_threshold: float,
) -> list[list[SpeakerClusterClip]]:
    """Cluster clips by connected cosine edges."""
    remaining = set(range(len(clips)))
    components: list[list[SpeakerClusterClip]] = []
    while remaining:
        current = remaining.pop()
        stack = [current]
        indices = [current]
        while stack:
            left = stack.pop()
            for right in list(remaining):
                if _cosine(clips[left].vector, clips[right].vector) >= same_speaker_threshold:
                    remaining.remove(right)
                    stack.append(right)
                    indices.append(right)
        components.append([clips[index] for index in indices])
    return components


def _nearest_other_speaker(
    speaker_id: int,
    centroids: dict[int, list[float]],
) -> tuple[int | None, float | None]:
    """Return nearest other speaker centroid."""
    own = centroids.get(speaker_id)
    if own is None:
        return None, None
    candidates = [(other_id, _cosine(own, vector)) for other_id, vector in centroids.items() if other_id != speaker_id]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def _speaker_warnings(
    clips: list[SpeakerClusterClip],
    scores: list[float],
    components: list[list[SpeakerClusterClip]],
    warning_count: int,
    critical_count: int,
) -> list[str]:
    """Return warning labels for one speaker cluster."""
    warnings: list[str] = []
    if len(clips) < 2:
        return ["too_few_clips"]
    if critical_count:
        warnings.append("critical_centroid_outlier")
    elif warning_count:
        warnings.append("warning_centroid_outlier")
    component_sizes = sorted((len(component) for component in components), reverse=True)
    if len(component_sizes) > 1 and component_sizes[1] >= 2:
        warnings.append("multi_component")
    if scores and _mean(scores) < 0.55:
        warnings.append("low_internal_mean")
    if scores and min(scores) < 0.25:
        warnings.append("very_low_internal_min")
    return warnings


def _speaker_status(warnings: list[str]) -> str:
    """Return a stable speaker status from warning labels."""
    if not warnings:
        return "ok"
    if warnings == ["too_few_clips"]:
        return "insufficient"
    if any(item in warnings for item in {"critical_centroid_outlier", "multi_component", "low_internal_mean"}):
        return "mixed"
    return "warning"


def _close_centroid_pairs(
    centroids: dict[int, list[float]],
    merge_speaker_threshold: float,
) -> list[SpeakerCentroidPair]:
    """Return speaker pairs whose centroids are suspiciously close."""
    pairs: list[SpeakerCentroidPair] = []
    ordered = sorted(centroids.items())
    for index, (left_id, left) in enumerate(ordered):
        for right_id, right in ordered[index + 1 :]:
            score = _cosine(left, right)
            if score >= merge_speaker_threshold:
                pairs.append(SpeakerCentroidPair(left_id, right_id, score))
    return sorted(pairs, key=lambda item: item.score, reverse=True)


def _summary_verdict(
    reports: list[SpeakerClusterReport],
    close_pairs: list[SpeakerCentroidPair],
) -> str:
    """Return a concise project-level verdict."""
    mixed = [report for report in reports if report.status == "mixed"]
    if mixed and close_pairs:
        return "unstable: mixed speaker clusters and close speaker centroids"
    if mixed:
        return "possible under-split: at least one speaker has multiple internal clusters"
    if close_pairs:
        return "possible over-split: some speaker centroids are very close"
    return "usable: speaker clusters look internally coherent"


def _summary_payload(summary: SpeakerClusterQualitySummary) -> dict[str, object]:
    """Return a JSON-safe summary payload."""
    return {
        "report_path": str(summary.report_path),
        "provider": summary.provider,
        "model": summary.model,
        "same_speaker_threshold": summary.same_speaker_threshold,
        "merge_speaker_threshold": summary.merge_speaker_threshold,
        "warning_score": summary.warning_score,
        "critical_score": summary.critical_score,
        "scoring_mode": "all-segments" if summary.score_all_segments else "sampled",
        "verdict": summary.verdict,
        "speakers": [_report_payload(report) for report in summary.reports],
        "close_pairs": [
            {"left_speaker_id": pair.left_speaker_id, "right_speaker_id": pair.right_speaker_id, "score": pair.score}
            for pair in summary.close_pairs
        ],
    }


def _report_payload(report: SpeakerClusterReport) -> dict[str, object]:
    """Return a JSON-safe speaker report payload."""
    return {
        "speaker_id": report.speaker_id,
        "label": report.label,
        "segment_count": report.segment_count,
        "clip_count": report.clip_count,
        "centroid_mean": report.centroid_mean,
        "centroid_min": report.centroid_min,
        "warning_clip_count": report.warning_clip_count,
        "critical_clip_count": report.critical_clip_count,
        "intra_mean": report.intra_mean,
        "intra_min": report.intra_min,
        "component_count": report.component_count,
        "component_sizes": report.component_sizes,
        "nearest_speaker_id": report.nearest_speaker_id,
        "nearest_score": report.nearest_score,
        "status": report.status,
        "warnings": report.warnings,
        "samples": [_sample_score_payload(sample) for sample in report.samples],
    }


def _sample_score_payload(sample: SpeakerClusterSampleScore) -> dict[str, object]:
    """Return a JSON-safe sample score payload."""
    return {
        "index": sample.index,
        "sentence_id": sample.sentence_id,
        "begin_time_ms": sample.begin_time_ms,
        "end_time_ms": sample.end_time_ms,
        "text": sample.text,
        "centroid_score": sample.centroid_score,
        "status": sample.status,
    }


def _mean(values: list[float]) -> float:
    """Return the mean of a non-empty float list."""
    return sum(values) / len(values)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return element-wise vector mean."""
    if not vectors:
        raise ValueError("Cannot average empty vectors.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding vectors must have the same dimension.")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]


def _normalize(vector: list[float]) -> list[float]:
    """Return a normalized embedding vector."""
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        raise ValueError("Embedding vector norm must not be zero.")
    return [item / norm for item in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two normalized vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    return sum(left[index] * right[index] for index in range(len(left)))
