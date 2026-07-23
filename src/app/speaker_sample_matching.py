"""Per-sentence voiceprint identity diagnostics for project speakers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.core.progress import CliProgressReporter, emit_progress
from app.infra.ffmpeg import extract_audio_clip
from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label
from app.project_manager import (
    ensure_project_dirs,
    load_manifest,
    resolve_project_audio_path,
    save_manifest,
)
from app.speaker_clip_embeddings import (
    clip_embedding_cache_key,
    read_clip_embedding_cache,
    write_clip_embedding_cache,
)
from app.speaker_labeling import load_speaker_person_mapping, load_transcript_result
from app.speaker_matching import (
    VoiceprintCandidate,
    _KnownSpeakerVector,
    _cosine,
    _known_speaker_vectors,
    _normalize,
    _ranked_matches,
)
from app.speaker_pipeline_params import (
    DEFAULT_FOREIGN_REASSIGN_THRESHOLD,
    DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
    DEFAULT_IDENTITY_CONFLICT_MARGIN,
    DEFAULT_SAMPLE_IDENTITY_THRESHOLD as DEFAULT_SAMPLE_IDENTITY_THRESHOLD,
)
from app.utils import safe_write_json
from app.voiceprint_audio import (
    trim_embedding_audio_silence,
)
from app.voiceprint_embedding import (
    embed_audio_file,
    ensure_library_embeddings,
    resolve_voiceprint_embedding_options,
)

LOW_INFO_TEXT_CHARS = 4
MIN_SAMPLE_TEXT_CHARS = 8
MIN_SAMPLE_DURATION_MS = 1200


@dataclass(frozen=True, slots=True)
class SpeakerSampleMatch:
    """One transcript sample matched against known voiceprint identities."""

    speaker_id: int
    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    text: str
    assigned_person_id: int | None
    assigned_name: str | None
    assigned_score: float | None
    best_person_id: int | None
    best_name: str | None
    best_score: float | None
    best_other_person_id: int | None
    best_other_name: str | None
    best_other_score: float | None
    margin_score: float | None
    status: str
    candidates: tuple[VoiceprintCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeakerSampleMatchReport:
    """Aggregated sample identity diagnostics for one project speaker."""

    speaker_id: int
    label: str
    assigned_person_id: int | None
    assigned_name: str | None
    sample_count: int
    status_counts: dict[str, int]
    samples: list[SpeakerSampleMatch]


@dataclass(frozen=True, slots=True)
class SpeakerSampleMatchSummary:
    """Full per-sample identity matching summary."""

    report_path: Path
    provider: str
    model: str
    threshold: float
    conflict_margin: float
    ambiguous_margin: float
    reports: list[SpeakerSampleMatchReport]
    verdict: str


@dataclass(frozen=True, slots=True)
class _SampleMatchContext:
    """Resolved data needed for sample identity matching."""

    project_root: Path
    source: Path
    segments_by_speaker: dict[int, list[SentenceSegment]]
    known: dict[int, _KnownSpeakerVector]
    assigned_person_by_speaker: dict[int, int]
    provider: str
    model: str


def match_project_speaker_samples(
    project_dir: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
    threshold: float,
    conflict_margin: float = DEFAULT_IDENTITY_CONFLICT_MARGIN,
    ambiguous_margin: float = DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
    foreign_threshold: float = DEFAULT_FOREIGN_REASSIGN_THRESHOLD,
    max_seconds: float,
    padding_seconds: float,
    write_report: bool,
    workers: int = 1,
    progress: CliProgressReporter | None = None,
) -> SpeakerSampleMatchSummary:
    """
    Match every usable transcript sample against known voiceprint identities.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        provider: Optional embedding provider.
        model: Optional embedding model.
        threshold: Minimum per-sample score treated as identity evidence.
        conflict_margin: Required other-person lead for a conflict.
        ambiguous_margin: Boundary below which identity is ambiguous.
        foreign_threshold: Minimum top-match score for a sentence in an
            unassigned cluster to be flagged as belonging to a known person.
        max_seconds: Maximum clip duration per transcript sample.
        padding_seconds: Audio context padding around each sample.
        write_report: Whether to write ``speaker_sample_matches.json``.
        workers: Maximum parallel sample embedding workers.
        progress: Optional CLI progress reporter.

    Returns:
        Per-sample identity match summary.
    """
    context = _sample_match_context(project_dir, store_dir, provider, model)
    reports = _match_sample_groups(
        context,
        threshold=threshold,
        conflict_margin=conflict_margin,
        ambiguous_margin=ambiguous_margin,
        foreign_threshold=foreign_threshold,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        workers=workers,
        progress=progress,
    )
    summary = SpeakerSampleMatchSummary(
        context.project_root / "speakers" / "speaker_sample_matches.json",
        context.provider,
        context.model,
        threshold,
        conflict_margin,
        ambiguous_margin,
        reports,
        _summary_verdict(reports),
    )
    if write_report:
        emit_progress(progress, "Writing speaker sample match diagnostics")
        safe_write_json(summary.report_path, speaker_sample_match_payload(summary))
        manifest = load_manifest(context.project_root)
        manifest.speakers["sample_matches"] = "speakers/speaker_sample_matches.json"
        save_manifest(context.project_root, manifest)
    return summary


def speaker_sample_match_payload(
    summary: SpeakerSampleMatchSummary,
) -> dict[str, object]:
    """Return a JSON-safe sample match payload."""
    return {
        "report_path": str(summary.report_path),
        "provider": summary.provider,
        "model": summary.model,
        "threshold": summary.threshold,
        "conflict_margin": summary.conflict_margin,
        "ambiguous_margin": summary.ambiguous_margin,
        "verdict": summary.verdict,
        "speakers": [_report_payload(report) for report in summary.reports],
    }


def _sample_match_context(
    project_dir: Path,
    store_dir: Path | None,
    provider: str | None,
    model: str | None,
) -> _SampleMatchContext:
    """Resolve project, transcript, voiceprint, and assignment inputs."""
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    # Same backfill as _match_context: sample matching runs standalone too, and
    # a library embedded only under another model key must not read as empty.
    ensure_library_embeddings(
        store_dir=store_dir, provider=resolved_provider, model=resolved_model
    )
    paths = ensure_project_dirs(project_dir)
    manifest = load_manifest(paths.root)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    known = _known_speaker_vectors(store_dir, resolved_model)
    return _SampleMatchContext(
        paths.root,
        resolve_project_audio_path(paths.root, manifest),
        _segments_by_speaker(result.sentences),
        known,
        _assigned_person_map(paths.root, known),
        resolved_provider,
        resolved_model,
    )


def _segments_by_speaker(
    segments: list[SentenceSegment],
) -> dict[int, list[SentenceSegment]]:
    """Group usable transcript segments by speaker id."""
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for segment in segments:
        if segment.speaker_id is None or not segment.text.strip():
            continue
        if segment.end_time_ms > segment.begin_time_ms:
            grouped[segment.speaker_id].append(segment)
    return dict(grouped)


def _assigned_person_map(
    project_root: Path, known: dict[int, _KnownSpeakerVector]
) -> dict[int, int]:
    """Resolve project speaker ids to known voiceprint person ids."""
    assigned: dict[int, int] = {}
    assigned.update(
        _speaker_person_map(
            project_root / "speakers" / "speaker_person_map.json", known
        )
    )
    for speaker_id, person_id in _accepted_match_map(
        project_root / "speakers" / "speaker_matches.json", known
    ).items():
        assigned.setdefault(speaker_id, person_id)
    for speaker_id, person_id in _speaker_name_map(
        project_root / "speakers" / "speaker_map.json", known
    ).items():
        assigned.setdefault(speaker_id, person_id)
    return assigned


def _speaker_person_map(
    path: Path, known: dict[int, _KnownSpeakerVector]
) -> dict[int, int]:
    """Load explicit speaker-to-person mappings."""
    raw = load_speaker_person_mapping(path)
    return {
        speaker_id: person_id
        for speaker_id, value in raw.items()
        if (person_id := _resolve_person_ref(value, known)) is not None
    }


def _accepted_match_map(
    path: Path, known: dict[int, _KnownSpeakerVector]
) -> dict[int, int]:
    """Load accepted speaker matches as identity assignments."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    rows = payload.get("matches") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    assigned: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("accepted"):
            continue
        try:
            speaker_id = int(row["speaker_id"])
        except KeyError, TypeError, ValueError:
            continue
        person_id = _resolve_person_ref(
            row.get("accepted_person_public_id") or row.get("accepted_person_id"), known
        )
        if person_id is not None:
            assigned[speaker_id] = person_id
    return assigned


def _speaker_name_map(
    path: Path, known: dict[int, _KnownSpeakerVector]
) -> dict[int, int]:
    """Resolve named project speakers by exact voiceprint person name."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    by_name = {_normalize_name(item.name): item.person_id for item in known.values()}
    assigned: dict[int, int] = {}
    for key, value in payload.items():
        try:
            speaker_id = int(key)
        except TypeError, ValueError:
            continue
        person_id = by_name.get(_normalize_name(str(value)))
        if person_id is not None:
            assigned[speaker_id] = person_id
    return assigned


def _resolve_person_ref(
    value: object, known: dict[int, _KnownSpeakerVector]
) -> int | None:
    """Resolve an internal or public person id against known vectors."""
    if isinstance(value, str):
        stripped = value.strip()
        for person_id, person in known.items():
            if person.person_public_id == stripped:
                return person_id
    try:
        person_id = int(value)
    except TypeError, ValueError:
        return None
    return person_id if person_id in known else None


def _normalize_name(value: str) -> str:
    """Normalize person names for exact assignment fallback."""
    return " ".join(value.strip().split()).casefold()


def _match_sample_groups(
    context: _SampleMatchContext,
    *,
    threshold: float,
    conflict_margin: float,
    ambiguous_margin: float,
    foreign_threshold: float,
    max_seconds: float,
    padding_seconds: float,
    workers: int,
    progress: CliProgressReporter | None,
) -> list[SpeakerSampleMatchReport]:
    """Match all project sample groups."""
    speaker_items = sorted(context.segments_by_speaker.items())
    total = sum(len(segments) for _speaker_id, segments in speaker_items)
    emit_progress(progress, "Matching speaker samples", total=total, completed=0)
    cache = _read_sample_cache(context.project_root)
    if workers > 1 and total > 1:
        reports = _match_sample_groups_parallel(
            context,
            speaker_items,
            threshold=threshold,
            conflict_margin=conflict_margin,
            ambiguous_margin=ambiguous_margin,
            foreign_threshold=foreign_threshold,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            workers=workers,
            cache=cache,
            progress=progress,
        )
        _write_sample_cache(context.project_root, cache)
        return reports
    reports = [
        _match_one_speaker_samples(
            context,
            speaker_id,
            segments,
            threshold=threshold,
            conflict_margin=conflict_margin,
            ambiguous_margin=ambiguous_margin,
            foreign_threshold=foreign_threshold,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            cache=cache,
            cache_lock=Lock(),
            progress=progress,
        )
        for speaker_id, segments in speaker_items
    ]
    _write_sample_cache(context.project_root, cache)
    return reports


def _match_sample_groups_parallel(
    context: _SampleMatchContext,
    speaker_items: list[tuple[int, list[SentenceSegment]]],
    *,
    threshold: float,
    conflict_margin: float,
    ambiguous_margin: float,
    foreign_threshold: float,
    max_seconds: float,
    padding_seconds: float,
    workers: int,
    cache: dict[str, list[float]],
    progress: CliProgressReporter | None,
) -> list[SpeakerSampleMatchReport]:
    """Match transcript samples concurrently while preserving speaker order."""
    cache_lock = Lock()
    rows_by_speaker: dict[int, list[tuple[int, SpeakerSampleMatch]]] = defaultdict(list)
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for speaker_id, segments in speaker_items:
            assigned_person_id = context.assigned_person_by_speaker.get(speaker_id)
            assigned = (
                context.known.get(assigned_person_id)
                if assigned_person_id is not None
                else None
            )
            for index, segment in enumerate(segments):
                future = executor.submit(
                    _match_one_sample,
                    context,
                    speaker_id,
                    segment,
                    assigned,
                    threshold=threshold,
                    conflict_margin=conflict_margin,
                    ambiguous_margin=ambiguous_margin,
                    foreign_threshold=foreign_threshold,
                    max_seconds=max_seconds,
                    padding_seconds=padding_seconds,
                    cache=cache,
                    cache_lock=cache_lock,
                )
                futures[future] = (speaker_id, index)
        for future in as_completed(futures):
            speaker_id, index = futures[future]
            rows_by_speaker[speaker_id].append((index, future.result()))
            emit_progress(
                progress, f"Matched {speaker_id_to_label(speaker_id)} sample", advance=1
            )
    reports: list[SpeakerSampleMatchReport] = []
    for speaker_id, segments in speaker_items:
        assigned_person_id = context.assigned_person_by_speaker.get(speaker_id)
        assigned = (
            context.known.get(assigned_person_id)
            if assigned_person_id is not None
            else None
        )
        rows = [
            row
            for _index, row in sorted(
                rows_by_speaker[speaker_id], key=lambda item: item[0]
            )
        ]
        reports.append(
            SpeakerSampleMatchReport(
                speaker_id,
                speaker_id_to_label(speaker_id),
                None if assigned is None else assigned.person_id,
                None if assigned is None else assigned.name,
                len(rows),
                dict(Counter(row.status for row in rows)),
                rows,
            )
        )
    return reports


def _match_one_speaker_samples(
    context: _SampleMatchContext,
    speaker_id: int,
    segments: list[SentenceSegment],
    *,
    threshold: float,
    conflict_margin: float,
    ambiguous_margin: float,
    foreign_threshold: float,
    max_seconds: float,
    padding_seconds: float,
    cache: dict[str, list[float]],
    cache_lock: Lock,
    progress: CliProgressReporter | None,
) -> SpeakerSampleMatchReport:
    """Match all samples for one project speaker."""
    assigned_person_id = context.assigned_person_by_speaker.get(speaker_id)
    assigned = (
        context.known.get(assigned_person_id)
        if assigned_person_id is not None
        else None
    )
    rows: list[SpeakerSampleMatch] = []
    for segment in segments:
        row = _match_one_sample(
            context,
            speaker_id,
            segment,
            assigned,
            threshold=threshold,
            conflict_margin=conflict_margin,
            ambiguous_margin=ambiguous_margin,
            foreign_threshold=foreign_threshold,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            cache=cache,
            cache_lock=cache_lock,
        )
        rows.append(row)
        emit_progress(
            progress, f"Matched {speaker_id_to_label(speaker_id)} sample", advance=1
        )
    return SpeakerSampleMatchReport(
        speaker_id,
        speaker_id_to_label(speaker_id),
        None if assigned is None else assigned.person_id,
        None if assigned is None else assigned.name,
        len(rows),
        dict(Counter(row.status for row in rows)),
        rows,
    )


def _match_one_sample(
    context: _SampleMatchContext,
    speaker_id: int,
    segment: SentenceSegment,
    assigned: _KnownSpeakerVector | None,
    *,
    threshold: float,
    conflict_margin: float,
    ambiguous_margin: float,
    foreign_threshold: float,
    max_seconds: float,
    padding_seconds: float,
    cache: dict[str, list[float]],
    cache_lock: Lock,
) -> SpeakerSampleMatch:
    """Match one transcript sample against assigned and best-known identities."""
    if _is_low_information_segment(segment):
        return _sample_row(
            speaker_id, segment, assigned, None, None, None, "low-info", ()
        )
    vector = _sample_vector(
        context, speaker_id, segment, max_seconds, padding_seconds, cache, cache_lock
    )
    candidates = tuple(_ranked_matches(vector, context.known, limit=3))
    if assigned is None:
        return _unassigned_sample_row(
            speaker_id,
            segment,
            candidates,
            accepted_person_ids=frozenset(context.assigned_person_by_speaker.values()),
            conflict_margin=conflict_margin,
            foreign_threshold=foreign_threshold,
        )
    assigned_score = _cosine(vector, assigned.vector)
    best = candidates[0] if candidates else None
    best_other = next(
        (
            candidate
            for candidate in candidates
            if candidate.person_id != assigned.person_id
        ),
        None,
    )
    margin_score = None if best_other is None else assigned_score - best_other.score
    status = _identity_status(
        assigned_score,
        None if best_other is None else best_other.score,
        margin_score,
        threshold,
        conflict_margin,
        ambiguous_margin,
    )
    return _sample_row(
        speaker_id,
        segment,
        assigned,
        assigned_score,
        best,
        best_other,
        status,
        candidates,
    )


def _unassigned_sample_row(
    speaker_id: int,
    segment: SentenceSegment,
    candidates: tuple[VoiceprintCandidate, ...],
    *,
    accepted_person_ids: frozenset[int],
    conflict_margin: float,
    foreign_threshold: float,
) -> SpeakerSampleMatch:
    """Classify a sample whose speaker cluster has no confirmed identity.

    A below-threshold (unnamed) cluster has no assigned baseline, so the
    assigned-vs-other conflict test does not apply. We instead inspect the
    sentence's own best library match and only act when ALL of the following
    hold, so we never disturb sentences that genuinely belong to the cluster:

    - The top match is one of the meeting's **already-confirmed** speakers
      (``accepted_person_ids``). The unnamed cluster's own owner is, by
      definition, not confirmed, so its sentences (which match itself) are
      never flagged — this is what stops a whole cluster from lighting up.
    - The top match is strong (``>= foreign_threshold``).
    - The top match is a clear winner over the runner-up
      (``>= conflict_margin``), so spurious near-ties are left alone.

    Such a sentence is surfaced as ``identity-foreign`` for stabilization to
    reassign to that confirmed speaker; everything else stays ``no-assignment``
    and remains in the original cluster.

    Args:
        speaker_id: Project speaker id of the unnamed cluster.
        segment: Transcript sample under inspection.
        candidates: Ranked library matches for this sample (best first).
        accepted_person_ids: Voiceprint person ids confirmed on some speaker in
            this project; only these are eligible foreign reassignment targets.
        conflict_margin: Required lead of the top match over the runner-up.
        foreign_threshold: Minimum top-match score to treat as foreign.

    Returns:
        Sample row tagged ``identity-foreign`` (with the foreign person carried
        in ``best_other``) or ``no-assignment``.
    """
    best = candidates[0] if candidates else None
    if best is None or best.person_id not in accepted_person_ids:
        return _sample_row(
            speaker_id, segment, None, None, best, None, "no-assignment", candidates
        )
    runner_up_score = candidates[1].score if len(candidates) > 1 else 0.0
    clear_margin = best.score - runner_up_score
    if best.score >= foreign_threshold and clear_margin >= conflict_margin:
        # Carry the foreign person through ``best_other`` so the reassignment
        # pipeline reads it via the same field as identity-conflict rows.
        return _sample_row(
            speaker_id, segment, None, None, best, best, "identity-foreign", candidates
        )
    return _sample_row(
        speaker_id, segment, None, None, best, None, "no-assignment", candidates
    )


def _sample_vector(
    context: _SampleMatchContext,
    speaker_id: int,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
    cache: dict[str, list[float]],
    cache_lock: Lock,
) -> list[float]:
    """Return a cached embedding vector for one transcript sample."""
    key = clip_embedding_cache_key(
        provider=context.provider,
        model=context.model,
        speaker_id=speaker_id,
        segment=segment,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
    )
    with cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return _normalize(cached)
    clip_path = _sample_clip_path(context.project_root, speaker_id, segment)
    _write_sample_clip(context.source, clip_path, segment, max_seconds, padding_seconds)
    embedding_path = _sample_embedding_clip_path(clip_path)
    trim_embedding_audio_silence(clip_path, embedding_path)
    raw_vector = embed_audio_file(embedding_path, provider=context.provider)
    with cache_lock:
        cache[key] = raw_vector
    return _normalize(raw_vector)


def _identity_status(
    assigned_score: float,
    best_other_score: float | None,
    margin_score: float | None,
    threshold: float,
    conflict_margin: float,
    ambiguous_margin: float,
) -> str:
    """Return the identity diagnostic status for one matched sample."""
    if best_other_score is not None and margin_score is not None:
        if margin_score <= -conflict_margin:
            return (
                "identity-conflict"
                if best_other_score >= threshold
                else "identity-weak"
            )
        if (
            max(assigned_score, best_other_score) >= threshold
            and abs(margin_score) < ambiguous_margin
        ):
            return "identity-ambiguous"
    if assigned_score < threshold and (
        best_other_score is None or best_other_score < threshold
    ):
        return "identity-weak"
    return "identity-ok"


def _sample_row(
    speaker_id: int,
    segment: SentenceSegment,
    assigned: _KnownSpeakerVector | None,
    assigned_score: float | None,
    best: VoiceprintCandidate | None,
    best_other: VoiceprintCandidate | None,
    status: str,
    candidates: tuple[VoiceprintCandidate, ...],
) -> SpeakerSampleMatch:
    """Build one sample match row."""
    margin_score = (
        None
        if assigned_score is None or best_other is None
        else assigned_score - best_other.score
    )
    return SpeakerSampleMatch(
        speaker_id,
        segment.sentence_id,
        segment.begin_time_ms,
        segment.end_time_ms,
        segment.text,
        None if assigned is None else assigned.person_id,
        None if assigned is None else assigned.name,
        assigned_score,
        None if best is None else best.person_id,
        None if best is None else best.name,
        None if best is None else best.score,
        None if best_other is None else best_other.person_id,
        None if best_other is None else best_other.name,
        None if best_other is None else best_other.score,
        margin_score,
        status,
        candidates,
    )


def _is_low_information_segment(segment: SentenceSegment) -> bool:
    """Return whether a sample is too short for identity-level judgment."""
    text = "".join(char for char in segment.text.strip() if char.isalnum())
    duration_ms = segment.end_time_ms - segment.begin_time_ms
    if len(text) <= LOW_INFO_TEXT_CHARS:
        return True
    return len(text) < MIN_SAMPLE_TEXT_CHARS and duration_ms < MIN_SAMPLE_DURATION_MS


def _sample_clip_path(
    project_root: Path, speaker_id: int, segment: SentenceSegment
) -> Path:
    """Return a deterministic clip path for one transcript sample."""
    sentence = "unknown" if segment.sentence_id is None else str(segment.sentence_id)
    filename = f"sentence_{sentence}_{segment.begin_time_ms}_{segment.end_time_ms}.wav"
    return (
        project_root
        / "tmp"
        / "voiceprint_sample_match"
        / f"speaker_{speaker_id}"
        / filename
    )


def _sample_embedding_clip_path(clip_path: Path) -> Path:
    """Return the preprocessed embedding clip path for one sample."""
    return clip_path.with_name(f"{clip_path.stem}_embedding.wav")


def _write_sample_clip(
    source: Path,
    output: Path,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
) -> None:
    """Extract one bounded sample clip."""
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


def _read_sample_cache(project_root: Path) -> dict[str, list[float]]:
    """Read the shared clip embedding cache (raw vectors)."""
    return read_clip_embedding_cache(project_root)


def _write_sample_cache(project_root: Path, cache: dict[str, list[float]]) -> None:
    """Persist the shared clip embedding cache."""
    write_clip_embedding_cache(project_root, cache)


def _summary_verdict(reports: list[SpeakerSampleMatchReport]) -> str:
    """Return a project-level identity match verdict."""
    counts = Counter(
        status
        for report in reports
        for status, count in report.status_counts.items()
        for _ in range(count)
    )
    if counts["identity-conflict"]:
        return (
            "identity-conflict: at least one sample matches another known person better"
        )
    if counts["identity-foreign"]:
        return "identity-foreign: some sentences in an unassigned cluster match a known person"
    if counts["identity-ambiguous"]:
        return "identity-ambiguous: some samples are close to another known person"
    if counts["identity-weak"]:
        return "identity-weak: some samples do not strongly match their assigned person"
    return "identity-usable: assigned speaker identities look consistent"


def _report_payload(report: SpeakerSampleMatchReport) -> dict[str, object]:
    """Return a JSON-safe speaker sample match report."""
    return {
        "speaker_id": report.speaker_id,
        "label": report.label,
        "assigned_person_id": report.assigned_person_id,
        "assigned_name": report.assigned_name,
        "sample_count": report.sample_count,
        "status_counts": report.status_counts,
        "samples": [_sample_payload(sample) for sample in report.samples],
    }


def _sample_payload(sample: SpeakerSampleMatch) -> dict[str, object]:
    """Return a JSON-safe sample match row."""
    return {
        "speaker_id": sample.speaker_id,
        "sentence_id": sample.sentence_id,
        "begin_time_ms": sample.begin_time_ms,
        "end_time_ms": sample.end_time_ms,
        "text": sample.text,
        "assigned_person_id": sample.assigned_person_id,
        "assigned_name": sample.assigned_name,
        "assigned_score": sample.assigned_score,
        "best_person_id": sample.best_person_id,
        "best_name": sample.best_name,
        "best_score": sample.best_score,
        "best_other_person_id": sample.best_other_person_id,
        "best_other_name": sample.best_other_name,
        "best_other_score": sample.best_other_score,
        "margin_score": sample.margin_score,
        "status": sample.status,
        "candidates": [
            {
                "person_id": candidate.person_id,
                "person_public_id": candidate.person_public_id,
                "name": candidate.name,
                "score": candidate.score,
            }
            for candidate in sample.candidates
        ],
    }
