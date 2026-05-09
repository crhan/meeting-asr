"""Speaker naming helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.models import SentenceSegment, TranscriptResult
from app.postprocess import detect_speaker_ids, filter_filler_speakers, speaker_id_to_label
from app.srt_utils import ms_to_srt_timestamp
from app.utils import safe_write_json


@dataclass(slots=True)
class SpeakerSummary:
    """Terminal summary for one speaker."""

    speaker_id: int
    anonymous_label: str
    segment_count: int
    first_begin_time_ms: int
    sample_segments: list[SentenceSegment]


def load_transcript_result(path: Path) -> TranscriptResult:
    """
    Load normalized sentences.json.

    Args:
        path: Sentences JSON path.

    Returns:
        Transcript result.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    sentences = [SentenceSegment(**item) for item in payload.get("sentences", [])]
    sentences = filter_filler_speakers(sentences)
    full_text = "".join(sentence.text for sentence in sentences)
    result = TranscriptResult(full_text, sentences, [])
    result.detected_speakers = detect_speaker_ids(result)
    return result


def build_default_mapping(result: TranscriptResult) -> dict[int, str]:
    """
    Build anonymous fallback mapping.

    Args:
        result: Transcript result.

    Returns:
        Mapping from speaker id to label.
    """
    return {speaker_id: speaker_id_to_label(speaker_id) for speaker_id in result.detected_speakers}


def build_speaker_summaries(result: TranscriptResult, *, sample_count: int = 5) -> list[SpeakerSummary]:
    """
    Build compact samples for each speaker.

    Args:
        result: Transcript result.
        sample_count: Maximum samples per speaker.

    Returns:
        Speaker summaries.
    """
    grouped: dict[int, list[SentenceSegment]] = defaultdict(list)
    for sentence in result.sentences:
        if sentence.speaker_id is not None and sentence.text.strip():
            grouped[sentence.speaker_id].append(sentence)
    summaries: list[SpeakerSummary] = []
    for speaker_id in sorted(grouped):
        samples = grouped[speaker_id][:sample_count]
        summaries.append(
            SpeakerSummary(
                speaker_id=speaker_id,
                anonymous_label=speaker_id_to_label(speaker_id),
                segment_count=len(grouped[speaker_id]),
                first_begin_time_ms=samples[0].begin_time_ms,
                sample_segments=samples,
            )
        )
    return summaries


def write_speaker_mapping(path: Path, speaker_mapping: dict[int, str]) -> Path:
    """
    Write speaker mapping JSON.

    Args:
        path: Output path.
        speaker_mapping: Mapping.

    Returns:
        Written path.
    """
    payload = {str(key): value for key, value in sorted(speaker_mapping.items())}
    return safe_write_json(path, payload)


def write_speaker_person_mapping(path: Path, speaker_person_mapping: dict[int, int | str]) -> Path:
    """
    Write project speaker to voiceprint person id mapping.

    Args:
        path: Output path.
        speaker_person_mapping: Mapping from project speaker id to voiceprint person id.

    Returns:
        Written path.
    """
    payload = {str(key): value for key, value in sorted(speaker_person_mapping.items())}
    return safe_write_json(path, payload)


def write_ignored_speakers(path: Path, speaker_ids: set[int]) -> Path:
    """
    Write explicitly ignored project speaker ids.

    Args:
        path: Output path.
        speaker_ids: Project speaker ids deliberately kept anonymous.

    Returns:
        Written path.
    """
    payload = {"ignored_speakers": sorted(speaker_ids)}
    return safe_write_json(path, payload)


def load_ignored_speakers(path: Path) -> set[int]:
    """
    Load explicitly ignored project speaker ids.

    Args:
        path: Ignore metadata JSON path.

    Returns:
        Set of project speaker ids deliberately kept anonymous.
    """
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {int(value) for value in payload}
    return {int(value) for value in payload.get("ignored_speakers", [])}


def load_speaker_person_mapping(path: Path) -> dict[int, int | str]:
    """
    Load project speaker to voiceprint person id mapping.

    Args:
        path: Mapping JSON path.

    Returns:
        Project speaker id to voiceprint person id reference mapping.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, int | str] = {}
    for key, value in payload.items():
        mapping[int(key)] = value
    return mapping


@dataclass(frozen=True, slots=True)
class SentenceReassignmentSpec:
    """Lightweight identity for one sentence whose speaker_id should change.

    The triple ``(sentence_id, begin_time_ms, end_time_ms)`` matches the
    sentence inside the persisted JSON payload. ``sentence_id`` is preferred
    when present; the timing pair is used as a fallback because some legacy
    ASR payloads omit a stable id.
    """

    sentence_id: int | None
    begin_time_ms: int
    end_time_ms: int
    new_speaker_id: int


def apply_sentence_reassignments(
    asr_dir: Path,
    reassignments: Iterable[SentenceReassignmentSpec],
) -> tuple[Path, ...]:
    """Update speaker_id in persisted ASR sentence files.

    Mutates ``sentences.json`` and ``sentences_corrected.json`` (when present)
    in place, matching each reassignment by sentence identity. The full payload
    is rewritten so filler-only speaker tracks and unrelated sentences are
    preserved verbatim.

    Args:
        asr_dir: Project ``asr/`` directory.
        reassignments: Sentence speaker reassignments to apply.

    Returns:
        Paths of updated transcript files.

    Raises:
        ValueError: When a reassignment cannot be matched to any sentence.
    """
    specs = list(reassignments)
    if not specs:
        return ()
    candidate_paths = (asr_dir / "sentences.json", asr_dir / "sentences_corrected.json")
    written: list[Path] = []
    for path in candidate_paths:
        if not path.exists():
            continue
        if _rewrite_sentences_with_reassignments(path, specs):
            written.append(path)
    if not written:
        raise ValueError(f"No transcript file under {asr_dir} accepted reassignments.")
    return tuple(written)


def _rewrite_sentences_with_reassignments(
    path: Path,
    reassignments: Sequence[SentenceReassignmentSpec],
) -> bool:
    """Rewrite one sentence file with the given reassignments.

    Args:
        path: Sentence file (raw or corrected) to update.
        reassignments: Reassignments to apply.

    Returns:
        ``True`` when the file was rewritten, ``False`` when the payload was
        missing the ``sentences`` key (treated as a no-op).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return False
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        return False
    by_sentence_id: dict[int, SentenceReassignmentSpec] = {}
    by_timing: dict[tuple[int, int], SentenceReassignmentSpec] = {}
    for spec in reassignments:
        if spec.sentence_id is not None:
            by_sentence_id[int(spec.sentence_id)] = spec
        by_timing[(int(spec.begin_time_ms), int(spec.end_time_ms))] = spec
    matched_any = False
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        spec = _match_reassignment(sentence, by_sentence_id, by_timing)
        if spec is None:
            continue
        sentence["speaker_id"] = int(spec.new_speaker_id)
        matched_any = True
    if matched_any:
        payload["detected_speakers"] = _recompute_detected_speakers(sentences)
    safe_write_json(path, payload)
    return True


def _match_reassignment(
    sentence: dict,
    by_sentence_id: dict[int, SentenceReassignmentSpec],
    by_timing: dict[tuple[int, int], SentenceReassignmentSpec],
) -> SentenceReassignmentSpec | None:
    """Return the reassignment matching one sentence payload."""
    sentence_id = sentence.get("sentence_id")
    if sentence_id is not None:
        try:
            spec = by_sentence_id.get(int(sentence_id))
        except (TypeError, ValueError):
            spec = None
        if spec is not None:
            return spec
    try:
        begin = int(sentence.get("begin_time_ms", 0))
        end = int(sentence.get("end_time_ms", 0))
    except (TypeError, ValueError):
        return None
    return by_timing.get((begin, end))


def _recompute_detected_speakers(sentences: list) -> list[int]:
    """Return sorted speaker ids present in the rewritten sentence list."""
    ids: set[int] = set()
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        speaker_id = sentence.get("speaker_id")
        if speaker_id is None:
            continue
        try:
            ids.add(int(speaker_id))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


def write_named_outputs(
    *,
    output_dir: Path,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> tuple[Path, Path, Path]:
    """
    Write named speaker outputs.

    Args:
        output_dir: Output directory.
        result: Transcript result.
        speaker_mapping: Speaker name mapping.

    Returns:
        Paths for map, transcript, and SRT.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = write_speaker_mapping(output_dir / "speaker_map.json", speaker_mapping)
    transcript_path = output_dir / "transcript_named.txt"
    srt_path = output_dir / "subtitle_named.srt"
    transcript_path.write_text(render_named_speaker_text(result, speaker_mapping), encoding="utf-8")
    srt_path.write_text(render_named_srt(result, speaker_mapping), encoding="utf-8")
    return map_path, transcript_path, srt_path


def render_named_speaker_text(result: TranscriptResult, speaker_mapping: dict[int, str]) -> str:
    """
    Render timestamped transcript with speaker names.

    Args:
        result: Transcript result.
        speaker_mapping: Speaker mapping.

    Returns:
        Named transcript.
    """
    lines: list[str] = []
    for sentence in result.sentences:
        text = sentence.text.strip()
        if not text:
            continue
        label = _speaker_name(sentence.speaker_id, speaker_mapping)
        start = _format_plain_timestamp(sentence.begin_time_ms)
        end = _format_plain_timestamp(sentence.end_time_ms)
        lines.append(f"[{start} - {end}] {label}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_named_srt(result: TranscriptResult, speaker_mapping: dict[int, str]) -> str:
    """
    Render SRT with speaker names.

    Args:
        result: Transcript result.
        speaker_mapping: Speaker mapping.

    Returns:
        SRT content.
    """
    blocks: list[str] = []
    index = 1
    for sentence in result.sentences:
        text = sentence.text.strip()
        if not text:
            continue
        label = _speaker_name(sentence.speaker_id, speaker_mapping)
        timestamp = (
            f"{ms_to_srt_timestamp(sentence.begin_time_ms)} "
            f"--> {ms_to_srt_timestamp(sentence.end_time_ms)}"
        )
        blocks.append("\n".join([str(index), timestamp, f"{label}: {text}"]))
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _speaker_name(speaker_id: int | None, mapping: dict[int, str]) -> str:
    """Return mapped speaker name or anonymous fallback."""
    if speaker_id is None:
        return "Speaker Unknown"
    return mapping.get(speaker_id, speaker_id_to_label(speaker_id))


def _format_plain_timestamp(ms: int) -> str:
    """Format timestamp for named transcript."""
    value = max(0, int(ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
