"""Speaker naming helpers."""

from __future__ import annotations

import json
from collections import defaultdict
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
