"""Parse DashScope transcription JSON and render text outputs."""

from __future__ import annotations

from typing import Any

from app.models import SentenceSegment, TranscriptResult
from app.utils import format_ms_timestamp


def parse_transcription_result(raw_json: dict[str, Any]) -> TranscriptResult:
    """
    Parse DashScope result JSON into normalized transcript data.

    Args:
        raw_json: Downloaded transcription result JSON.

    Returns:
        Normalized transcript result.
    """
    sentences = [_segment_from_payload(item, idx) for idx, item in enumerate(_find_sentence_payloads(raw_json))]
    sentences = [item for item in sentences if item is not None]
    full_text = _find_text(raw_json) or "".join(sentence.text for sentence in sentences)
    detected = detect_speaker_ids(TranscriptResult(full_text=full_text, sentences=sentences, detected_speakers=[]))
    return TranscriptResult(full_text=full_text, sentences=sentences, detected_speakers=detected)


def speaker_id_to_label(speaker_id: int | None) -> str:
    """
    Convert speaker id to an anonymous label.

    Args:
        speaker_id: ASR speaker id.

    Returns:
        Speaker label.
    """
    if speaker_id is None:
        return "Speaker Unknown"
    if 0 <= speaker_id < 26:
        return f"Speaker {chr(ord('A') + speaker_id)}"
    return f"Speaker {speaker_id + 1}"


def merge_adjacent_sentences(
    sentences: list[SentenceSegment],
    max_gap_ms: int = 1200,
    same_speaker_only: bool = True,
) -> list[SentenceSegment]:
    """
    Merge adjacent sentences for readable speaker transcript output.

    Args:
        sentences: Original sentences.
        max_gap_ms: Maximum gap allowed for merge.
        same_speaker_only: Only merge same speaker when true.

    Returns:
        Merged segments.
    """
    merged: list[SentenceSegment] = []
    for sentence in sentences:
        if not sentence.text.strip():
            continue
        if merged and _can_merge(merged[-1], sentence, max_gap_ms, same_speaker_only):
            previous = merged[-1]
            previous.end_time_ms = max(previous.end_time_ms, sentence.end_time_ms)
            previous.text = f"{previous.text.rstrip()}{sentence.text.strip()}"
            continue
        merged.append(SentenceSegment(**sentence.to_dict()))
    return merged


def render_plain_text(result: TranscriptResult) -> str:
    """
    Render plain transcript text.

    Args:
        result: Transcript result.

    Returns:
        Plain text.
    """
    if result.full_text.strip():
        return result.full_text.strip() + "\n"
    return "\n".join(sentence.text for sentence in result.sentences if sentence.text.strip()) + "\n"


def render_speaker_text(result: TranscriptResult) -> str:
    """
    Render timestamped speaker transcript.

    Args:
        result: Transcript result.

    Returns:
        Speaker transcript text.
    """
    lines: list[str] = []
    for sentence in result.sentences:
        text = sentence.text.strip()
        if not text:
            continue
        start = format_ms_timestamp(sentence.begin_time_ms)
        end = format_ms_timestamp(sentence.end_time_ms)
        lines.append(f"[{start} - {end}] {speaker_id_to_label(sentence.speaker_id)}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def detect_speaker_ids(result: TranscriptResult) -> list[int]:
    """
    Detect speaker ids present in a result.

    Args:
        result: Transcript result.

    Returns:
        Sorted speaker ids.
    """
    speakers = {item.speaker_id for item in result.sentences if item.speaker_id is not None}
    return sorted(speakers)


def _can_merge(first: SentenceSegment, second: SentenceSegment, max_gap_ms: int, same_speaker_only: bool) -> bool:
    """Return whether two segments can be merged."""
    if same_speaker_only and first.speaker_id != second.speaker_id:
        return False
    return second.begin_time_ms - first.end_time_ms <= max_gap_ms


def _find_sentence_payloads(raw_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Find sentence-like payloads across common DashScope result shapes."""
    candidates = [
        raw_json.get("transcripts"),
        raw_json.get("sentences"),
        raw_json.get("results"),
    ]
    for key in ("output", "result", "data"):
        nested = raw_json.get(key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("transcripts"), nested.get("sentences"), nested.get("results")])
    return _flatten_sentences(candidates)


def _flatten_sentences(candidates: list[Any]) -> list[dict[str, Any]]:
    """Flatten transcript containers into sentence dictionaries."""
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            for entry in candidate:
                if isinstance(entry, dict) and isinstance(entry.get("sentences"), list):
                    items.extend(item for item in entry["sentences"] if isinstance(item, dict))
                elif isinstance(entry, dict):
                    items.append(entry)
        elif isinstance(candidate, dict) and isinstance(candidate.get("sentences"), list):
            items.extend(item for item in candidate["sentences"] if isinstance(item, dict))
    return items


def _segment_from_payload(payload: dict[str, Any], fallback_id: int) -> SentenceSegment | None:
    """Build one segment from a loose DashScope sentence payload."""
    text = str(payload.get("text") or payload.get("sentence") or payload.get("transcript") or "").strip()
    if not text:
        return None
    begin = _to_int(payload.get("begin_time") or payload.get("begin_time_ms") or payload.get("start_time"))
    end = _to_int(payload.get("end_time") or payload.get("end_time_ms") or payload.get("stop_time"))
    speaker = _optional_int(payload.get("speaker_id"))
    sentence_id = _optional_int(payload.get("sentence_id") or payload.get("id")) or fallback_id
    return SentenceSegment(begin_time_ms=begin, end_time_ms=max(end, begin), text=text, speaker_id=speaker, sentence_id=sentence_id)


def _find_text(raw_json: dict[str, Any]) -> str | None:
    """Find a full text field across common result shapes."""
    for key in ("text", "transcript", "full_text"):
        value = raw_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in raw_json.values():
        if isinstance(value, dict):
            nested = _find_text(value)
            if nested:
                return nested
    return None


def _to_int(value: Any) -> int:
    """Coerce a time value to integer milliseconds."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    """Coerce optional integer values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
