"""SRT rendering utilities."""

from __future__ import annotations

from app.models import SentenceSegment
from app.postprocess import speaker_id_to_label


def ms_to_srt_timestamp(ms: int) -> str:
    """
    Convert milliseconds to SRT timestamp format.

    Args:
        ms: Milliseconds.

    Returns:
        Timestamp like ``00:00:01,234``.
    """
    value = max(0, int(ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(sentences: list[SentenceSegment], include_speaker_label: bool = True) -> str:
    """
    Build legal SRT content.

    Args:
        sentences: Sentence segments.
        include_speaker_label: Prefix text with speaker labels.

    Returns:
        SRT text.
    """
    blocks: list[str] = []
    index = 1
    for sentence in sentences:
        text = sentence.text.strip()
        if not text:
            continue
        prefix = f"{speaker_id_to_label(sentence.speaker_id)}: " if include_speaker_label else ""
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{ms_to_srt_timestamp(sentence.begin_time_ms)} --> {ms_to_srt_timestamp(sentence.end_time_ms)}",
                    f"{prefix}{text}",
                ]
            )
        )
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")
