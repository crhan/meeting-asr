"""Compare DingTalk and local SRT speaker labels."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SrtEntry:
    """One parsed SRT entry."""

    start_ms: int
    end_ms: int
    speaker: str
    text: str
    source: str


def parse_srt(path: Path, *, source: str) -> list[SrtEntry]:
    """
    Parse a simple SRT file.

    Args:
        path: SRT path.
        source: Source label.

    Returns:
        Parsed entries.
    """
    content = path.read_text(encoding="utf-8-sig")
    entries: list[SrtEntry] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start, end = [
            _parse_srt_time(item.strip()) for item in lines[1].split("-->", 1)
        ]
        speaker, text = _split_speaker(" ".join(lines[2:]))
        entries.append(SrtEntry(start, end, speaker, text, source))
    return entries


def build_report(*, dingtalk: list[SrtEntry], ours: list[SrtEntry]) -> str:
    """
    Build a speaker comparison markdown report.

    Args:
        dingtalk: DingTalk entries.
        ours: Local entries.

    Returns:
        Markdown report.
    """
    pairs = Counter()
    mixed: list[str] = []
    for ding in dingtalk:
        overlaps = _overlaps(ding, ours)
        labels = Counter(entry.speaker for entry in overlaps)
        if labels:
            pairs[(ding.speaker, labels.most_common(1)[0][0])] += 1
        if len(labels) > 1:
            mixed.append(
                f"- `{ding.speaker}` {ding.start_ms}-{ding.end_ms} overlaps {dict(labels)}"
            )
    lines = ["# Speaker Comparison", "", "## Dominant Label Pairs"]
    for (ding_label, our_label), count in pairs.most_common():
        lines.append(f"- `{ding_label}` -> `{our_label}`: {count}")
    lines.extend(["", "## DingTalk Blocks Covering Multiple Local Speakers"])
    lines.extend(mixed or ["- None"])
    return "\n".join(lines) + "\n"


def _overlaps(entry: SrtEntry, candidates: list[SrtEntry]) -> list[SrtEntry]:
    """Return candidates that overlap an entry."""
    return [
        candidate
        for candidate in candidates
        if min(entry.end_ms, candidate.end_ms) > max(entry.start_ms, candidate.start_ms)
    ]


def _split_speaker(text: str) -> tuple[str, str]:
    """Split speaker prefix from text."""
    for sep in (":", "："):
        if sep in text:
            speaker, body = text.split(sep, 1)
            return speaker.strip(), body.strip()
    return "Unknown", text.strip()


def _parse_srt_time(value: str) -> int:
    """Parse SRT timestamp to milliseconds."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value)
    if not match:
        return 0
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis
