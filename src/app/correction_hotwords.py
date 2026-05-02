"""ASR hotword payload generation from accepted correction knowledge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from app.correction_types import CorrectionUnderstanding
from app.utils import safe_write_json

DEFAULT_HOTWORD_WEIGHT = 4
MAX_HOTWORD_TEXT_LEN = 80


@dataclass(frozen=True, slots=True)
class AsrHotword:
    """One DashScope ASR hotword entry."""

    text: str
    weight: int = DEFAULT_HOTWORD_WEIGHT
    category: str = "unknown"
    source: str = "correction"

    def to_dashscope_dict(self) -> dict[str, object]:
        """Return the payload accepted by DashScope VocabularyService."""
        return {"text": self.text, "weight": self.weight}


def hotwords_from_understanding(
    understanding: list[CorrectionUnderstanding],
    *,
    category: str,
) -> list[AsrHotword]:
    """
    Build ASR hotwords from a correction proposal understanding.

    Args:
        understanding: Inferred accepted correction rules.
        category: Term category to attach to the hotword artifact.

    Returns:
        Unique hotwords suitable for ASR biasing.
    """
    values = []
    for item in understanding:
        word = normalize_hotword_text(item.corrected_text)
        if word:
            values.append(AsrHotword(text=word, category=category, source="correction_understanding"))
    return dedupe_hotwords(values)


def hotwords_from_terms(terms: list[tuple[str, str]]) -> list[AsrHotword]:
    """
    Build ASR hotwords from accepted lexicon terms.

    Args:
        terms: ``(canonical, category)`` rows.

    Returns:
        Unique hotwords suitable for ASR biasing.
    """
    values = []
    for canonical, category in terms:
        word = normalize_hotword_text(canonical)
        if word:
            values.append(AsrHotword(text=word, category=category or "unknown", source="lexicon"))
    return dedupe_hotwords(values)


def write_hotword_artifact(path: Path, hotwords: list[AsrHotword]) -> Path:
    """
    Write a stable hotword table artifact.

    Args:
        path: Output JSON path.
        hotwords: Hotwords to write.

    Returns:
        Written path.
    """
    payload = {
        "schema_version": 1,
        "count": len(hotwords),
        "hash": hotword_hash(hotwords),
        "dashscope_vocabulary": dashscope_vocabulary(hotwords),
        "hotwords": [asdict(item) for item in hotwords],
    }
    return safe_write_json(path, payload)


def dashscope_vocabulary(hotwords: list[AsrHotword]) -> list[dict[str, object]]:
    """
    Convert hotwords to DashScope vocabulary rows.

    Args:
        hotwords: Internal hotword entries.

    Returns:
        DashScope vocabulary payload rows.
    """
    return [item.to_dashscope_dict() for item in hotwords]


def hotword_hash(hotwords: list[AsrHotword]) -> str:
    """
    Return a stable hash for a hotword table.

    Args:
        hotwords: Hotword entries.

    Returns:
        SHA-256 hash prefix.
    """
    payload = json.dumps(dashscope_vocabulary(hotwords), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def dedupe_hotwords(hotwords: list[AsrHotword]) -> list[AsrHotword]:
    """
    Deduplicate hotwords by text while preserving first-seen order.

    Args:
        hotwords: Candidate hotword entries.

    Returns:
        Deduplicated hotword entries.
    """
    seen: set[str] = set()
    result = []
    for item in hotwords:
        if item.text in seen:
            continue
        seen.add(item.text)
        result.append(item)
    return result


def normalize_hotword_text(value: str) -> str | None:
    """
    Normalize and validate one hotword text.

    Args:
        value: Raw term text.

    Returns:
        Cleaned hotword text, or None when unsuitable.
    """
    text = " ".join(value.strip().split())
    if not text or len(text) > MAX_HOTWORD_TEXT_LEN:
        return None
    return text
