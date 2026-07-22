"""Shared per-clip embedding cache for project speaker diagnostics.

Speaker matching (probe vectors), cluster quality, per-sentence sample
matching, and re-split analysis all embed clips cut with the same boundary
math (padding + max-duration window, then silence trim). Historically each
subsystem kept its own cache file with an identically-shaped key, so one
stabilization pass embedded the same sentence audio up to three times.

This module is the single cache all of them share. Vectors are stored RAW
(pre-normalization): per-clip consumers L2-normalize after lookup, while the
probe path averages raw vectors before normalizing — both behaviours are
bit-identical to the previous per-subsystem caches, so scores do not move.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.models import SentenceSegment
from app.utils import safe_write_json
from app.voiceprint_audio import VOICEPRINT_AUDIO_PREPROCESS_VERSION

CLIP_EMBEDDING_CACHE_RELATIVE_PATH = (
    Path("tmp") / "voiceprint_clips" / "clip_embeddings.json"
)


def clip_embedding_cache_key(
    *,
    provider: str | None,
    model: str,
    speaker_id: int,
    segment: SentenceSegment,
    max_seconds: float,
    padding_seconds: float,
) -> str:
    """
    Return the stable cache key for one extracted clip embedding.

    The key covers everything that shapes the clip audio and the embedding:
    provider/model, audio preprocess version, the sentence identity and time
    range, and the clip window parameters. Version 3 marks the switch to raw
    (pre-normalization) stored vectors; older per-subsystem caches used
    version 2 keys and are simply never hit.

    Args:
        provider: Embedding provider key.
        model: Embedding model key.
        speaker_id: Project speaker id owning the sentence.
        segment: Transcript sentence backing the clip.
        max_seconds: Maximum clip duration.
        padding_seconds: Context padding around the sentence.

    Returns:
        Hex cache key.
    """
    payload = {
        "version": 3,
        "provider": provider,
        "model": model,
        "audio_preprocess": VOICEPRINT_AUDIO_PREPROCESS_VERSION,
        "speaker_id": speaker_id,
        "sentence_id": segment.sentence_id,
        "begin_time_ms": segment.begin_time_ms,
        "end_time_ms": segment.end_time_ms,
        "text": segment.text,
        "max_seconds": max_seconds,
        "padding_seconds": padding_seconds,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def clip_embedding_cache_path(project_root: Path) -> Path:
    """Return the shared project-local clip embedding cache path."""
    return project_root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH


def read_clip_embedding_cache(project_root: Path) -> dict[str, list[float]]:
    """
    Read the shared clip embedding cache.

    Args:
        project_root: Project root directory.

    Returns:
        Mapping of cache key to raw embedding vector (empty when absent or
        unreadable).
    """
    path = clip_embedding_cache_path(project_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    return _valid_cache_payload(payload)


def write_clip_embedding_cache(
    project_root: Path, cache: dict[str, list[float]]
) -> None:
    """
    Persist clip embeddings, merging with entries written by other stages.

    The cache file is shared across subsystems that run at different times in
    one workflow; merge-on-write keeps a later stage from clobbering vectors
    another stage persisted after this one loaded its snapshot.

    Args:
        project_root: Project root directory.
        cache: Vectors to persist.
    """
    merged = read_clip_embedding_cache(project_root)
    merged.update(cache)
    safe_write_json(clip_embedding_cache_path(project_root), merged)


def _valid_cache_payload(payload: object) -> dict[str, list[float]]:
    """Filter a JSON payload down to valid embedding vectors."""
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


__all__ = [
    "CLIP_EMBEDDING_CACHE_RELATIVE_PATH",
    "clip_embedding_cache_key",
    "clip_embedding_cache_path",
    "read_clip_embedding_cache",
    "write_clip_embedding_cache",
]
