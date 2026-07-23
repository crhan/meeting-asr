"""Tests for the shared per-clip embedding cache."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import SentenceSegment
from app.speaker_clip_embeddings import (
    clip_embedding_cache_key,
    clip_embedding_cache_path,
    read_clip_embedding_cache,
    write_clip_embedding_cache,
)


def _segment() -> SentenceSegment:
    return SentenceSegment(1000, 4000, "测试句子。", 2, 7)


def test_cache_key_is_stable_and_parameter_sensitive() -> None:
    """Same clip parameters share one key; any parameter change forks it."""
    base = clip_embedding_cache_key(
        provider="local-speechbrain",
        model="ecapa",
        speaker_id=2,
        segment=_segment(),
        max_seconds=12.0,
        padding_seconds=0.5,
    )
    assert base == clip_embedding_cache_key(
        provider="local-speechbrain",
        model="ecapa",
        speaker_id=2,
        segment=_segment(),
        max_seconds=12.0,
        padding_seconds=0.5,
    )
    assert base != clip_embedding_cache_key(
        provider="local-speechbrain",
        model="other-model",
        speaker_id=2,
        segment=_segment(),
        max_seconds=12.0,
        padding_seconds=0.5,
    )
    assert base != clip_embedding_cache_key(
        provider="local-speechbrain",
        model="ecapa",
        speaker_id=2,
        segment=_segment(),
        max_seconds=8.0,
        padding_seconds=0.5,
    )


def test_write_merges_with_existing_entries(tmp_path: Path) -> None:
    """Writing a snapshot must not drop entries persisted by another stage."""
    write_clip_embedding_cache(tmp_path, {"a": [1.0, 2.0]})
    write_clip_embedding_cache(tmp_path, {"b": [3.0, 4.0]})

    cache = read_clip_embedding_cache(tmp_path)

    assert cache == {"a": [1.0, 2.0], "b": [3.0, 4.0]}


def test_read_ignores_corrupt_payloads(tmp_path: Path) -> None:
    """Unreadable or malformed cache files degrade to a cold cache."""
    path = clip_embedding_cache_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert read_clip_embedding_cache(tmp_path) == {}

    path.write_text(
        json.dumps({"good": [1.0], "bad": "nope", "worse": [None]}),
        encoding="utf-8",
    )
    assert read_clip_embedding_cache(tmp_path) == {"good": [1.0]}
