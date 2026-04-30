"""Voiceprint embedding generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from app.config import load_settings
from app.uploader import upload_file_to_oss
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
    upsert_voiceprint_embedding,
)

DEFAULT_VOICEPRINT_MODEL = "bailian-audio-embedding"
DEFAULT_VOICEPRINT_PROVIDER = "bailian"


@dataclass(frozen=True, slots=True)
class VoiceprintEmbedSummary:
    """Summary for embedding stored voiceprint samples."""

    db_path: Path
    model: str
    embedded_count: int
    skipped_count: int


def embed_voiceprint_samples(
    *,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str,
    rebuild: bool,
) -> VoiceprintEmbedSummary:
    """
    Generate embeddings for stored voiceprint samples.

    Args:
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        endpoint: Optional provider endpoint.
        model: Embedding model key for SQLite.
        rebuild: Rebuild existing embeddings when true.

    Returns:
        Embedding summary.
    """
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_all_voiceprint_samples(db_path)
    embedded_ids = set() if rebuild else list_embedded_sample_ids(model, db_path)
    embedded_count = 0
    skipped_count = 0
    for sample in samples:
        if sample.sample_id in embedded_ids:
            skipped_count += 1
            continue
        vector = embed_audio_file(sample.clip_path, provider=provider, endpoint=endpoint)
        upsert_voiceprint_embedding(sample.sample_id, model, vector, db_path)
        embedded_count += 1
    return VoiceprintEmbedSummary(db_path, model, embedded_count, skipped_count)


def embed_audio_file(path: Path, *, provider: str | None, endpoint: str | None) -> list[float]:
    """
    Generate one audio embedding.

    Args:
        path: Local WAV clip.
        provider: Embedding provider.
        endpoint: Optional provider endpoint.

    Returns:
        Embedding vector.
    """
    settings = load_settings(require_oss=False)
    normalized_provider = (provider or settings.voiceprint_embedding_provider).strip().lower()
    if normalized_provider != DEFAULT_VOICEPRINT_PROVIDER:
        raise ValueError(f"Unsupported voiceprint embedding provider: {provider}")
    return _embed_audio_with_bailian(path, endpoint)


def _embed_audio_with_bailian(path: Path, endpoint: str | None) -> list[float]:
    """
    Generate one embedding through the Bailian/AnalyticDB voiceprint endpoint.

    Args:
        path: Local WAV clip.
        endpoint: Optional endpoint override.

    Returns:
        Embedding vector.
    """
    settings = load_settings(require_oss=True)
    resolved_endpoint = endpoint or settings.voiceprint_embedding_endpoint
    if not resolved_endpoint:
        raise ValueError(
            "voiceprint.embedding_endpoint is required for Bailian voiceprint embedding. "
            "This is not a local install or a Tongyi vision model name. "
            "Get the address from AnalyticDB MySQL voiceprint retrieval or AI Application call information, "
            "then configure it with "
            '`meeting-asr config set voiceprint.embedding_endpoint "http://<adb-ai-app-host>:8100/audio/embedding"` '
            "or pass `--endpoint`."
        )
    audio_url = _upload_embedding_audio(path)
    response = requests.post(
        resolved_endpoint,
        headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
        json={"input_audio": audio_url},
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Bailian voiceprint embedding failed: HTTP {response.status_code} {response.text}")
    return _extract_embedding_vector(response.json())


def _upload_embedding_audio(path: Path) -> str:
    """
    Upload one local clip and return a signed URL for embedding.

    Args:
        path: Local WAV clip.

    Returns:
        Signed HTTP URL.
    """
    settings = load_settings(require_oss=True)
    digest = _sha256_file(path)[:16]
    object_name = f"meeting-asr/voiceprint-embedding/{digest}-{path.name}"
    return upload_file_to_oss(path, object_name=object_name, settings=settings)


def _extract_embedding_vector(payload: dict[str, Any]) -> list[float]:
    """
    Extract an embedding vector from provider response JSON.

    Args:
        payload: Response JSON.

    Returns:
        Embedding vector.
    """
    result = payload.get("result")
    if isinstance(result, list):
        return [float(item) for item in result]
    raise RuntimeError(f"Bailian voiceprint embedding response did not contain a vector: {payload}")


def _sha256_file(path: Path) -> str:
    """
    Hash a clip without loading it all into memory.

    Args:
        path: Clip path.

    Returns:
        SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
