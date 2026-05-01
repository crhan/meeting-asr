"""Voiceprint embedding generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from app.cli_ui import CliProgressReporter, emit_progress
from app.config import get_cache_dir, load_settings
from app.uploader import upload_file_to_oss
from app.utils import retry
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
    upsert_voiceprint_embedding,
)

VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN = "local-speechbrain"
VOICEPRINT_PROVIDER_BAILIAN = "bailian"
DEFAULT_VOICEPRINT_PROVIDER = VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN
LOCAL_SPEECHBRAIN_MODEL = "speechbrain-spkrec-ecapa-voxceleb"
BAILIAN_VOICEPRINT_MODEL = "bailian-audio-embedding"
SUPPORTED_VOICEPRINT_PROVIDERS = (VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN, VOICEPRINT_PROVIDER_BAILIAN)

_PROVIDER_ALIASES = {
    "speechbrain": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "local": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "local-speechbrain": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "bailian": VOICEPRINT_PROVIDER_BAILIAN,
    "aliyun": VOICEPRINT_PROVIDER_BAILIAN,
    "adb": VOICEPRINT_PROVIDER_BAILIAN,
}
_DEFAULT_MODELS = {
    VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN: LOCAL_SPEECHBRAIN_MODEL,
    VOICEPRINT_PROVIDER_BAILIAN: BAILIAN_VOICEPRINT_MODEL,
}
_SPEECHBRAIN_CLASSIFIER: Any | None = None


@dataclass(frozen=True, slots=True)
class VoiceprintEmbedSummary:
    """Summary for embedding stored voiceprint samples."""

    db_path: Path
    provider: str
    model: str
    embedded_count: int
    skipped_count: int


def embed_voiceprint_samples(
    *,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str | None,
    rebuild: bool,
    progress: CliProgressReporter | None = None,
) -> VoiceprintEmbedSummary:
    """
    Generate embeddings for stored voiceprint samples.

    Args:
        store_dir: Optional voiceprint store directory.
        provider: Embedding provider.
        endpoint: Optional provider endpoint.
        model: Embedding model key for SQLite.
        rebuild: Rebuild existing embeddings when true.
        progress: Optional progress reporter.

    Returns:
        Embedding summary.
    """
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(provider=provider, model=model)
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_all_voiceprint_samples(db_path)
    embedded_ids = set() if rebuild else list_embedded_sample_ids(resolved_model, db_path)
    embedded_count = 0
    skipped_count = 0
    emit_progress(progress, "Embedding voiceprint samples", total=len(samples), completed=0)
    for sample in samples:
        if sample.sample_id in embedded_ids:
            skipped_count += 1
            emit_progress(progress, "Skipping existing voiceprint embedding", advance=1)
            continue
        vector = embed_audio_file(sample.clip_path, provider=resolved_provider, endpoint=endpoint)
        upsert_voiceprint_embedding(sample.sample_id, resolved_model, vector, db_path)
        embedded_count += 1
        emit_progress(progress, "Embedded voiceprint sample", advance=1)
    emit_progress(progress, "Voiceprint embeddings ready")
    return VoiceprintEmbedSummary(db_path, resolved_provider, resolved_model, embedded_count, skipped_count)


def resolve_voiceprint_embedding_options(*, provider: str | None, model: str | None) -> tuple[str, str]:
    """
    Resolve provider and model names from CLI options plus global config.

    Args:
        provider: Optional provider override.
        model: Optional model storage key override.

    Returns:
        Normalized provider and model storage key.
    """
    resolved_provider = resolve_voiceprint_provider(provider)
    return resolved_provider, model or _DEFAULT_MODELS[resolved_provider]


def resolve_voiceprint_provider(provider: str | None) -> str:
    """
    Resolve a provider override or global provider config.

    Args:
        provider: Optional provider override.

    Returns:
        Normalized provider name.
    """
    if provider is None:
        settings = load_settings(require_oss=False, require_dashscope=False)
        provider = settings.voiceprint_embedding_provider
    normalized = _PROVIDER_ALIASES.get(provider.strip().lower())
    if normalized is None:
        supported = ", ".join(SUPPORTED_VOICEPRINT_PROVIDERS)
        raise ValueError(f"Unsupported voiceprint embedding provider: {provider}. Supported providers: {supported}")
    return normalized


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
    normalized_provider = resolve_voiceprint_provider(provider)
    if normalized_provider == VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN:
        return _embed_audio_with_local_speechbrain(path)
    return _embed_audio_with_bailian(path, endpoint)


def _embed_audio_with_local_speechbrain(path: Path) -> list[float]:
    """
    Generate one embedding with the local SpeechBrain ECAPA model.

    Args:
        path: Local WAV clip.

    Returns:
        Embedding vector.
    """
    classifier = _load_speechbrain_classifier()
    signal = classifier.load_audio(str(path))
    embedding = classifier.encode_batch(signal)
    return _flatten_embedding(embedding)


def _load_speechbrain_classifier() -> Any:
    """
    Load and cache the local SpeechBrain speaker encoder.

    Returns:
        SpeechBrain EncoderClassifier instance.
    """
    global _SPEECHBRAIN_CLASSIFIER
    if _SPEECHBRAIN_CLASSIFIER is not None:
        return _SPEECHBRAIN_CLASSIFIER
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(_speechbrain_install_message()) from exc
    _SPEECHBRAIN_CLASSIFIER = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(get_cache_dir() / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"),
    )
    return _SPEECHBRAIN_CLASSIFIER


def _speechbrain_install_message() -> str:
    """
    Return the local provider dependency installation message.

    Returns:
        Actionable install message.
    """
    return (
        "local-speechbrain voiceprint embedding requires optional dependencies. "
        "Install them with `uv sync --extra local-voiceprint` in the repo, or reinstall the tool with "
        '`uv tool install --editable ".[local-voiceprint]" --force`.'
    )


def _flatten_embedding(embedding: Any) -> list[float]:
    """
    Convert a tensor-like embedding into a plain vector.

    Args:
        embedding: Tensor-like object returned by SpeechBrain.

    Returns:
        Embedding vector.
    """
    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu()
    if hasattr(embedding, "squeeze"):
        embedding = embedding.squeeze()
    values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    return [float(item) for item in _flatten_values(values)]


def _flatten_values(values: Any) -> list[float]:
    """
    Flatten nested list-like numeric values.

    Args:
        values: Numeric scalar or nested list-like values.

    Returns:
        Flat numeric list.
    """
    if isinstance(values, list):
        flattened: list[float] = []
        for item in values:
            flattened.extend(_flatten_values(item))
        return flattened
    return [float(values)]


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
    response = _post_bailian_embedding(
        endpoint=resolved_endpoint,
        api_key=settings.dashscope_api_key,
        audio_url=audio_url,
    )
    return _extract_embedding_vector(response.json())


def _post_bailian_embedding(*, endpoint: str, api_key: str, audio_url: str) -> requests.Response:
    """
    Post one embedding request with retry for transient HTTP failures.

    Args:
        endpoint: Bailian/AnalyticDB embedding endpoint.
        api_key: DashScope-compatible API key.
        audio_url: Signed URL for the WAV clip.

    Returns:
        Successful HTTP response.
    """

    def _post() -> requests.Response:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input_audio": audio_url},
            timeout=120,
        )
        if response.status_code >= 400:
            response.raise_for_status()
        return response

    try:
        return retry(_post, attempts=3, delay_seconds=1.0)
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            raise RuntimeError(
                f"Bailian voiceprint embedding failed: HTTP {response.status_code} {response.text}"
            ) from exc
        raise


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
