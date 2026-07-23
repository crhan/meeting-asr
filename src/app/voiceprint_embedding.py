"""Voiceprint embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.progress import CliProgressReporter, emit_progress
from app.config import get_cache_dir, get_configured_voiceprint_provider
from app.utils import suppress_noisy_dependency_info_logs
from app.voiceprint_audio import (
    VOICEPRINT_AUDIO_PREPROCESS_VERSION,
    ensure_normalized_voiceprint_sample,
)
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
    upsert_voiceprint_embedding,
)

VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN = "local-speechbrain"
VOICEPRINT_PROVIDER_LOCAL_CAMPP = "local-campp"
DEFAULT_VOICEPRINT_PROVIDER = VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN
LOCAL_SPEECHBRAIN_BASE_MODEL = "speechbrain-spkrec-ecapa-voxceleb"
LOCAL_SPEECHBRAIN_MODEL = (
    f"{LOCAL_SPEECHBRAIN_BASE_MODEL}+{VOICEPRINT_AUDIO_PREPROCESS_VERSION}"
)
LOCAL_CAMPP_BASE_MODEL = "campplus-sv-zh-cn-16k-common"
LOCAL_CAMPP_MODEL = f"{LOCAL_CAMPP_BASE_MODEL}+{VOICEPRINT_AUDIO_PREPROCESS_VERSION}"
SUPPORTED_VOICEPRINT_PROVIDERS = (
    VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    VOICEPRINT_PROVIDER_LOCAL_CAMPP,
)

_PROVIDER_ALIASES = {
    "speechbrain": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "local": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "local-speechbrain": VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    "campp": VOICEPRINT_PROVIDER_LOCAL_CAMPP,
    "campplus": VOICEPRINT_PROVIDER_LOCAL_CAMPP,
    "cam++": VOICEPRINT_PROVIDER_LOCAL_CAMPP,
    "local-campp": VOICEPRINT_PROVIDER_LOCAL_CAMPP,
}
_DEFAULT_MODELS = {
    VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN: LOCAL_SPEECHBRAIN_MODEL,
    VOICEPRINT_PROVIDER_LOCAL_CAMPP: LOCAL_CAMPP_MODEL,
}
_PROVIDER_BY_BASE_MODEL = {
    LOCAL_SPEECHBRAIN_BASE_MODEL: VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN,
    LOCAL_CAMPP_BASE_MODEL: VOICEPRINT_PROVIDER_LOCAL_CAMPP,
}
_SPEECHBRAIN_CLASSIFIER: Any | None = None
_SPEECHBRAIN_CLASSIFIER_LOCK = Lock()


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
    model: str | None,
    rebuild: bool,
    sample_ids: set[int] | None = None,
    progress: CliProgressReporter | None = None,
) -> VoiceprintEmbedSummary:
    """
    Generate embeddings for stored voiceprint samples.

    Args:
        store_dir: Optional voiceprint store directory.
        provider: Optional local provider alias.
        model: Embedding model key for SQLite.
        rebuild: Rebuild existing embeddings when true.
        sample_ids: Optional registry row ids to embed.  When provided, no
            unrelated library samples are read or modified.
        progress: Optional progress reporter.

    Returns:
        Embedding summary.
    """
    resolved_provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    db_path = get_voiceprint_db_path(store_dir)
    samples = list_all_voiceprint_samples(db_path)
    if sample_ids is not None:
        available = {sample.sample_id for sample in samples}
        missing = sample_ids.difference(available)
        if missing:
            joined = ", ".join(str(value) for value in sorted(missing))
            raise LookupError(f"No voiceprint sample found for row id(s): {joined}")
        samples = [sample for sample in samples if sample.sample_id in sample_ids]
    embedded_ids = (
        set() if rebuild else list_embedded_sample_ids(resolved_model, db_path)
    )
    embedded_count = 0
    skipped_count = 0
    emit_progress(
        progress, "Embedding voiceprint samples", total=len(samples), completed=0
    )
    for sample in samples:
        if sample.sample_id in embedded_ids:
            skipped_count += 1
            emit_progress(progress, "Skipping existing voiceprint embedding", advance=1)
            continue
        normalized_path = ensure_normalized_voiceprint_sample(
            sample, store_dir=store_dir
        )
        vector = embed_audio_file(normalized_path, provider=resolved_provider)
        upsert_voiceprint_embedding(sample.sample_id, resolved_model, vector, db_path)
        embedded_count += 1
        emit_progress(progress, "Embedded voiceprint sample", advance=1)
    emit_progress(progress, "Voiceprint embeddings ready")
    return VoiceprintEmbedSummary(
        db_path, resolved_provider, resolved_model, embedded_count, skipped_count
    )


def resolve_voiceprint_embedding_options(
    *, provider: str | None, model: str | None
) -> tuple[str, str]:
    """
    Resolve provider and model names from CLI options plus global config.

    Args:
        provider: Optional provider override.
        model: Optional model storage key override.

    Returns:
        Normalized provider and model storage key.
    """
    if provider is None and model:
        inferred = _provider_for_model_key(model)
        if inferred is not None:
            return inferred, model
    resolved_provider = resolve_voiceprint_provider(provider)
    return resolved_provider, model or _DEFAULT_MODELS[resolved_provider]


def resolve_voiceprint_provider(provider: str | None) -> str:
    """
    Resolve a provider override or global provider config.

    Args:
        provider: Optional provider override. When omitted, the
            ``voiceprint.provider`` config value (or its environment variable)
            is used before falling back to the built-in default.

    Returns:
        Normalized provider name.
    """
    if provider is None:
        provider = get_configured_voiceprint_provider()
    if provider is None:
        return DEFAULT_VOICEPRINT_PROVIDER
    normalized = _PROVIDER_ALIASES.get(provider.strip().lower())
    if normalized is None:
        supported = ", ".join(SUPPORTED_VOICEPRINT_PROVIDERS)
        raise ValueError(
            f"Unsupported voiceprint embedding provider: {provider}. Supported providers: {supported}"
        )
    return normalized


def _provider_for_model_key(model: str) -> str | None:
    """
    Infer the embedding provider from a model storage key.

    Model keys are provider-specific (``<base-model>+<preprocess-version>``),
    so commands that only accept ``--model`` can still address a non-default
    provider without a separate provider flag.

    Args:
        model: Model storage key.

    Returns:
        Provider name, or None for unrecognized custom keys.
    """
    for base_model, provider in _PROVIDER_BY_BASE_MODEL.items():
        if model == base_model or model.startswith(f"{base_model}+"):
            return provider
    return None


def embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
    """
    Generate one audio embedding.

    Args:
        path: Local WAV clip.
        provider: Embedding provider.

    Returns:
        Embedding vector.
    """
    normalized_provider = resolve_voiceprint_provider(provider)
    if normalized_provider == VOICEPRINT_PROVIDER_LOCAL_CAMPP:
        return _embed_audio_with_local_campp(path)
    if normalized_provider != VOICEPRINT_PROVIDER_LOCAL_SPEECHBRAIN:
        supported = ", ".join(SUPPORTED_VOICEPRINT_PROVIDERS)
        raise ValueError(
            f"Unsupported voiceprint embedding provider: {normalized_provider}. Supported providers: {supported}"
        )
    return _embed_audio_with_local_speechbrain(path)


def _embed_audio_with_local_campp(path: Path) -> list[float]:
    """
    Generate one embedding with the local CAM++ (3D-Speaker) model.

    Args:
        path: Local WAV clip.

    Returns:
        Embedding vector.
    """
    from app.infra import campplus

    return campplus.embed_audio(path)


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
    with _SPEECHBRAIN_CLASSIFIER_LOCK:
        if _SPEECHBRAIN_CLASSIFIER is not None:
            return _SPEECHBRAIN_CLASSIFIER
        with suppress_noisy_dependency_info_logs():
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier  # type: ignore[no-redef]
                except ImportError as exc:
                    raise RuntimeError(_speechbrain_install_message()) from exc
            _SPEECHBRAIN_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(
                    get_cache_dir() / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
                ),
            )
        return _SPEECHBRAIN_CLASSIFIER


def _speechbrain_install_message() -> str:
    """
    Return the local provider dependency installation message.

    Returns:
        Actionable install message.
    """
    return (
        "local-speechbrain voiceprint embedding requires standard dependencies. "
        "Install them with `uv sync` in the repo, or refresh the global tool with "
        "`scripts/install-tool.sh` from the repo."
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
