"""ASR model capability registry."""

from __future__ import annotations

DEFAULT_ASR_MODEL = "fun-asr"
QWEN_FILETRANS_MODEL = "qwen3-asr-flash-filetrans"

SUPPORTED_ASR_MODELS = (
    DEFAULT_ASR_MODEL,
    QWEN_FILETRANS_MODEL,
)

QWEN_FILETRANS_MODELS = {
    QWEN_FILETRANS_MODEL,
    "qwen3-asr-flash-filetrans-2025-11-17",
}

DASHSCOPE_VOCABULARY_MODELS = {
    "fun-asr",
    "fun-asr-2025-11-07",
    "fun-asr-2025-08-25",
    "fun-asr-mtl",
    "fun-asr-mtl-2025-08-25",
    "paraformer-v2",
    "paraformer-8k-v2",
}


def is_qwen_filetrans_model(model: str) -> bool:
    """
    Return whether a model uses the Qwen file transcription API.

    Args:
        model: ASR model id.

    Returns:
        True for Qwen async file-transcription models.
    """
    return model.strip() in QWEN_FILETRANS_MODELS


def supports_asr_hotwords(model: str) -> bool:
    """
    Return whether DashScope vocabulary hotwords are supported.

    Args:
        model: ASR model id.

    Returns:
        True when ``vocabulary_id`` can be sent to the ASR request.
    """
    return model.strip() in DASHSCOPE_VOCABULARY_MODELS
