"""Tests for voiceprint embedding providers."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import app.voiceprint_embedding as voiceprint_embedding


class _FakeSpeechBrainClassifier:
    """Minimal SpeechBrain classifier fake for local provider tests."""

    def __init__(self) -> None:
        """Initialize captured call state."""
        self.loaded_path: str | None = None

    def load_audio(self, path: str) -> object:
        """
        Record the loaded audio path.

        Args:
            path: Audio file path.

        Returns:
            Opaque fake signal object.
        """
        self.loaded_path = path
        return object()

    def encode_batch(self, signal: object) -> list[list[list[float]]]:
        """
        Return a nested embedding like SpeechBrain does.

        Args:
            signal: Opaque fake signal object.

        Returns:
            Nested embedding values.
        """
        return [[[0.1, 0.2, 0.3]]]


def test_local_speechbrain_embedding_uses_load_audio_and_encode_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Local SpeechBrain provider should use the current SpeechBrain inference API."""
    classifier = _FakeSpeechBrainClassifier()
    clip_path = tmp_path / "clip.wav"
    clip_path.write_bytes(b"fake wav")
    monkeypatch.setattr(voiceprint_embedding, "_load_speechbrain_classifier", lambda: classifier)

    vector = voiceprint_embedding._embed_audio_with_local_speechbrain(clip_path)

    assert classifier.loaded_path == str(clip_path)
    assert vector == [0.1, 0.2, 0.3]


def test_speechbrain_loader_hides_model_fetch_info(
    monkeypatch,
    capsys,
) -> None:
    """SpeechBrain model loading INFO should not pollute progress rendering."""
    logger, original_state = _capture_speechbrain_fetch_logger()
    monkeypatch.setattr(voiceprint_embedding, "_SPEECHBRAIN_CLASSIFIER", None)
    calls: dict[str, str] = {}
    _install_fake_speechbrain(monkeypatch, logger, calls)

    try:
        classifier = voiceprint_embedding._load_speechbrain_classifier()
    finally:
        _restore_logger(logger, original_state)

    captured = capsys.readouterr()
    assert classifier == "classifier"
    assert calls["source"] == "speechbrain/spkrec-ecapa-voxceleb"
    assert "Fetch hyperparams.yaml" not in captured.err


def _capture_speechbrain_fetch_logger() -> tuple[logging.Logger, tuple[int, list[logging.Handler], bool]]:
    """Capture SpeechBrain fetch logger output for a test."""
    logger = logging.getLogger("speechbrain.utils.fetching")
    original_state = (logger.level, list(logger.handlers), logger.propagate)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.handlers = [stream_handler]
    logger.propagate = False
    return logger, original_state


def _restore_logger(logger: logging.Logger, state: tuple[int, list[logging.Handler], bool]) -> None:
    """Restore a logger after a test."""
    level, handlers, propagate = state
    logger.handlers = handlers
    logger.propagate = propagate
    logger.setLevel(level)


def _install_fake_speechbrain(
    monkeypatch,
    logger: logging.Logger,
    calls: dict[str, str],
) -> None:
    """Install a fake SpeechBrain module tree into sys.modules."""
    speechbrain_module = types.ModuleType("speechbrain")
    inference_module = types.ModuleType("speechbrain.inference")
    speaker_module = types.ModuleType("speechbrain.inference.speaker")

    class FakeEncoderClassifier:
        """Fake SpeechBrain loader that resets its logger to INFO."""

        @classmethod
        def from_hparams(cls, *, source: str, savedir: str) -> str:
            """Simulate SpeechBrain's model loading side effects."""
            calls["source"] = source
            calls["savedir"] = savedir
            logger.setLevel(logging.INFO)
            logger.info("Fetch hyperparams.yaml")
            return "classifier"

    speaker_module.EncoderClassifier = FakeEncoderClassifier
    inference_module.speaker = speaker_module
    speechbrain_module.inference = inference_module
    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", speaker_module)
