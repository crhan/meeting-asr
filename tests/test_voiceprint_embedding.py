"""Tests for voiceprint embedding providers."""

from __future__ import annotations

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
