"""Tests for the local CAM++ embedding infrastructure."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from app.infra import campplus


def test_checkpoint_source_is_pinned() -> None:
    """The checkpoint download must stay pinned to a verified artifact."""
    assert campplus.CAMPP_CHECKPOINT_URL.startswith("https://modelscope.cn/models/")
    assert campplus.CAMPP_CHECKPOINT_FILENAME in campplus.CAMPP_CHECKPOINT_URL
    assert len(campplus.CAMPP_CHECKPOINT_SHA256) == 64


def test_load_waveform_monoizes_and_resamples(tmp_path: Path) -> None:
    """Stereo non-16k input must come out as 16 kHz mono."""
    clip = tmp_path / "clip.wav"
    _write_wav(clip, sample_rate=8000, seconds=0.5, channels=2)

    waveform = campplus._load_waveform(clip)

    assert waveform.shape[0] == 1
    assert waveform.shape[1] == int(0.5 * campplus.CAMPP_SAMPLE_RATE)


def test_extract_fbank_is_mean_normalized(tmp_path: Path) -> None:
    """Features must be 80-dim fbank with per-dimension zero mean."""
    clip = tmp_path / "clip.wav"
    _write_wav(clip, sample_rate=16000, seconds=1.0, channels=1)

    features = campplus._extract_fbank(campplus._load_waveform(clip))

    assert features.shape[1] == campplus.CAMPP_FEAT_DIM
    assert features.shape[0] > 90
    assert abs(float(features.mean())) < 1e-4


def _write_wav(path: Path, *, sample_rate: int, seconds: float, channels: int) -> None:
    """Write a small sine-wave PCM16 WAV clip."""
    frame_count = int(sample_rate * seconds)
    frames = bytearray()
    for index in range(frame_count):
        value = int(20000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames += struct.pack("<h", value) * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(frames))
