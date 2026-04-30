"""Tests for environment diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.commands import doctor
from app.config import save_config_values

runner = CliRunner()


def test_doctor_warns_when_voiceprint_endpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default doctor should surface optional voiceprint config without failing."""
    _prepare_doctor(monkeypatch, tmp_path)
    save_config_values({"dashscope.api_key": "secret"})

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "warn voiceprint-embedding: voiceprint.embedding_endpoint is not configured" in result.output
    assert "Repair prompts:" in result.output
    assert "meeting-asr config set voiceprint.embedding_endpoint" in result.output


def test_doctor_can_require_voiceprint_embedding_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict voiceprint doctor mode should fail when the embedding endpoint is missing."""
    _prepare_doctor(monkeypatch, tmp_path)
    save_config_values(
        {
            "dashscope.api_key": "secret",
            "oss.access_key_id": "ak",
            "oss.access_key_secret": "sk",
            "oss.bucket_name": "bucket",
            "oss.region": "cn-test",
            "oss.endpoint": "oss-cn-test.aliyuncs.com",
        }
    )

    result = runner.invoke(app, ["doctor", "--require-voiceprint-embedding"])

    assert result.exit_code == 1
    assert "fail voiceprint-embedding: voiceprint.embedding_endpoint is not configured" in result.output
    assert "meeting-asr doctor --require-oss --require-voiceprint-embedding" in result.output


def test_doctor_accepts_voiceprint_embedding_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Valid voiceprint endpoint should pass strict voiceprint doctor mode."""
    _prepare_doctor(monkeypatch, tmp_path)
    save_config_values(
        {
            "dashscope.api_key": "secret",
            "oss.access_key_id": "ak",
            "oss.access_key_secret": "sk",
            "oss.bucket_name": "bucket",
            "oss.region": "cn-test",
            "oss.endpoint": "oss-cn-test.aliyuncs.com",
            "voiceprint.embedding_endpoint": "http://adb.example:8100/audio/embedding",
        }
    )

    result = runner.invoke(app, ["doctor", "--require-voiceprint-embedding"])

    assert result.exit_code == 0
    assert "ok   voiceprint-embedding: provider=bailian" in result.output
    assert "Repair prompts:" not in result.output


def _prepare_doctor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Configure deterministic doctor dependencies for tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path used as XDG config home.

    Returns:
        None.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in _CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(doctor, "_check_python", lambda: doctor.CheckResult("python", "ok", "test"))
    monkeypatch.setattr(
        doctor,
        "_check_python_packages",
        lambda *, require_oss: doctor.CheckResult("python-packages", "ok", f"require_oss={require_oss}"),
    )
    monkeypatch.setattr(doctor, "_check_ffmpeg", lambda: doctor.CheckResult("ffmpeg", "ok", "test"))
    monkeypatch.setattr(doctor, "_check_preview_player", lambda: doctor.CheckResult("preview-player", "ok", "test"))


_CONFIG_ENV_NAMES = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "OSS_ACCESS_KEY_ID",
    "OSS_ACCESS_KEY_SECRET",
    "OSS_BUCKET_NAME",
    "OSS_REGION",
    "OSS_ENDPOINT",
    "VOICEPRINT_EMBEDDING_ENDPOINT",
    "VOICEPRINT_EMBEDDING_PROVIDER",
)
