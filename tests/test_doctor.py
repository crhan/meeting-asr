"""Tests for environment diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.commands import doctor
from app.config import save_config_values

runner = CliRunner()


def test_doctor_warns_when_local_voiceprint_dependencies_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default doctor should surface missing local voiceprint dependencies without failing."""
    _prepare_doctor(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "_missing_optional_modules", lambda modules: ["speechbrain"])
    save_config_values({"dashscope.api_key": "secret"})

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Meeting-ASR Doctor" in result.output
    assert "Summary: 6 ok, 1 warn, 0 fail" in result.output
    assert "  WARN  voiceprint-embedding" in result.output
    assert "provider=local-speechbrain" in result.output
    assert "missing optional packages: speechbrain" in result.output
    assert "Repair Prompts:" in result.output
    assert "uv sync --extra local-voiceprint" in result.output


def test_doctor_json_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Doctor should expose the same checks as stable JSON."""
    _prepare_doctor(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "_missing_optional_modules", lambda modules: ["speechbrain"])
    save_config_values({"dashscope.api_key": "secret"})

    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["summary"] == {"ok": 6, "warn": 1, "fail": 0}
    assert payload["checks"][-1]["name"] == "voiceprint-embedding"
    assert payload["checks"][-1]["status"] == "warn"
    assert "missing optional packages: speechbrain" in payload["checks"][-1]["detail"]


def test_doctor_help_does_not_expose_python_docstring_sections() -> None:
    """Typer help should be user-facing, not Python API documentation."""
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "Args:" not in result.output
    assert "Returns:" not in result.output


def test_doctor_can_require_local_voiceprint_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict voiceprint doctor mode should fail when local dependencies are missing."""
    _prepare_doctor(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "_missing_optional_modules", lambda modules: ["speechbrain", "torch"])
    save_config_values({"dashscope.api_key": "secret"})

    result = runner.invoke(app, ["doctor", "--require-voiceprint-embedding"])

    assert result.exit_code == 1
    assert "Summary: 6 ok, 0 warn, 1 fail" in result.output
    assert "  FAIL  voiceprint-embedding" in result.output
    assert "missing optional packages: speechbrain, torch" in result.output
    assert "meeting-asr doctor --require-voiceprint-embedding" in result.output


def test_doctor_accepts_bailian_voiceprint_embedding_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Valid Bailian endpoint should pass strict voiceprint doctor mode."""
    _prepare_doctor(monkeypatch, tmp_path)
    save_config_values(
        {
            "dashscope.api_key": "secret",
            "oss.access_key_id": "ak",
            "oss.access_key_secret": "sk",
            "oss.bucket_name": "bucket",
            "oss.region": "cn-test",
            "oss.endpoint": "oss-cn-test.aliyuncs.com",
            "voiceprint.embedding_provider": "bailian",
            "voiceprint.embedding_endpoint": "http://adb.example:8100/audio/embedding",
        }
    )

    result = runner.invoke(app, ["doctor", "--require-voiceprint-embedding"])

    assert result.exit_code == 0
    assert "Summary: 7 ok, 0 warn, 0 fail" in result.output
    assert "  OK    voiceprint-embedding" in result.output
    assert "provider=bailian" in result.output
    assert "Repair Prompts:" not in result.output


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
    monkeypatch.setattr(doctor, "_check_editor", lambda: doctor.CheckResult("editor", "ok", "test"))


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
    "MEETING_ASR_EDITOR",
)
