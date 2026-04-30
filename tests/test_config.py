"""Tests for XDG global config handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.config import get_config_path, get_default_projects_dir, load_settings, save_config_values, set_config_value

runner = CliRunner()


def test_config_path_uses_xdg_config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Config path should follow XDG_CONFIG_HOME."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert get_config_path() == tmp_path / "meeting-asr" / "config.json"


def test_default_projects_dir_uses_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default projects should follow XDG_DATA_HOME."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert get_default_projects_dir() == tmp_path / "meeting-asr" / "projects"


def test_relative_xdg_data_home_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """XDG base dirs must be absolute."""
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")

    assert get_default_projects_dir() == Path.home() / ".local" / "share" / "meeting-asr" / "projects"


def test_load_settings_reads_global_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings should be loaded from the XDG global config file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    set_config_value("dashscope.api_key", "config-key")

    settings = load_settings()

    assert settings.dashscope_api_key == "config-key"
    assert settings.dashscope_base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_process_env_overrides_global_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit process env remains useful for CI."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    save_config_values({"dashscope.api_key": "config-key"})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

    assert load_settings().dashscope_api_key == "env-key"


def test_load_settings_does_not_read_cwd_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A working-directory .env must not silently configure the CLI."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _clear_runtime_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=from-dotenv\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required config"):
        load_settings()


def test_config_import_env_command_writes_masked_global_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Users can migrate legacy .env values without printing secrets."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=secret\nOSS_BUCKET_NAME=bucket\n", encoding="utf-8")

    result = runner.invoke(app, ["config", "import-env", str(env_file)])
    show_result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "Imported 2 value(s)" in result.output
    assert "dashscope.api_key=********" in show_result.output
    assert "oss.bucket_name=bucket" in show_result.output


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove config-related process environment variables."""
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "OSS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET",
        "OSS_BUCKET_NAME",
        "OSS_REGION",
        "OSS_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
