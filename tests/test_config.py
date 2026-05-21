"""Tests for XDG global config handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.config import (
    DEFAULT_DASHSCOPE_CORRECTION_CONCURRENCY,
    DEFAULT_DASHSCOPE_CORRECTION_MODEL,
    DEFAULT_DASHSCOPE_SUMMARY_MODEL,
    get_cache_dir,
    get_config_path,
    get_default_projects_dir,
    load_settings,
    save_config_values,
    set_config_value,
)

runner = CliRunner()


def test_config_path_uses_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config path should follow XDG_CONFIG_HOME."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert get_config_path() == tmp_path / "meeting-asr" / "config.json"


def test_default_projects_dir_uses_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default projects should follow XDG_DATA_HOME."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert get_default_projects_dir() == tmp_path / "meeting-asr" / "projects"


def test_cache_dir_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Global model caches should follow XDG_CACHE_HOME."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert get_cache_dir() == tmp_path / "meeting-asr"


def test_relative_xdg_data_home_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XDG base dirs must be absolute."""
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")

    assert (
        get_default_projects_dir()
        == Path.home() / ".local" / "share" / "meeting-asr" / "projects"
    )


def test_load_settings_reads_global_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings should be loaded from the XDG global config file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    set_config_value("dashscope.api_key", "config-key")

    settings = load_settings()

    assert settings.dashscope_api_key == "config-key"
    assert settings.dashscope_base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert settings.dashscope_summary_model == DEFAULT_DASHSCOPE_SUMMARY_MODEL
    assert settings.dashscope_correction_model == DEFAULT_DASHSCOPE_CORRECTION_MODEL
    assert settings.dashscope_model_endpoints == {}
    assert (
        settings.dashscope_correction_concurrency
        == DEFAULT_DASHSCOPE_CORRECTION_CONCURRENCY
    )
    assert settings.dashscope_asr_vocabulary_id is None
    assert settings.correction_polish_auto_accept is False


def test_ui_editor_config_is_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The editor used by correction review should be configurable."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    set_config_value("ui.editor", "code --wait")

    settings = load_settings(require_dashscope=False)
    keys_result = runner.invoke(app, ["config", "keys"])
    show_result = runner.invoke(app, ["config", "show"])

    assert settings.ui_editor == "code --wait"
    assert "ui.editor" in keys_result.output
    assert "ui.editor=code --wait" in show_result.output


def test_model_endpoint_routes_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Model endpoint routing should be configurable through XDG config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    set_config_value("dashscope.api_key", "config-key")
    set_config_value(
        "dashscope.model_endpoints",
        '{"qwen3.6-*":"compatible","qwen-plus":"generation"}',
    )

    settings = load_settings()

    assert settings.dashscope_model_endpoints == {
        "qwen3.6-*": "compatible",
        "qwen-plus": "generation",
    }


def test_polish_auto_accept_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transcript polish auto-accept should be opt-in through config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    set_config_value("correction.polish_auto_accept", "true")

    settings = load_settings(require_dashscope=False)
    keys_result = runner.invoke(app, ["config", "keys"])
    show_result = runner.invoke(app, ["config", "show"])

    assert settings.correction_polish_auto_accept is True
    assert "correction.polish_auto_accept" in keys_result.output
    assert "correction.polish_auto_accept=true" in show_result.output


def test_config_show_prints_masked_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config show should have a script-friendly JSON mode without leaking secrets."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    save_config_values({"dashscope.api_key": "secret", "ui.editor": "vim"})

    result = runner.invoke(app, ["config", "show", "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["config_file"] == str(get_config_path())
    assert payload["revealed"] is False
    assert payload["values"]["dashscope.api_key"] == "********"
    assert payload["values"]["ui.editor"] == "vim"


def test_load_settings_without_dashscope_keeps_voiceprint_local_implicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local voiceprint embedding is implicit and does not require DashScope credentials."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)

    settings = load_settings(require_dashscope=False)

    assert settings.dashscope_api_key == ""
    assert not hasattr(settings, "voiceprint_embedding_provider")


def test_process_env_overrides_global_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit process env remains useful for CI."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _clear_runtime_env(monkeypatch)
    save_config_values({"dashscope.api_key": "config-key"})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

    assert load_settings().dashscope_api_key == "env-key"


def test_load_settings_does_not_read_cwd_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    env_file.write_text(
        "DASHSCOPE_API_KEY=secret\nOSS_BUCKET_NAME=bucket\n", encoding="utf-8"
    )

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
        "DASHSCOPE_CORRECTION_MODEL",
        "DASHSCOPE_MODEL_ENDPOINTS",
        "DASHSCOPE_CORRECTION_CONCURRENCY",
        "DASHSCOPE_ASR_VOCABULARY_ID",
        "OSS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET",
        "OSS_BUCKET_NAME",
        "OSS_REGION",
        "OSS_ENDPOINT",
        "MEETING_ASR_EDITOR",
    ):
        monkeypatch.delenv(name, raising=False)
