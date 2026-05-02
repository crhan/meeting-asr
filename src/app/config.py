"""Global XDG configuration loading for Meeting-ASR."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_CONFIG_DIR = "meeting-asr"
CONFIG_FILENAME = "config.json"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_DASHSCOPE_SUMMARY_MODEL = "qwen-plus"
DEFAULT_DASHSCOPE_CORRECTION_MODEL = DEFAULT_DASHSCOPE_SUMMARY_MODEL
DEFAULT_VOICEPRINT_EMBEDDING_PROVIDER = "local-speechbrain"


@dataclass(frozen=True, slots=True)
class ConfigKey:
    """One supported global configuration key."""

    name: str
    field_name: str
    env_name: str
    secret: bool = False
    default: str | None = None


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from global config plus process environment."""

    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_summary_model: str = DEFAULT_DASHSCOPE_SUMMARY_MODEL
    dashscope_correction_model: str = DEFAULT_DASHSCOPE_CORRECTION_MODEL
    dashscope_asr_vocabulary_id: str | None = None
    oss_access_key_id: str | None = None
    oss_access_key_secret: str | None = None
    oss_bucket_name: str | None = None
    oss_region: str | None = None
    oss_endpoint: str | None = None
    voiceprint_embedding_endpoint: str | None = None
    voiceprint_embedding_provider: str = DEFAULT_VOICEPRINT_EMBEDDING_PROVIDER
    ui_editor: str | None = None
    config_path: Path | None = None


CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey("dashscope.api_key", "dashscope_api_key", "DASHSCOPE_API_KEY", secret=True),
    ConfigKey("dashscope.base_url", "dashscope_base_url", "DASHSCOPE_BASE_URL", default=DEFAULT_DASHSCOPE_BASE_URL),
    ConfigKey(
        "dashscope.summary_model",
        "dashscope_summary_model",
        "DASHSCOPE_SUMMARY_MODEL",
        default=DEFAULT_DASHSCOPE_SUMMARY_MODEL,
    ),
    ConfigKey(
        "dashscope.correction_model",
        "dashscope_correction_model",
        "DASHSCOPE_CORRECTION_MODEL",
        default=DEFAULT_DASHSCOPE_CORRECTION_MODEL,
    ),
    ConfigKey("dashscope.asr_vocabulary_id", "dashscope_asr_vocabulary_id", "DASHSCOPE_ASR_VOCABULARY_ID"),
    ConfigKey("oss.access_key_id", "oss_access_key_id", "OSS_ACCESS_KEY_ID", secret=True),
    ConfigKey("oss.access_key_secret", "oss_access_key_secret", "OSS_ACCESS_KEY_SECRET", secret=True),
    ConfigKey("oss.bucket_name", "oss_bucket_name", "OSS_BUCKET_NAME"),
    ConfigKey("oss.region", "oss_region", "OSS_REGION"),
    ConfigKey("oss.endpoint", "oss_endpoint", "OSS_ENDPOINT"),
    ConfigKey("voiceprint.embedding_endpoint", "voiceprint_embedding_endpoint", "VOICEPRINT_EMBEDDING_ENDPOINT"),
    ConfigKey(
        "voiceprint.embedding_provider",
        "voiceprint_embedding_provider",
        "VOICEPRINT_EMBEDDING_PROVIDER",
        default=DEFAULT_VOICEPRINT_EMBEDDING_PROVIDER,
    ),
    ConfigKey("ui.editor", "ui_editor", "MEETING_ASR_EDITOR"),
)

_KEYS_BY_NAME = {item.name: item for item in CONFIG_KEYS}
_KEYS_BY_FIELD = {item.field_name: item for item in CONFIG_KEYS}
_KEYS_BY_ENV = {item.env_name: item for item in CONFIG_KEYS}


def get_config_path() -> Path:
    """
    Return the XDG-compliant global config path.

    Returns:
        ``$XDG_CONFIG_HOME/meeting-asr/config.json`` or fallback.
    """
    return _xdg_base_dir("XDG_CONFIG_HOME", Path.home() / ".config") / APP_CONFIG_DIR / CONFIG_FILENAME


def get_data_dir() -> Path:
    """
    Return the XDG-compliant global data directory.

    Returns:
        ``$XDG_DATA_HOME/meeting-asr`` or fallback.
    """
    return _xdg_base_dir("XDG_DATA_HOME", Path.home() / ".local" / "share") / APP_CONFIG_DIR


def get_cache_dir() -> Path:
    """
    Return the XDG-compliant global cache directory.

    Returns:
        ``$XDG_CACHE_HOME/meeting-asr`` or fallback.
    """
    return _xdg_base_dir("XDG_CACHE_HOME", Path.home() / ".cache") / APP_CONFIG_DIR


def get_default_projects_dir() -> Path:
    """
    Return the default XDG data directory for projects.

    Returns:
        Global projects parent directory.
    """
    return get_data_dir() / "projects"


def load_config_values(path: Path | None = None) -> dict[str, str]:
    """
    Load raw config values from the global config file.

    Args:
        path: Optional config path override.

    Returns:
        Config values keyed by public names.
    """
    config_path = path or get_config_path()
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return _normalize_config_payload(payload)


def save_config_values(values: dict[str, str], path: Path | None = None) -> Path:
    """
    Save config values with restrictive permissions.

    Args:
        values: Config values.
        path: Optional config path override.

    Returns:
        Written config path.
    """
    config_path = path or get_config_path()
    normalized_values = _normalize_config_payload(values)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.chmod(0o700)
    config_path.write_text(json.dumps(normalized_values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.chmod(0o600)
    return config_path


def set_config_value(key: str, value: str, path: Path | None = None) -> tuple[str, Path]:
    """
    Set one global config value.

    Args:
        key: Public, field, or environment key.
        value: Value to persist.
        path: Optional config path override.

    Returns:
        Normalized key and written path.
    """
    normalized_key = normalize_config_key(key)
    values = load_config_values(path)
    values[normalized_key] = value.strip()
    return normalized_key, save_config_values(values, path)


def unset_config_value(key: str, path: Path | None = None) -> tuple[str, Path]:
    """
    Remove one global config value.

    Args:
        key: Public, field, or environment key.
        path: Optional config path override.

    Returns:
        Normalized key and written path.
    """
    normalized_key = normalize_config_key(key)
    values = load_config_values(path)
    values.pop(normalized_key, None)
    return normalized_key, save_config_values(values, path)


def normalize_config_key(key: str) -> str:
    """
    Normalize public, field, or environment variable keys.

    Args:
        key: Key to normalize.

    Returns:
        Public dotted config key.
    """
    cleaned = key.strip()
    if cleaned in _KEYS_BY_NAME:
        return cleaned
    if cleaned in _KEYS_BY_FIELD:
        return _KEYS_BY_FIELD[cleaned].name
    upper = cleaned.upper()
    if upper in _KEYS_BY_ENV:
        return _KEYS_BY_ENV[upper].name
    supported = ", ".join(item.name for item in CONFIG_KEYS)
    raise ValueError(f"Unsupported config key: {key}. Supported keys: {supported}")


def load_settings(*, require_oss: bool = False, require_dashscope: bool = True) -> Settings:
    """
    Load runtime settings from global config and process environment.

    Args:
        require_oss: Whether OSS values must be present.
        require_dashscope: Whether the DashScope API key must be present.

    Returns:
        Runtime settings.
    """
    values = load_config_values()
    return Settings(
        dashscope_api_key=_read_value(values, "dashscope.api_key", required=require_dashscope) or "",
        dashscope_base_url=_read_value(values, "dashscope.base_url", required=False) or DEFAULT_DASHSCOPE_BASE_URL,
        dashscope_summary_model=(
            _read_value(values, "dashscope.summary_model", required=False) or DEFAULT_DASHSCOPE_SUMMARY_MODEL
        ),
        dashscope_correction_model=(
            _read_value(values, "dashscope.correction_model", required=False) or DEFAULT_DASHSCOPE_CORRECTION_MODEL
        ),
        dashscope_asr_vocabulary_id=_read_value(values, "dashscope.asr_vocabulary_id", required=False),
        oss_access_key_id=_read_value(values, "oss.access_key_id", required=require_oss),
        oss_access_key_secret=_read_value(values, "oss.access_key_secret", required=require_oss),
        oss_bucket_name=_read_value(values, "oss.bucket_name", required=require_oss),
        oss_region=_read_value(values, "oss.region", required=require_oss),
        oss_endpoint=_read_value(values, "oss.endpoint", required=require_oss),
        voiceprint_embedding_endpoint=_read_value(values, "voiceprint.embedding_endpoint", required=False),
        voiceprint_embedding_provider=(
            _read_value(values, "voiceprint.embedding_provider", required=False)
            or DEFAULT_VOICEPRINT_EMBEDDING_PROVIDER
        ),
        ui_editor=_read_value(values, "ui.editor", required=False),
        config_path=get_config_path(),
    )


def get_configured_editor(path: Path | None = None) -> str | None:
    """
    Return the configured editor command without requiring cloud credentials.

    Args:
        path: Optional config path override.

    Returns:
        Editor command, or None when unset.
    """
    values = load_config_values(path)
    return _read_value(values, "ui.editor", required=False)


def visible_config_items(*, reveal: bool = False, path: Path | None = None) -> list[tuple[str, str]]:
    """
    Build display-safe config items.

    Args:
        reveal: Show secrets when true.
        path: Optional config path override.

    Returns:
        Ordered key/value tuples.
    """
    values = load_config_values(path)
    items: list[tuple[str, str]] = []
    for config_key in CONFIG_KEYS:
        value = _read_value(values, config_key.name, required=False)
        items.append((config_key.name, _display_value(config_key, value, reveal=reveal)))
    return items


def import_env_file(path: Path, *, overwrite: bool = False) -> tuple[int, Path]:
    """
    Import legacy dotenv-style config into the global config file.

    Args:
        path: Source dotenv file.
        overwrite: Replace existing values.

    Returns:
        Imported count and written config path.
    """
    from dotenv import dotenv_values

    source = path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Env file does not exist: {source}")
    values = load_config_values()
    count = 0
    for raw_key, raw_value in dotenv_values(source).items():
        if raw_value is None:
            continue
        normalized_key = _key_from_env_name(raw_key)
        if normalized_key is None:
            continue
        if normalized_key in values and not overwrite:
            continue
        values[normalized_key] = raw_value.strip()
        count += 1
    return count, save_config_values(values)


def _read_value(config_values: dict[str, str], key: str, *, required: bool) -> str | None:
    """Read one value from process env, global config, or default."""
    config_key = _KEYS_BY_NAME[key]
    value = os.getenv(config_key.env_name)
    if value is None:
        value = config_values.get(key)
    if value is None:
        value = config_key.default
    value = value.strip() if value is not None else None
    if required and not value:
        raise ValueError(_missing_config_message(config_key))
    return value or None


def _xdg_base_dir(env_name: str, fallback: Path) -> Path:
    """Return an absolute XDG base directory or its fallback."""
    raw_value = os.getenv(env_name)
    if not raw_value:
        return fallback
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else fallback


def _normalize_config_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Normalize config keys and drop empty values."""
    normalized: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = normalize_config_key(str(raw_key))
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            normalized[key] = value
    return normalized


def _key_from_env_name(env_name: str) -> str | None:
    """Return a public config key for a legacy environment variable."""
    config_key = _KEYS_BY_ENV.get(env_name.strip().upper())
    return config_key.name if config_key else None


def _display_value(config_key: ConfigKey, value: str | None, *, reveal: bool) -> str:
    """Mask secrets for terminal display."""
    if not value:
        return "<unset>"
    if config_key.secret and not reveal:
        return "********"
    return value


def _missing_config_message(config_key: ConfigKey) -> str:
    """Build an actionable missing-config message."""
    return (
        f"Missing required config: {config_key.name}. "
        f"Run `meeting-asr config set {config_key.name} <value>` "
        f"or set {config_key.env_name} in the process environment."
    )
