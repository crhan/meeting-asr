"""Tests for the standalone installer script."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib


def test_install_script_has_valid_bash_syntax() -> None:
    """The standalone installer should parse as Bash."""
    script = Path("scripts/install-tool.sh")

    subprocess.run(["bash", "-n", str(script)], check=True)


def test_install_script_print_only_uses_stable_uv_tool_command() -> None:
    """The normal developer installer should use editable mode."""
    result = subprocess.run(
        ["scripts/install-tool.sh", "--print-only"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "uv tool install --python 3.14" in result.stdout
    assert "--force" not in result.stdout
    assert "--refresh" not in result.stdout
    assert "--reinstall" not in result.stdout
    assert "--editable" in result.stdout
    assert "Web UI: yes ([web] extra, default)" in result.stdout
    assert ".\\[web\\]" in result.stdout
    assert "Local voiceprint: standard dependency" in result.stdout
    assert "local-voiceprint" not in result.stdout


def test_install_script_wheel_mode_is_explicit() -> None:
    """Wheel installation should be an explicit release/user verification mode."""
    result = subprocess.run(
        ["scripts/install-tool.sh", "--print-only", "--wheel"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Mode: wheel" in result.stdout
    assert "--editable" not in result.stdout
    assert "env MEETING_ASR_BUILD_WEB=1 uv tool install" in result.stdout


def test_install_script_force_is_explicit() -> None:
    """Force should be an opt-in escape hatch for executable conflicts."""
    result = subprocess.run(
        ["scripts/install-tool.sh", "--print-only", "--force"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "uv tool install --python 3.14 --force --editable" in result.stdout


def test_pyproject_tracks_source_files_for_uv_cache() -> None:
    """uv should rebuild local wheels when source files change."""
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert {"file": "pyproject.toml"} in payload["tool"]["uv"]["cache-keys"]
    assert {"file": "src/**/*.py"} in payload["tool"]["uv"]["cache-keys"]


def test_install_script_verifies_installed_source_fingerprint() -> None:
    """The installer should fail fast when uv serves a stale local wheel."""
    script = Path("scripts/install-tool.sh").read_text(encoding="utf-8")

    assert "Code match:" in script
    assert "Web UI dependencies:" in script
    assert "it installs the web extra by default" in script
    assert "Installed package code does not match this checkout" in script
    assert "UV_NO_CACHE=1" in script


def test_install_script_verifies_web_assets_not_just_deps() -> None:
    """A web install must prove the SPA bundle is present: deps can be installed while a
    stale cached wheel (or an unbuilt editable checkout) carries no static assets."""
    script = Path("scripts/install-tool.sh").read_text(encoding="utf-8")

    assert "Web UI assets:" in script
    assert 'web" / "static" / "index.html' in script
    assert "are missing from this install" in script


def test_pyproject_busts_uv_cache_on_web_source_changes() -> None:
    """uv decides wheel-cache reuse before the build hook runs, so the SPA *sources* and the
    build flag must be cache keys -- the static build output alone only changes when the hook
    already ran, which a cache hit prevents."""
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    cache_keys = payload["tool"]["uv"]["cache-keys"]

    assert {"file": "web/src/**/*"} in cache_keys
    assert {"file": "web/package.json"} in cache_keys
    assert {"file": "web/package-lock.json"} in cache_keys
    assert {"file": "web/index.html"} in cache_keys
    assert {"file": "web/vite.config.ts"} in cache_keys
    assert {"file": "web/tsconfig.json"} in cache_keys
    assert {"env": "MEETING_ASR_BUILD_WEB"} in cache_keys
