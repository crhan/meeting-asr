"""Tests for the standalone installer script."""

from __future__ import annotations

from pathlib import Path
import subprocess


def test_install_script_has_valid_bash_syntax() -> None:
    """The standalone installer should parse as Bash."""
    script = Path("scripts/install-tool.sh")

    subprocess.run(["bash", "-n", str(script)], check=True)


def test_install_script_print_only_uses_stable_uv_tool_command() -> None:
    """The installer should encode the uv tool flags that avoid stale Python/cache issues."""
    result = subprocess.run(
        ["scripts/install-tool.sh", "--print-only"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "uv tool install --python 3.14 --force --reinstall --refresh" in result.stdout
    assert "local-voiceprint" in result.stdout
    assert "--editable" not in result.stdout
