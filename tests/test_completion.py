"""Tests for generated shell completion scripts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from app.commands.completion import (
    _bash_script,
    _csh_script,
    _zsh_profile_script,
    _zsh_script,
)


def test_completion_scripts_include_project_commands() -> None:
    """Completion scripts should expose project workflow commands."""
    assert "doctor config project audio oss completion" in _bash_script()
    assert "project" in _zsh_script()
    assert "n/project/" in _csh_script()


def test_zsh_profile_script_adds_cli_path() -> None:
    """Installed zsh fragment should prepend the binary directory."""
    script = _zsh_profile_script(Path("/tmp/meeting-asr-bin"))

    assert 'export PATH="/tmp/meeting-asr-bin:$PATH"' in script


def test_generated_bash_script_has_valid_syntax(tmp_path: Path) -> None:
    """Generated Bash script should parse."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not installed")
    path = tmp_path / "meeting-asr.bash"
    path.write_text(_bash_script(), encoding="utf-8")
    subprocess.run([bash, "-n", str(path)], check=True)


def test_generated_zsh_script_has_valid_syntax(tmp_path: Path) -> None:
    """Generated zsh script should parse."""
    zsh = shutil.which("zsh")
    if not zsh:
        pytest.skip("zsh is not installed")
    path = tmp_path / "meeting-asr.zsh"
    path.write_text(_zsh_script(), encoding="utf-8")
    subprocess.run([zsh, "-n", str(path)], check=True)
