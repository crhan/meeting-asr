"""Tests for generated shell completion scripts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.commands.completion import (
    CompletionShell,
    _bash_script,
    _csh_script,
    _detect_cli_bin_dir,
    _fish_script,
    _install_completion,
    _profile_script,
    _zsh_script,
)

runner = CliRunner()


def test_completion_scripts_include_project_commands() -> None:
    """Completion scripts should expose project workflow commands."""
    assert "_MEETING_ASR_COMPLETE=complete_bash" in _bash_script()
    assert "_MEETING_ASR_COMPLETE=complete_zsh" in _zsh_script()
    assert "_MEETING_ASR_COMPLETE=complete_fish" in _fish_script()
    assert "n/project/" in _csh_script()
    assert "n/speakers/" in _csh_script()


def test_profile_script_adds_cli_path() -> None:
    """Installed zsh fragment should prepend the binary directory."""
    script = _profile_script(CompletionShell.zsh, Path("/tmp/meeting-asr-bin"))

    assert "export PATH=/tmp/meeting-asr-bin:$PATH" in script
    assert "_MEETING_ASR_COMPLETE=complete_zsh" in script


def test_detect_cli_bin_dir_preserves_user_facing_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Completion PATH should use ~/.local/bin, not the uv tool private venv bin."""
    user_bin = tmp_path / "home" / ".local" / "bin"
    tool_bin = tmp_path / "home" / ".local" / "share" / "uv" / "tools" / "meeting-asr" / "bin"
    user_bin.mkdir(parents=True)
    tool_bin.mkdir(parents=True)
    tool_executable = tool_bin / "meeting-asr"
    tool_executable.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    user_executable = user_bin / "meeting-asr"
    user_executable.symlink_to(tool_executable)
    monkeypatch.setattr(shutil, "which", lambda command: str(user_executable))

    assert _detect_cli_bin_dir() == user_bin


def test_bash_completion_runtime_uses_command_tree() -> None:
    """Runtime completion should include nested commands and options."""
    root_commands = _bash_complete("meeting-asr ", 1)
    project_commands = _bash_complete("meeting-asr project ", 2)
    speaker_commands = _bash_complete("meeting-asr project speakers ", 3)
    voiceprint_commands = _bash_complete("meeting-asr voiceprint ", 2)
    lexicon_commands = _bash_complete("meeting-asr lexicon ", 2)
    transcribe_options = _bash_complete("meeting-asr project transcribe --", 3)

    assert "audio" not in root_commands
    assert "help" in root_commands
    assert "voiceprint" in root_commands
    assert "lexicon" in root_commands
    assert "list" in project_commands
    assert "review" in project_commands
    assert "transcribe" in project_commands
    assert "transcript" in project_commands
    assert "speakers" in project_commands
    assert "review" in speaker_commands
    assert "browse" in voiceprint_commands
    assert "list" in lexicon_commands
    assert "hotwords" in lexicon_commands
    assert "stats" in lexicon_commands
    assert "--oss-upload" in transcribe_options
    assert "--asr-hotwords" in transcribe_options
    assert "--audio-format" in transcribe_options
    assert all(not item.startswith("plain,") for item in project_commands)


def test_bash_completion_runtime_includes_value_completions() -> None:
    """Runtime completion should expose finite parameter values."""
    upload_modes = _bash_complete("meeting-asr project transcribe --oss-upload ", 4)
    hotword_modes = _bash_complete("meeting-asr project transcribe --asr-hotwords ", 4)
    voiceprint_providers = _bash_complete("meeting-asr voiceprint embed --provider ", 4)
    cli_languages = _bash_complete("meeting-asr --lang ", 2)
    config_keys = _bash_complete("meeting-asr config set dash", 3)
    install_shells = _bash_complete("meeting-asr completion install ", 3)

    assert upload_modes == ["auto", "true", "false"]
    assert hotword_modes == ["auto", "off"]
    assert voiceprint_providers == ["local-speechbrain", "bailian"]
    assert cli_languages == ["auto", "en", "zh"]
    assert "dashscope.api_key" in config_keys
    assert "dashscope.base_url" in config_keys
    assert "dashscope.summary_model" in config_keys
    assert "dashscope.asr_vocabulary_id" in config_keys
    assert "zsh" in install_shells
    assert "csh" not in install_shells


def test_install_completion_writes_target(tmp_path: Path) -> None:
    """Completion install should write a shell fragment without touching rc files."""
    target = tmp_path / "meeting-asr.zsh"
    result = _install_completion(
        shell=CompletionShell.zsh,
        target=target,
        bin_dir=tmp_path / "bin",
        update_rc=False,
    )

    assert result.target_path == target.resolve()
    assert result.rc_updated is False
    assert "_MEETING_ASR_COMPLETE=complete_zsh" in target.read_text(encoding="utf-8")


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


def _bash_complete(comp_words: str, comp_cword: int) -> list[str]:
    """Run Bash-style completion through the Typer test runner."""
    result = runner.invoke(
        app,
        [],
        env={
            "_MEETING_ASR_COMPLETE": "complete_bash",
            "COMP_WORDS": comp_words,
            "COMP_CWORD": str(comp_cword),
        },
        prog_name="meeting-asr",
    )
    assert result.exit_code == 0
    return result.output.splitlines()
