"""Shell completion script generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from textwrap import dedent

import typer

COMMAND_NAMES = ("meeting-asr",)
TOP_LEVEL_COMMANDS = "doctor config project audio oss completion"
PROJECT_COMMANDS = "create prepare transcribe run status git-init speakers --help"
CONFIG_COMMANDS = "path show keys set unset import-env --help"
COMPLETION_OPTIONS = "bash csh tcsh zsh install --help"
INSTALLABLE_SHELLS = {"zsh"}

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Generate or install shell completion scripts.")


@dataclass(frozen=True)
class CompletionInstallResult:
    """Result of installing a shell completion fragment."""

    target_path: Path
    zshrc_updated: bool


@app.command("bash")
def bash_command() -> None:
    """Print a Bash completion script."""
    typer.echo(_bash_script())


@app.command("zsh")
def zsh_command() -> None:
    """Print a zsh completion script."""
    typer.echo(_zsh_script())


@app.command("csh")
def csh_command() -> None:
    """Print a csh completion script."""
    typer.echo(_csh_script())


@app.command("tcsh")
def tcsh_command() -> None:
    """Print a tcsh completion script."""
    typer.echo(_csh_script())


@app.command("install")
def install_command(
    shell: str = typer.Argument("zsh"),
    target: Path | None = typer.Option(None, "--target", "-t"),
    bin_dir: Path | None = typer.Option(None, "--bin-dir"),
    update_zshrc: bool = typer.Option(True, "--update-zshrc/--no-update-zshrc"),
) -> None:
    """Install a static shell completion profile fragment."""
    if shell.strip().lower() not in INSTALLABLE_SHELLS:
        raise typer.BadParameter("Only zsh install is supported.")
    result = _install_zsh_completion(target=target, bin_dir=bin_dir, update_zshrc=update_zshrc)
    typer.echo(f"Installed zsh completion: {result.target_path}")
    if result.zshrc_updated:
        typer.echo("Updated ~/.zshrc to load ~/.zshrc.profile.d/*.zshrc.")
    typer.echo(f"Restart zsh or run: source {result.target_path}")


def _bash_script() -> str:
    """Build a small Bash completion script."""
    return dedent(
        f"""
        _meeting_asr_completion() {{
          local cur="${{COMP_WORDS[COMP_CWORD]}}"
          COMPREPLY=( $(compgen -W "{TOP_LEVEL_COMMANDS}" -- "$cur") )
        }}
        complete -F _meeting_asr_completion {' '.join(COMMAND_NAMES)}
        """
    ).strip()


def _zsh_script() -> str:
    """Build a zsh completion script."""
    return dedent(
        f"""
        #compdef meeting-asr
        _meeting_asr() {{
          local -a commands
          commands=({TOP_LEVEL_COMMANDS})
          if (( CURRENT == 2 )); then
            _describe -t commands 'meeting-asr commands' commands
            return
          fi
          _files
        }}
        compdef _meeting_asr meeting-asr
        """
    ).strip()


def _csh_script() -> str:
    """Build a csh/tcsh completion script."""
    return "\n\n".join(_csh_command(command_name) for command_name in COMMAND_NAMES)


def _csh_command(command_name: str) -> str:
    """Build one csh completion rule."""
    return f"complete {command_name} 'p/1/({TOP_LEVEL_COMMANDS})/' 'n/project/({PROJECT_COMMANDS})/'"


def _install_zsh_completion(target: Path | None, bin_dir: Path | None, update_zshrc: bool) -> CompletionInstallResult:
    """Write the zsh profile fragment."""
    target_path = target.expanduser().resolve() if target else Path.home() / ".zshrc.profile.d" / "80-meeting-asr.zshrc"
    executable_dir = bin_dir.expanduser().resolve() if bin_dir else _detect_cli_bin_dir()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(_zsh_profile_script(executable_dir), encoding="utf-8")
    target_path.chmod(0o644)
    updated = _ensure_zshrc_loads_profile_dir(target_path.parent) if update_zshrc else False
    return CompletionInstallResult(target_path, updated)


def _detect_cli_bin_dir() -> Path:
    """Detect the directory containing meeting-asr."""
    executable = shutil.which(COMMAND_NAMES[0])
    if executable:
        return Path(executable).resolve().parent
    return Path.home() / ".local" / "bin"


def _zsh_profile_script(bin_dir: Path) -> str:
    """Build installed zsh profile fragment."""
    return "\n".join(["#!/bin/zsh", "# meeting-asr CLI", f'export PATH="{bin_dir}:$PATH"', "", _zsh_script(), ""])


def _ensure_zshrc_loads_profile_dir(profile_dir: Path) -> bool:
    """Append a profile.d loader to ~/.zshrc when missing."""
    zshrc_path = Path.home() / ".zshrc"
    existing = zshrc_path.read_text(encoding="utf-8") if zshrc_path.exists() else ""
    if str(profile_dir) in existing or ".zshrc.profile.d/*.zshrc" in existing:
        return False
    with zshrc_path.open("a", encoding="utf-8") as zshrc_file:
        zshrc_file.write(f'\nfor f in "{profile_dir}"/*.zshrc(N); do\n  source "$f"\ndone\n')
    return True
