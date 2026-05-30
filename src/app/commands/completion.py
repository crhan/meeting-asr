"""Shell completion script generation and installation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import shlex
import shutil

import typer
from typer.completion import get_completion_script
from typer.core import TyperGroup
from typer.main import get_command

from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper

COMMAND_NAME = "meeting-asr"
COMPLETE_VAR = "_MEETING_ASR_COMPLETE"
ZSH_PROFILE_LOADER = ".zshrc.profile.d/*.zshrc"
BASH_COMPLETION_LOADER = ".bash_completion.d/*.bash"
RC_MARKER_PREFIX = "# meeting-asr completion"


class CompletionShell(str, Enum):
    """Shells with generated completion scripts."""

    bash = "bash"
    zsh = "zsh"
    fish = "fish"
    powershell = "powershell"
    pwsh = "pwsh"
    csh = "csh"
    tcsh = "tcsh"


class InstallShell(str, Enum):
    """Shells supported by ``completion install``."""

    bash = "bash"
    zsh = "zsh"
    fish = "fish"
    powershell = "powershell"
    pwsh = "pwsh"


INSTALLABLE_SHELLS = {
    CompletionShell.bash,
    CompletionShell.zsh,
    CompletionShell.fish,
    CompletionShell.powershell,
    CompletionShell.pwsh,
}

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    help="Generate or install shell completion scripts.",
)


@dataclass(frozen=True)
class CompletionInstallResult:
    """Result of installing a shell completion fragment."""

    shell: CompletionShell
    target_path: Path
    rc_path: Path | None
    rc_updated: bool


@app.command("bash")
def bash_command() -> None:
    """Print a Bash completion script."""
    typer.echo(_bash_script())


@app.command("zsh")
def zsh_command() -> None:
    """Print a zsh completion script."""
    typer.echo(_zsh_script())


@app.command("fish")
def fish_command() -> None:
    """Print a fish completion script."""
    typer.echo(_fish_script())


@app.command("powershell")
def powershell_command() -> None:
    """Print a PowerShell completion script."""
    typer.echo(_powershell_script())


@app.command("pwsh")
def pwsh_command() -> None:
    """Print a pwsh completion script."""
    typer.echo(_pwsh_script())


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
    shell: InstallShell = typer.Argument(InstallShell.zsh),
    target: Path | None = typer.Option(None, "--target", "-t"),
    bin_dir: Path | None = typer.Option(None, "--bin-dir"),
    update_rc: bool = typer.Option(
        True,
        "--update-rc/--no-update-rc",
        "--update-zshrc/--no-update-zshrc",
    ),
) -> None:
    """Install a shell completion fragment."""
    result = _install_completion(
        shell=CompletionShell(shell.value),
        target=target,
        bin_dir=bin_dir,
        update_rc=update_rc,
    )
    typer.echo(f"Installed {result.shell.value} completion: {result.target_path}")
    if result.rc_updated and result.rc_path:
        typer.echo(f"Updated shell startup file: {result.rc_path}")
    typer.echo(_activation_hint(result))


def _bash_script() -> str:
    """Build a Bash completion script from the Typer command tree."""
    return _typer_completion_script(CompletionShell.bash)


def _zsh_script() -> str:
    """Build a zsh completion script from the Typer command tree."""
    return _typer_completion_script(CompletionShell.zsh)


def _fish_script() -> str:
    """Build a fish completion script from the Typer command tree."""
    return _typer_completion_script(CompletionShell.fish)


def _powershell_script() -> str:
    """Build a PowerShell completion script from the Typer command tree."""
    return _typer_completion_script(CompletionShell.powershell)


def _pwsh_script() -> str:
    """Build a pwsh completion script from the Typer command tree."""
    return _typer_completion_script(CompletionShell.pwsh)


def _typer_completion_script(shell: CompletionShell) -> str:
    """
    Build a dynamic completion script.

    Args:
        shell: Target shell.

    Returns:
        Completion script that asks the CLI for command-tree completions.
    """
    return get_completion_script(
        prog_name=COMMAND_NAME, complete_var=COMPLETE_VAR, shell=shell.value
    )


def _csh_script() -> str:
    """Build a csh/tcsh completion script from the command tree."""
    return "\n\n".join(_csh_command(command_name) for command_name in _command_names())


def _csh_command(command_name: str) -> str:
    """
    Build one csh completion rule.

    Args:
        command_name: Program name.

    Returns:
        A csh ``complete`` directive.
    """
    root = _root_command()
    rules = [f"'p/1/({_words(_subcommands(root))})/'"]
    for group_name, subcommands in _nested_subcommand_rules(root):
        rules.append(f"'n/{group_name}/({_words(subcommands)})/'")
    return f"complete {command_name} " + " ".join(rules)


def _install_completion(
    *,
    shell: CompletionShell,
    target: Path | None,
    bin_dir: Path | None,
    update_rc: bool,
) -> CompletionInstallResult:
    """
    Install completion for one shell.

    Args:
        shell: Target shell.
        target: Optional explicit install path.
        bin_dir: Optional directory containing the executable.
        update_rc: Whether to source the fragment from shell startup files.

    Returns:
        Installation result.
    """
    if shell not in INSTALLABLE_SHELLS:
        raise typer.BadParameter(
            "Install is supported for bash, zsh, fish, powershell, and pwsh."
        )
    target_path = (
        target.expanduser().resolve() if target else _default_completion_target(shell)
    )
    executable_dir = (
        bin_dir.expanduser().resolve() if bin_dir else _detect_cli_bin_dir()
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(_profile_script(shell, executable_dir), encoding="utf-8")
    target_path.chmod(0o644)
    rc_path = _startup_file(shell)
    rc_updated = (
        _ensure_startup_loads_completion(shell, target_path, rc_path)
        if update_rc and rc_path
        else False
    )
    return CompletionInstallResult(
        shell=shell, target_path=target_path, rc_path=rc_path, rc_updated=rc_updated
    )


def _default_completion_target(shell: CompletionShell) -> Path:
    """
    Return the default user-level install path for a shell.

    Args:
        shell: Target shell.

    Returns:
        Default completion script path.
    """
    home = Path.home()
    if shell == CompletionShell.bash:
        return home / ".bash_completion.d" / "meeting-asr.bash"
    if shell == CompletionShell.zsh:
        return home / ".zshrc.profile.d" / "80-meeting-asr.zshrc"
    if shell == CompletionShell.fish:
        return home / ".config" / "fish" / "completions" / "meeting-asr.fish"
    return home / ".config" / "powershell" / "meeting-asr-completion.ps1"


def _detect_cli_bin_dir() -> Path:
    """
    Detect the user-facing directory containing ``meeting-asr``.

    Returns:
        Executable directory, or the user bin fallback.
    """
    executable = shutil.which(COMMAND_NAME)
    if executable:
        return Path(executable).expanduser().parent
    return Path.home() / ".local" / "bin"


def _profile_script(shell: CompletionShell, bin_dir: Path) -> str:
    """
    Build an installed completion fragment with PATH setup.

    Args:
        shell: Target shell.
        bin_dir: Directory containing the executable.

    Returns:
        Shell script fragment.
    """
    lines = [
        _script_header(shell),
        _path_export(shell, bin_dir),
        "",
        _completion_script(shell),
        "",
    ]
    return "\n".join(lines)


def _completion_script(shell: CompletionShell) -> str:
    """
    Return the completion script for a shell.

    Args:
        shell: Target shell.

    Returns:
        Completion script text.
    """
    if shell in {CompletionShell.csh, CompletionShell.tcsh}:
        return _csh_script()
    return _typer_completion_script(shell)


def _script_header(shell: CompletionShell) -> str:
    """
    Return a script header for installed fragments.

    Args:
        shell: Target shell.

    Returns:
        Header text.
    """
    if shell == CompletionShell.fish:
        return "# meeting-asr CLI completion"
    if shell in {CompletionShell.powershell, CompletionShell.pwsh}:
        return "# meeting-asr CLI completion"
    return f"#!/bin/{shell.value}\n# meeting-asr CLI completion"


def _path_export(shell: CompletionShell, bin_dir: Path) -> str:
    """
    Build a shell-specific PATH prepend line.

    Args:
        shell: Target shell.
        bin_dir: Directory to prepend.

    Returns:
        Shell code that updates PATH.
    """
    path = str(bin_dir)
    if shell == CompletionShell.fish:
        return f"set -gx PATH {_fish_quote(path)} $PATH"
    if shell in {CompletionShell.powershell, CompletionShell.pwsh}:
        return f"$env:Path = {_powershell_quote(path)} + [System.IO.Path]::PathSeparator + $env:Path"
    return f"export PATH={shlex.quote(path)}:$PATH"


def _startup_file(shell: CompletionShell) -> Path | None:
    """
    Return the startup file to update for a shell.

    Args:
        shell: Target shell.

    Returns:
        Startup file path, or ``None`` when the shell auto-loads completions.
    """
    home = Path.home()
    if shell == CompletionShell.zsh:
        return home / ".zshrc"
    if shell == CompletionShell.bash:
        return home / ".bashrc"
    if shell in {CompletionShell.powershell, CompletionShell.pwsh}:
        return home / ".config" / "powershell" / "Microsoft.PowerShell_profile.ps1"
    return None


def _ensure_startup_loads_completion(
    shell: CompletionShell, target_path: Path, rc_path: Path
) -> bool:
    """
    Ensure a startup file sources the installed completion fragment.

    Args:
        shell: Target shell.
        target_path: Completion fragment path.
        rc_path: Startup file path.

    Returns:
        Whether the startup file changed.
    """
    existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
    loader = _startup_loader(shell, target_path)
    marker = f"{RC_MARKER_PREFIX}: {shell.value}"
    if (
        marker in existing
        or str(target_path) in existing
        or _legacy_loader_present(shell, existing)
    ):
        return False
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    with rc_path.open("a", encoding="utf-8") as rc_file:
        rc_file.write(f"\n{marker}\n{loader}\n")
    return True


def _startup_loader(shell: CompletionShell, target_path: Path) -> str:
    """
    Build startup-file code that loads a completion fragment.

    Args:
        shell: Target shell.
        target_path: Completion fragment path.

    Returns:
        Startup-file loader snippet.
    """
    if shell == CompletionShell.zsh:
        profile_dir = target_path.parent
        return f'for f in "{profile_dir}"/*.zshrc(N); do\n  source "$f"\ndone'
    if shell == CompletionShell.bash:
        return (
            f'for f in "{target_path.parent}"/*.bash; do\n  [ -r "$f" ] && . "$f"\ndone'
        )
    return f". {_powershell_quote(str(target_path))}"


def _legacy_loader_present(shell: CompletionShell, existing: str) -> bool:
    """
    Detect old profile loader snippets to avoid duplicates.

    Args:
        shell: Target shell.
        existing: Startup file content.

    Returns:
        Whether a compatible loader is already present.
    """
    if shell == CompletionShell.zsh:
        return ZSH_PROFILE_LOADER in existing
    if shell == CompletionShell.bash:
        return BASH_COMPLETION_LOADER in existing
    return False


def _activation_hint(result: CompletionInstallResult) -> str:
    """
    Build post-install activation guidance.

    Args:
        result: Installation result.

    Returns:
        One-line activation hint.
    """
    if result.shell == CompletionShell.fish:
        return "Restart fish or open a new shell."
    if result.shell in {CompletionShell.powershell, CompletionShell.pwsh}:
        return (
            f"Restart PowerShell or run: . {_powershell_quote(str(result.target_path))}"
        )
    return f"Restart {result.shell.value} or run: source {shlex.quote(str(result.target_path))}"


def _root_command() -> Any:
    """
    Return the root Typer command backing the app.

    Returns:
        Root Typer command (``typer.core.TyperGroup``).
    """
    from app.cli import app as root_app

    return get_command(root_app)


def _command_names() -> tuple[str, ...]:
    """
    Return installed command names.

    Returns:
        Program names supported by generated scripts.
    """
    return (COMMAND_NAME,)


def _nested_subcommand_rules(
    command: Any,
) -> list[tuple[str, tuple[str, ...]]]:
    """
    Return subcommand lists for each nested Typer group.

    Args:
        command: Root command.

    Returns:
        Pairs of group name and subcommand names.
    """
    if not isinstance(command, TyperGroup):
        return []
    rules: list[tuple[str, tuple[str, ...]]] = []
    for name, child in command.commands.items():
        if isinstance(child, TyperGroup):
            rules.append((name, _subcommands(child)))
            rules.extend(_nested_subcommand_rules(child))
    return rules


def _subcommands(command: Any) -> tuple[str, ...]:
    """
    Return visible subcommand names for a Typer command.

    Args:
        command: Typer command or group.

    Returns:
        Subcommand names plus ``--help``.
    """
    if not isinstance(command, TyperGroup):
        return ("--help",)
    return tuple(
        name for name, child in command.commands.items() if not child.hidden
    ) + ("--help",)


def _words(values: tuple[str, ...]) -> str:
    """
    Join csh completion words.

    Args:
        values: Completion words.

    Returns:
        Space-separated word list.
    """
    return " ".join(values)


def _fish_quote(value: str) -> str:
    """
    Quote a string for fish shell.

    Args:
        value: Raw value.

    Returns:
        Single-quoted fish string.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _powershell_quote(value: str) -> str:
    """
    Quote a string for PowerShell.

    Args:
        value: Raw value.

    Returns:
        Single-quoted PowerShell string.
    """
    return "'" + value.replace("'", "''") + "'"
