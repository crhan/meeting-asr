"""Editor command handling for vocabulary correction workflows."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from app.config import get_configured_editor


def open_editor(path: Path, editor: str | None) -> None:
    """
    Open one file in a blocking editor command.

    Args:
        path: File path to open.
        editor: Optional editor command override.

    Returns:
        None.
    """
    command = editor_command(editor, path)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(_editor_failure_message(command, "command not found")) from exc
    except subprocess.CalledProcessError as exc:
        detail = f"exited with status {exc.returncode}"
        raise RuntimeError(_editor_failure_message(command, detail)) from exc


def editor_command(editor: str | None, path: Path) -> list[str]:
    """
    Build an editor command for one file.

    Args:
        editor: Optional editor command text.
        path: File path to append or inject.

    Returns:
        Command argv list.
    """
    command_text = editor or default_editor()
    parts = shlex.split(command_text)
    if not parts:
        raise ValueError("Editor command must not be empty.")
    file_text = str(path)
    if any("{file}" in part for part in parts):
        return [part.replace("{file}", file_text) for part in parts]
    return parts + [file_text]


def default_editor() -> str:
    """
    Return a practical default editor command.

    Returns:
        Editor command text.
    """
    configured = get_configured_editor() or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if configured:
        return configured
    if shutil.which("code"):
        return "code --wait"
    return "vim"


def _editor_failure_message(command: list[str], detail: str) -> str:
    """Return actionable editor failure guidance."""
    command_text = shlex.join(command)
    return (
        f"Editor failed: {command_text} ({detail}). "
        'Configure it with `meeting-asr config set ui.editor "code --wait"` or pass `--editor`.'
    )
