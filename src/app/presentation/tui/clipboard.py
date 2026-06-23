"""Clipboard helpers for terminal TUIs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipboardCopyResult:
    """Result of an OS clipboard copy attempt."""

    copied: bool
    method: str | None = None
    message: str | None = None


def copy_to_system_clipboard(text: str) -> ClipboardCopyResult:
    """Copy text through a platform clipboard command when one is available."""
    if sys.platform == "darwin" and shutil.which("pbcopy"):
        return _run_clipboard_command(["pbcopy"], text, "pbcopy")
    if shutil.which("wl-copy"):
        return _run_clipboard_command(["wl-copy"], text, "wl-copy")
    if shutil.which("xclip"):
        return _run_clipboard_command(
            ["xclip", "-selection", "clipboard"], text, "xclip"
        )
    if shutil.which("xsel"):
        return _run_clipboard_command(["xsel", "--clipboard", "--input"], text, "xsel")
    return ClipboardCopyResult(False, message="no system clipboard command found")


def _run_clipboard_command(
    command: list[str], text: str, method: str
) -> ClipboardCopyResult:
    """Run one clipboard command without invoking a shell."""
    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception as exc:  # noqa: BLE001
        return ClipboardCopyResult(False, method=method, message=str(exc))
    return ClipboardCopyResult(True, method=method)
