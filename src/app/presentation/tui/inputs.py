"""Shared TUI input widgets."""

from __future__ import annotations

from textual.binding import Binding
from textual.widgets import Input


class ReadlineInput(Input):
    """Input with common terminal cursor movement keys."""

    BINDINGS = [
        Binding(
            "ctrl+f", "cursor_right", "Move cursor right", show=False, priority=True
        ),
        Binding("ctrl+b", "cursor_left", "Move cursor left", show=False, priority=True),
    ]
