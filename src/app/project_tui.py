"""Compatibility module alias for project TUI presentation."""

from __future__ import annotations

import sys

from app.presentation.tui import project as _impl

sys.modules[__name__] = _impl
