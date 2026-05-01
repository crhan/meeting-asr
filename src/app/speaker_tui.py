"""Compatibility module alias for speaker TUI presentation."""

from __future__ import annotations

import sys

from app.presentation.tui import speaker as _impl

sys.modules[__name__] = _impl
