"""Compatibility module alias for speaker TUI status rendering."""

from __future__ import annotations

import sys

from app.presentation.tui import speaker_status as _impl

sys.modules[__name__] = _impl
