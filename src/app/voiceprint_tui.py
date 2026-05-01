"""Compatibility module alias for voiceprint TUI presentation."""

from __future__ import annotations

import sys

from app.presentation.tui import voiceprint as _impl

sys.modules[__name__] = _impl
