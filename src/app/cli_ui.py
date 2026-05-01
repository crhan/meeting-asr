"""Compatibility module alias for CLI progress presentation helpers."""

from __future__ import annotations

import sys

from app.presentation.cli import progress as _impl

sys.modules[__name__] = _impl
