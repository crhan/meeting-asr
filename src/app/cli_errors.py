"""Compatibility module alias for CLI error presentation helpers."""

from __future__ import annotations

import sys

from app.presentation.cli import errors as _impl

sys.modules[__name__] = _impl
