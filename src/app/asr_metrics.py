"""Compatibility module alias for ASR runtime metrics."""

from __future__ import annotations

import sys

from app.core import asr_metrics as _impl

sys.modules[__name__] = _impl
