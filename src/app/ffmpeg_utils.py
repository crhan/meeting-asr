"""Compatibility module alias for ffmpeg infrastructure helpers."""

from __future__ import annotations

import sys

from app.infra import ffmpeg as _impl

sys.modules[__name__] = _impl
