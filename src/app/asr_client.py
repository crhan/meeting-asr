"""Compatibility module alias for DashScope ASR infrastructure."""

from __future__ import annotations

import sys

from app.infra import dashscope_asr as _impl

sys.modules[__name__] = _impl
