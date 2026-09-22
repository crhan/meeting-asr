"""Process-wide torch thread budget for the local embedding models.

Why this exists: CAM++ / ECAPA are tiny models embedding a few seconds of audio,
yet torch defaults its intra-op pool to every core on the box. On a shared host
that already carries other load, eight OpenMP workers spin-waiting on each other
made one 9-second clip take 5-7 s instead of 0.2 s (measured 2026-09-22 on an
8-core host at load ~7: 8 threads 5.0-7.4 s, 4 threads 0.5 s, 2 threads 0.18 s,
1 thread 0.25 s). A 3,000-embedding stabilization run turned into 3+ hours that
way. The callers that embed in bulk also fan out over a small thread pool, so the
per-call default here is one intra-op thread; parallelism comes from the pool.

Configurable via ``voiceprint.torch_threads`` for hosts where more helps.
"""

from __future__ import annotations

import logging
from threading import Lock

from app.config import get_configured_torch_threads

LOGGER = logging.getLogger(__name__)

DEFAULT_TORCH_THREADS = 1

_APPLIED: int | None = None
_LOCK = Lock()


def resolve_torch_threads() -> int:
    """Return the configured intra-op thread count, else the built-in default."""
    configured = get_configured_torch_threads()
    return configured if configured is not None else DEFAULT_TORCH_THREADS


def configure_torch_threads() -> int:
    """
    Apply the intra-op thread budget to torch once per process.

    Idempotent: the first call sets ``torch.set_num_threads``; later calls return
    the value already applied without touching torch again.

    Returns:
        The thread count in effect.
    """
    global _APPLIED
    if _APPLIED is not None:
        return _APPLIED
    with _LOCK:
        if _APPLIED is not None:
            return _APPLIED
        import torch

        threads = resolve_torch_threads()
        torch.set_num_threads(threads)
        LOGGER.debug("torch intra-op threads set to %s", threads)
        _APPLIED = threads
        return threads


__all__ = ["DEFAULT_TORCH_THREADS", "configure_torch_threads", "resolve_torch_threads"]
