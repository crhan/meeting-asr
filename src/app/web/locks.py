"""Async write locks for project artifacts and shared global stores.

Single-worker uvicorn means we only need in-process mutual exclusion, not distributed
locks. Two concerns:

* Per-project writes (project.json, asr/sentences.json, speakers/*) must not interleave
  across concurrent requests on the same project.
* The global voiceprint and lexicon SQLite databases are shared across *all* projects,
  so a per-project lock does not protect them -- two jobs on different projects can race
  on the same global store. Hence a separate per-store lock keyed by store path.

Writers acquire every needed lock through :meth:`LockRegistry.acquire`, which always
locks in sorted key order so two writers requesting the same set never deadlock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class LockRegistry:
    """Lazily-created named :class:`asyncio.Lock` registry."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> asyncio.Lock:
        """Return (creating if needed) the lock for ``key``."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, *keys: str) -> AsyncGenerator[None]:
        """Acquire all named locks in a deadlock-free (sorted) order."""
        ordered = sorted(set(keys))
        acquired: list[asyncio.Lock] = []
        try:
            for key in ordered:
                lock = self.get(key)
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()


def project_lock_key(project_id: str) -> str:
    """Return the lock key for one project's on-disk artifacts."""
    return f"project:{project_id}"


def store_lock_key(label: str) -> str:
    """Return the lock key for one shared global store (e.g. ``voiceprints``)."""
    return f"store:{label}"
