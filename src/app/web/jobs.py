"""In-process job queue for long-running workflow operations.

A single-user local tool does not need Celery/Redis. Long operations (ASR intake,
embedding, matching, polish) are submitted here, run in the default thread-pool executor
so they never block the event loop, and stream progress to any number of SSE subscribers.

Jobs for the *same* project are serialised through the shared :class:`LockRegistry` --
the very same per-project lock the inline mutating routes (speaker save, correction
accept) take -- so a background pipeline run cannot interleave with an inline save (or a
second run) on one project; different projects run concurrently. Progress events are
buffered on the job (so a late/reconnecting subscriber can replay history) and fanned out
live to every subscriber queue.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from dataclasses import dataclass, field

from app.core.progress import CliProgressReporter
from app.web.locks import LockRegistry, project_lock_key
from app.web.progress_bridge import QueueProgressReporter

# A unit of work: given a progress reporter, do the blocking work and return a result.
JobFn = Callable[[CliProgressReporter], Any]

_SENTINEL: dict[str, object] = {"type": "end"}


@dataclass(slots=True)
class Job:
    """One queued or running background operation."""

    id: str
    kind: str
    project_id: str | None
    status: str = "queued"  # queued | running | done | error
    result: Any = None
    error: str | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, object]]] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def public(self) -> dict[str, object]:
        """Return a JSON-safe snapshot for the jobs API."""
        return {
            "id": self.id,
            "kind": self.kind,
            "project_id": self.project_id,
            "status": self.status,
            "error": self.error,
            "event_count": len(self.events),
        }


class JobManager:
    """Owns background jobs and their per-project serialisation.

    Per-project serialisation is delegated to the shared :class:`LockRegistry` so jobs and
    inline mutating routes for the same project contend on one lock, not two disjoint pools.
    """

    def __init__(self, locks: LockRegistry) -> None:
        self._jobs: dict[str, Job] = {}
        self._locks = locks
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop at startup.

        Jobs are often submitted from FastAPI's sync-route worker threads, which have no
        running loop; scheduling work onto the captured loop via ``call_soon_threadsafe``
        is the thread-safe way to spawn the job task from anywhere.
        """
        self._loop = loop

    def get(self, job_id: str) -> Job | None:
        """Return a job by id, or None."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """Return all known jobs (newest-tracked order is not guaranteed)."""
        return list(self._jobs.values())

    def submit(
        self,
        kind: str,
        fn: JobFn,
        *,
        project_id: str | None = None,
        store_locks: Sequence[str] = (),
    ) -> Job:
        """Schedule ``fn`` to run in the background and return its :class:`Job`.

        ``store_locks`` names shared-store locks (e.g. ``store_lock_key("lexicon")``) the
        job must hold for its whole duration -- a pipeline run learns into the lexicon and
        deletes global voiceprint samples during stabilization, so it has to contend on the
        same store locks the inline lexicon/voiceprint routes take, not just the per-project
        lock. Acquired together with the project lock in deadlock-free sorted order.
        """
        job = Job(id=uuid.uuid4().hex, kind=kind, project_id=project_id)
        self._jobs[job.id] = job
        loop = self._loop
        if loop is None:
            raise RuntimeError("JobManager loop not bound; call bind_loop at startup.")
        keys = tuple(store_locks)
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._run(job, fn, keys))
        )
        return job

    async def _run(self, job: Job, fn: JobFn, store_locks: tuple[str, ...]) -> None:
        keys: list[str] = []
        if job.project_id is not None:
            # Same key the inline routes use, so jobs and inline saves serialise together.
            keys.append(project_lock_key(job.project_id))
        keys.extend(store_locks)
        if keys:
            # acquire() sorts keys, so project + store locks never deadlock across writers.
            async with self._locks.acquire(*keys):
                await self._execute(job, fn)
        else:
            await self._execute(job, fn)

    async def _execute(self, job: Job, fn: JobFn) -> None:
        loop = asyncio.get_running_loop()
        job.status = "running"
        self._publish(job, {"type": "status", "status": "running"})
        reporter = QueueProgressReporter(
            loop, lambda payload: self._publish(job, payload)
        )
        try:
            job.result = await loop.run_in_executor(None, fn, reporter)
            job.status = "done"
            self._publish(job, {"type": "status", "status": "done"})
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the client
            job.status = "error"
            job.error = str(exc) or exc.__class__.__name__
            traceback.print_exc()
            self._publish(
                job, {"type": "status", "status": "error", "error": job.error}
            )
        finally:
            job.done.set()
            self._publish(job, _SENTINEL)

    def _publish(self, job: Job, payload: dict[str, object]) -> None:
        """Append to history and fan out to subscribers. Runs on the loop thread."""
        job.events.append(payload)
        for queue in job.subscribers:
            queue.put_nowait(payload)

    async def stream(self, job: Job) -> AsyncGenerator[dict[str, object]]:
        """Yield this job's events: buffered history first, then live updates.

        History snapshot and subscriber registration happen with no ``await`` between
        them, so on the single loop thread it is atomic -- no event is dropped or
        duplicated across the hand-off.
        """
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        history = list(job.events)
        job.subscribers.add(queue)
        try:
            for payload in history:
                yield payload
                if payload is _SENTINEL:
                    return
            while True:
                payload = await queue.get()
                yield payload
                if payload is _SENTINEL:
                    return
        finally:
            job.subscribers.discard(queue)
