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
import threading
import traceback
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from dataclasses import dataclass, field

from app.core.progress import CliProgressReporter
from app.web.locks import LockRegistry, project_lock_key
from app.web.progress_bridge import JobCancelledError, QueueProgressReporter

# A unit of work: given a progress reporter, do the blocking work and return a result.
JobFn = Callable[[CliProgressReporter], Any]

_SENTINEL: dict[str, object] = {"type": "end"}
_MAX_JOBS = 100
_MAX_EVENTS_PER_JOB = 500
_SUBSCRIBER_QUEUE_SIZE = 100


@dataclass(slots=True)
class Job:
    """One queued or running background operation."""

    id: str
    kind: str
    project_id: str | None
    created_at: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error | cancelled
    result: Any = None
    error: str | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, object]]] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # Full lock-key set (project + store locks) for "queued: waiting on whom" reporting.
    lock_keys: tuple[str, ...] = ()
    # Cooperative-cancel flag, checked from the executor thread at progress checkpoints;
    # a threading.Event because asyncio primitives must not be touched off-loop.
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    # The job's asyncio task; loop-thread only. Cancelling it while the job still waits
    # in the lock queue aborts cleanly before any work starts.
    task: asyncio.Future[None] | None = None

    def public(self) -> dict[str, object]:
        """Return a JSON-safe snapshot for the jobs API."""
        return {
            "id": self.id,
            "kind": self.kind,
            "project_id": self.project_id,
            "status": self.status,
            "error": self.error,
            "event_count": len(self.events),
            "created_at": self.created_at,
            "cancel_requested": self.cancel_requested.is_set(),
        }


class JobManager:
    """Owns background jobs and their per-project serialisation.

    Per-project serialisation is delegated to the shared :class:`LockRegistry` so jobs and
    inline mutating routes for the same project contend on one lock, not two disjoint pools.
    """

    def __init__(self, locks: LockRegistry) -> None:
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._locks = locks
        self._loop: asyncio.AbstractEventLoop | None = None
        # Strong references to in-flight job tasks: the event loop only keeps weak refs to
        # tasks, so a fire-and-forget ensure_future could be garbage-collected mid-run,
        # silently orphaning the job in "queued"/"running". Touched only on the loop thread.
        self._tasks: set[asyncio.Future[None]] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop at startup.

        Jobs are often submitted from FastAPI's sync-route worker threads, which have no
        running loop; scheduling work onto the captured loop via ``call_soon_threadsafe``
        is the thread-safe way to spawn the job task from anywhere.
        """
        self._loop = loop

    def get(self, job_id: str) -> Job | None:
        """Return a job by id, or None."""
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """Return all known jobs (newest-tracked order is not guaranteed)."""
        with self._jobs_lock:
            return list(self._jobs.values())

    def submit(
        self,
        kind: str,
        fn: JobFn,
        *,
        project_id: str | None = None,
        store_locks: Sequence[str] = (),
    ) -> tuple[Job, bool]:
        """Schedule ``fn`` to run in the background; returns ``(job, existing)``.

        ``store_locks`` names shared-store locks (e.g. ``store_lock_key("lexicon")``) the
        job must hold for its whole duration -- a pipeline run learns into the lexicon and
        deletes global voiceprint samples during stabilization, so it has to contend on the
        same store locks the inline lexicon/voiceprint routes take, not just the per-project
        lock. Acquired together with the project lock in deadlock-free sorted order.

        Submitting while an identical ``(kind, project_id)`` job is queued/running returns
        that in-flight job with ``existing=True`` instead of double-running it: a re-click
        re-attaches to the live progress (SSE history replay), it does not re-bill the LLM
        or queue duplicate heavy work. Differing options are deliberately merged into the
        older job -- acceptable for a single-user tool.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("JobManager loop not bound; call bind_loop at startup.")
        job = Job(id=uuid.uuid4().hex, kind=kind, project_id=project_id)
        with self._jobs_lock:
            if project_id is not None:
                for candidate in self._jobs.values():
                    if (
                        candidate.kind == kind
                        and candidate.project_id == project_id
                        and candidate.status in {"queued", "running"}
                        # A cancel-requested job is dying (cooperative cancel keeps it
                        # "running" until the next checkpoint); attaching a retry to it
                        # would make the retry silently terminate as cancelled.
                        and not candidate.cancel_requested.is_set()
                    ):
                        return candidate, True
            self._jobs[job.id] = job
            self._prune_jobs_locked()
            if len(self._jobs) > _MAX_JOBS:
                self._jobs.pop(job.id, None)
                raise RuntimeError(
                    "Too many active jobs; wait for current jobs to finish."
                )
        keys = tuple(store_locks)

        def _spawn() -> None:
            # Runs on the loop thread. Keep a strong reference until the task finishes --
            # the loop alone holds only weak refs and could let the task be GC'd mid-run.
            task = asyncio.ensure_future(self._run(job, fn, keys))
            job.task = task
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        loop.call_soon_threadsafe(_spawn)
        return job, False

    def request_cancel(self, job_id: str) -> Job | None:
        """Request cancellation of a job; returns the job, or None when unknown.

        Queued jobs (still waiting in the lock queue) are cancelled immediately via
        ``task.cancel()``. Running jobs are cancelled cooperatively: the flag is checked
        at every progress checkpoint, so the latency equals the gap between two progress
        events. Terminal jobs are a no-op.
        """
        job = self.get(job_id)
        if job is None:
            return None
        loop = self._loop
        if loop is None:
            return job

        def _cancel() -> None:
            # Loop thread: job.task and status transitions are only touched here.
            if job.status == "queued" and job.task is not None:
                job.task.cancel()
            elif job.status == "running" and not job.cancel_requested.is_set():
                job.cancel_requested.set()
                self._publish(
                    job,
                    {"type": "status", "status": "running", "cancel_requested": True},
                )

        loop.call_soon_threadsafe(_cancel)
        return job

    async def _run(self, job: Job, fn: JobFn, store_locks: tuple[str, ...]) -> None:
        keys: list[str] = []
        if job.project_id is not None:
            # Same key the inline routes use, so jobs and inline saves serialise together.
            keys.append(project_lock_key(job.project_id))
        keys.extend(store_locks)
        job.lock_keys = tuple(keys)
        try:
            if keys:
                # Tell the subscriber whom this job is waiting on -- an unexplained
                # "Queued…" while another project's run holds the store locks reads
                # like a hang. Status events are must-deliver.
                waiting_on = self._waiting_on(job)
                if waiting_on:
                    self._publish(
                        job,
                        {
                            "type": "status",
                            "status": "queued",
                            "waiting_on": waiting_on,
                        },
                    )
                # acquire() sorts keys, so project + store locks never deadlock.
                async with self._locks.acquire(*keys):
                    await self._execute(job, fn)
            else:
                await self._execute(job, fn)
        except asyncio.CancelledError:
            # request_cancel() cancels the task only while it still waits in the lock
            # queue; nothing has run yet, so this is a clean abort (not re-raised: the
            # cancellation is ours, ending the task normally is the intended outcome).
            job.status = "cancelled"
            job.error = "Cancelled while queued."
            self._publish(
                job, {"type": "status", "status": "cancelled", "error": job.error}
            )
            job.done.set()
            self._publish(job, _SENTINEL)

    def _waiting_on(self, job: Job) -> list[dict[str, object]]:
        """Running jobs whose lock keys intersect this job's (whom it queues behind)."""
        wanted = set(job.lock_keys)
        with self._jobs_lock:
            return [
                {
                    "job_id": other.id,
                    "kind": other.kind,
                    "project_id": other.project_id,
                }
                for other in self._jobs.values()
                if other.id != job.id
                and other.status == "running"
                and wanted & set(other.lock_keys)
            ]

    async def _execute(self, job: Job, fn: JobFn) -> None:
        loop = asyncio.get_running_loop()
        job.status = "running"
        self._publish(job, {"type": "status", "status": "running"})
        reporter = QueueProgressReporter(
            loop,
            lambda payload: self._publish(job, payload),
            should_cancel=job.cancel_requested.is_set,
        )
        try:
            job.result = await loop.run_in_executor(None, fn, reporter)
            job.status = "done"
            self._publish(job, {"type": "status", "status": "done"})
        except JobCancelledError:
            # Cooperative cancel: the worker unwound at a progress checkpoint. The
            # project is left in the same partial state a CLI Ctrl-C would leave;
            # a re-run reuses completed stages (audio/OSS) like any interrupted run.
            job.status = "cancelled"
            job.error = "Cancelled by user."
            self._publish(
                job, {"type": "status", "status": "cancelled", "error": job.error}
            )
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
        if len(job.events) > _MAX_EVENTS_PER_JOB:
            del job.events[: len(job.events) - _MAX_EVENTS_PER_JOB]
        # A slow consumer may overflow its queue; ordinary progress events can be dropped
        # (history replay on reconnect recovers them), but the end sentinel and status
        # transitions must survive: dropping the terminal done/error frame leaves a
        # non-reconnecting client believing the job is still running -- and the error
        # frame carries the failure message.
        must_deliver = payload is _SENTINEL or payload.get("type") == "status"
        for queue in tuple(job.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                if must_deliver:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass

    def _prune_jobs_locked(self) -> None:
        """Keep memory bounded by pruning oldest terminal jobs first."""
        while len(self._jobs) > _MAX_JOBS:
            terminal = [
                (job.created_at, job_id)
                for job_id, job in self._jobs.items()
                if job.status in {"done", "error", "cancelled"}
            ]
            if not terminal:
                return
            _, oldest_id = min(terminal)
            self._jobs.pop(oldest_id, None)

    async def stream(self, job: Job) -> AsyncGenerator[dict[str, object]]:
        """Yield this job's events: buffered history first, then live updates.

        History snapshot and subscriber registration happen with no ``await`` between
        them, so on the single loop thread it is atomic -- no event is dropped or
        duplicated across the hand-off.
        """
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(
            maxsize=_SUBSCRIBER_QUEUE_SIZE
        )
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
