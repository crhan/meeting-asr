"""Unit tests for the in-process web job manager."""

from __future__ import annotations

import asyncio

import pytest

from app.web.jobs import Job, JobManager, _MAX_EVENTS_PER_JOB, _MAX_JOBS, _SENTINEL
from app.web.locks import LockRegistry


def test_submit_without_bound_loop_does_not_register_job() -> None:
    """A misconfigured manager should fail without leaving an unreachable queued job."""
    manager = JobManager(LockRegistry())

    with pytest.raises(RuntimeError, match="loop not bound"):
        manager.submit("x", lambda _reporter: None)

    assert manager.list_jobs() == []


def test_job_event_history_is_bounded() -> None:
    """Progress history must not grow without bound for long jobs."""
    manager = JobManager(LockRegistry())
    job = Job(id="j", kind="test", project_id=None)

    for index in range(_MAX_EVENTS_PER_JOB + 10):
        manager._publish(job, {"index": index})

    assert len(job.events) == _MAX_EVENTS_PER_JOB
    assert job.events[0] == {"index": 10}


def test_full_subscriber_queue_still_receives_end_sentinel() -> None:
    """Bounded subscriber queues must not hang by dropping the terminal event."""
    manager = JobManager(LockRegistry())
    job = Job(id="j", kind="test", project_id=None)
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"index": 1})
    job.subscribers.add(queue)

    manager._publish(job, _SENTINEL)

    assert queue.get_nowait() is _SENTINEL


def test_full_subscriber_queue_still_receives_terminal_status() -> None:
    """Back-pressure must not drop the terminal done/error status frame.

    A non-reconnecting SSE consumer would otherwise see the stream end while still
    believing the job is running -- and the error frame carries the failure message.
    """
    manager = JobManager(LockRegistry())
    job = Job(id="j", kind="test", project_id=None)
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait({"type": "progress", "completed": 1})
    job.subscribers.add(queue)

    terminal: dict[str, object] = {"type": "status", "status": "error", "error": "boom"}
    manager._publish(job, terminal)

    assert queue.get_nowait() == terminal


def test_full_subscriber_queue_still_drops_ordinary_progress() -> None:
    """Ordinary progress events stay droppable under back-pressure (history replays them)."""
    manager = JobManager(LockRegistry())
    job = Job(id="j", kind="test", project_id=None)
    queue = asyncio.Queue(maxsize=1)
    first = {"type": "progress", "completed": 1}
    queue.put_nowait(first)
    job.subscribers.add(queue)

    manager._publish(job, {"type": "progress", "completed": 2})

    assert queue.get_nowait() == first  # the queued event was not evicted


def test_finished_jobs_are_pruned_to_bound_memory() -> None:
    """Old terminal jobs are pruned when the registry exceeds its cap."""
    manager = JobManager(LockRegistry())
    with manager._jobs_lock:
        for index in range(_MAX_JOBS + 5):
            manager._jobs[str(index)] = Job(
                id=str(index),
                kind="test",
                project_id=None,
                created_at=float(index),
                status="done",
            )
        manager._prune_jobs_locked()

    jobs = manager.list_jobs()
    assert len(jobs) == _MAX_JOBS
    assert {job.id for job in jobs}.isdisjoint({"0", "1", "2", "3", "4"})


def test_submit_deduplicates_identical_active_job() -> None:
    """Re-submitting the same (kind, project) re-attaches instead of double-running."""
    manager = JobManager(LockRegistry())
    running = Job(
        id="j1", kind="correction-polish", project_id="/p/a", status="running"
    )
    with manager._jobs_lock:
        manager._jobs[running.id] = running

    async def run() -> tuple[Job, bool]:
        manager.bind_loop(asyncio.get_running_loop())
        return manager.submit("correction-polish", lambda _r: None, project_id="/p/a")

    job, existing = asyncio.run(run())
    assert existing is True
    assert job is running


def test_submit_does_not_deduplicate_terminal_or_other_project() -> None:
    """Terminal jobs and other projects never absorb a new submit."""
    manager = JobManager(LockRegistry())
    finished = Job(id="j1", kind="correction-polish", project_id="/p/a", status="done")
    other = Job(id="j2", kind="correction-polish", project_id="/p/b", status="running")
    with manager._jobs_lock:
        manager._jobs[finished.id] = finished
        manager._jobs[other.id] = other

    async def run() -> tuple[Job, bool]:
        manager.bind_loop(asyncio.get_running_loop())
        job, existing = manager.submit(
            "correction-polish", lambda _r: None, project_id="/p/a"
        )
        # Cancel the freshly spawned task so the loop shuts down cleanly.
        await asyncio.sleep(0)
        if job.task is not None:
            job.task.cancel()
            await asyncio.sleep(0)
        return job, existing

    job, existing = asyncio.run(run())
    assert existing is False
    assert job.id not in {"j1", "j2"}


def test_cancel_queued_job_waiting_on_lock() -> None:
    """A job stuck in the lock queue cancels immediately with a cancelled terminal frame."""
    locks = LockRegistry()
    manager = JobManager(locks)

    async def run() -> Job:
        manager.bind_loop(asyncio.get_running_loop())
        async with locks.acquire("project:/p/a"):
            job, _ = manager.submit("pipeline-run", lambda _r: None, project_id="/p/a")
            # Let the task start and block on the held lock, then cancel it.
            await asyncio.sleep(0.05)
            assert job.status == "queued"
            manager.request_cancel(job.id)
            await asyncio.wait_for(job.done.wait(), timeout=2)
            return job

    job = asyncio.run(run())
    assert job.status == "cancelled"
    assert job.events[-1] is _SENTINEL
    statuses = [e.get("status") for e in job.events if e.get("type") == "status"]
    assert "cancelled" in statuses


def test_cooperative_cancel_unwinds_at_progress_checkpoint() -> None:
    """A running job cancels at its next progress emission."""
    import time as time_mod

    from app.core.progress import emit_progress

    manager = JobManager(LockRegistry())

    def work(reporter) -> None:
        # Slow enough that the cancel flag lands before the loop finishes; the
        # checkpoint raise then unwinds at the next emit.
        for index in range(1000):
            emit_progress(reporter, f"step {index}", total=1000, completed=index)
            time_mod.sleep(0.01)

    async def run() -> Job:
        manager.bind_loop(asyncio.get_running_loop())
        job, _ = manager.submit("pipeline-run", work, project_id="/p/c")
        while job.status == "queued":
            await asyncio.sleep(0.01)
        manager.request_cancel(job.id)
        await asyncio.wait_for(job.done.wait(), timeout=5)
        return job

    job = asyncio.run(run())
    assert job.status == "cancelled"
    assert job.error == "Cancelled by user."
