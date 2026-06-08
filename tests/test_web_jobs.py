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
