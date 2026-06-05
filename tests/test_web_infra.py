"""Unit tests for the web server's concurrency/infra primitives."""

from __future__ import annotations

import asyncio

from app.core.progress import CliProgressEvent
from app.web.locks import LockRegistry, project_lock_key, store_lock_key
from app.web.progress_bridge import QueueProgressReporter, event_to_payload


def test_event_to_payload_is_json_safe() -> None:
    event = CliProgressEvent(
        description="hi",
        total=5,
        completed=2,
        step_index=1,
        step_total=3,
        stage="ping",
        log_fields=(("k", "v"),),
    )
    payload = event_to_payload(event)
    assert payload["type"] == "progress"
    assert payload["description"] == "hi"
    assert payload["total"] == 5
    assert payload["completed"] == 2
    assert payload["log_fields"] == [["k", "v"]]


def test_queue_progress_reporter_uses_call_soon_threadsafe() -> None:
    """The reporter must marshal onto the loop, never touch asyncio objects directly."""

    calls: list[tuple] = []

    class FakeLoop:
        def call_soon_threadsafe(self, callback, *args):
            calls.append((callback, args))

    received: list[dict] = []
    reporter = QueueProgressReporter(FakeLoop(), received.append)  # type: ignore[arg-type]
    reporter(CliProgressEvent(description="x", stage="s"))

    # One marshalled call; invoking it delivers the payload to on_event.
    assert len(calls) == 1
    callback, args = calls[0]
    callback(*args)
    assert received and received[0]["description"] == "x"


def test_queue_progress_reporter_swallows_closed_loop() -> None:
    class ClosedLoop:
        def call_soon_threadsafe(self, callback, *args):
            raise RuntimeError("Event loop is closed")

    reporter = QueueProgressReporter(ClosedLoop(), lambda _payload: None)  # type: ignore[arg-type]
    # Must not raise even though the loop is gone.
    reporter(CliProgressEvent(description="x"))


def test_lock_registry_acquires_sorted_and_releases() -> None:
    registry = LockRegistry()
    a = project_lock_key("p1")
    b = store_lock_key("voiceprints")

    async def scenario() -> None:
        # Same lock objects are reused per key.
        assert registry.get(a) is registry.get(a)
        async with registry.acquire(b, a):
            assert registry.get(a).locked()
            assert registry.get(b).locked()
        # Released after the context exits.
        assert not registry.get(a).locked()
        assert not registry.get(b).locked()

    asyncio.run(scenario())


def test_lock_registry_serialises_same_key() -> None:
    registry = LockRegistry()
    key = project_lock_key("p1")
    order: list[str] = []

    async def worker(tag: str) -> None:
        async with registry.acquire(key):
            order.append(f"{tag}-enter")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-exit")

    async def scenario() -> None:
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(scenario())
    # Critical sections must not interleave.
    assert order in (
        ["a-enter", "a-exit", "b-enter", "b-exit"],
        ["b-enter", "b-exit", "a-enter", "a-exit"],
    )
