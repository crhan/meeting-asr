"""Bridge synchronous workflow progress events onto the asyncio event loop.

Long operations report progress through a synchronous ``CliProgressReporter`` callback
(``app.core.progress``). The web server runs those blocking operations in a
``ThreadPoolExecutor`` worker (via ``loop.run_in_executor``), so the reporter is invoked
from a non-loop thread. Touching asyncio objects (queues, events) from another thread is
unsafe -- the only sanctioned hand-off is ``loop.call_soon_threadsafe``. This module is
that hand-off; getting it wrong is how SSE silently stops delivering events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.core.progress import CliProgressEvent


def event_to_payload(event: CliProgressEvent) -> dict[str, object]:
    """Convert a progress event into a JSON-serialisable SSE payload."""
    return {
        "type": "progress",
        "description": event.description,
        "total": event.total,
        "completed": event.completed,
        "advance": event.advance,
        "step_index": event.step_index,
        "step_total": event.step_total,
        "reset_total": event.reset_total,
        "step_descriptions": list(event.step_descriptions),
        "log_kind": event.log_kind,
        "stage": event.stage,
        "project_id": event.project_id,
        "elapsed_seconds": event.elapsed_seconds,
        "last_success": event.last_success,
        "next_action": event.next_action,
        "log_fields": [list(pair) for pair in event.log_fields],
    }


class QueueProgressReporter:
    """A ``CliProgressReporter`` safe to call from a worker thread.

    Each call marshals the event onto the event loop and forwards it to ``on_event``,
    which then runs on the loop thread and may safely touch asyncio queues.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[dict[str, object]], None],
    ) -> None:
        self._loop = loop
        self._on_event = on_event

    def __call__(self, event: CliProgressEvent) -> None:
        """Marshal one progress event onto the loop thread."""
        payload = event_to_payload(event)
        try:
            self._loop.call_soon_threadsafe(self._on_event, payload)
        except RuntimeError:
            # Loop already closed (server shutting down); drop the event silently.
            pass
