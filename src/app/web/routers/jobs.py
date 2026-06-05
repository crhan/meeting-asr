"""Background job status and Server-Sent Events progress streaming.

The SSE endpoint replays a job's buffered history then streams live progress, so a client
that connects late (or reconnects) still sees the whole run. The ``/api/jobs/ping`` demo
job exercises the full executor -> progress-bridge -> SSE path end-to-end (P0 stack proof).
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.progress import CliProgressReporter, emit_progress
from app.web.deps import get_jobs, require_auth
from app.web.jobs import JobManager
from app.web.schemas import JobRef

router = APIRouter(
    prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(require_auth)]
)


def _ping_work(reporter: CliProgressReporter) -> dict[str, int]:
    """Emit a few heartbeats with blocking sleeps to prove non-blocking execution."""
    steps = 5
    for index in range(1, steps + 1):
        emit_progress(
            reporter,
            f"heartbeat {index}/{steps}",
            total=steps,
            completed=index,
            log_kind="heartbeat",
            stage="ping",
        )
        time.sleep(0.3)
    return {"pings": steps}


@router.post("/ping", response_model=JobRef)
def submit_ping(jobs: JobManager = Depends(get_jobs)) -> JobRef:
    """Submit a demo heartbeat job (P0 stack proof)."""
    job = jobs.submit("ping", _ping_work)
    return JobRef(job_id=job.id, kind=job.kind, status=job.status)


@router.get("")
def list_jobs(
    jobs: JobManager = Depends(get_jobs),
) -> dict[str, list[dict[str, object]]]:
    """List all known jobs."""
    return {"jobs": [job.public() for job in jobs.list_jobs()]}


@router.get("/{job_id}")
def get_job(job_id: str, jobs: JobManager = Depends(get_jobs)) -> dict[str, object]:
    """Return a single job's status snapshot."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return {**job.public(), "result": job.result}


@router.get("/{job_id}/events")
async def stream_job(
    job_id: str, jobs: JobManager = Depends(get_jobs)
) -> EventSourceResponse:
    """Stream a job's progress as Server-Sent Events (history then live)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    async def event_source():
        async for payload in jobs.stream(job):
            yield {"data": json.dumps(payload, ensure_ascii=False)}

    return EventSourceResponse(event_source())
