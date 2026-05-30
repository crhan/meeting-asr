"""Structured JSON logging for the eval scripts — one flat JSON object per line.

These scripts run as long background jobs whose logs the *agent* reads back with
cat / grep / json.loads, not a human watching a terminal. So the optimal format
is flat JSON lines, not rich's ANSI-coloured tables (escape codes pollute the
file the agent parses) nor stdlib's free-form text (fields can't be extracted
unambiguously). Every call like ``log.info("proj_done", proj=p, sps=29.3)``
renders to ``{"proj": "p", "sps": 29.3, "event": "proj_done", "level": "info",
"timestamp": "..."}`` and flushes immediately via PrintLoggerFactory.
"""

from __future__ import annotations

import sys

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ],
    logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()
"""Shared structured logger. Import as ``from _log import log``."""
