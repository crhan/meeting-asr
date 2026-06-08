"""Local web UI server for Meeting-ASR.

This package mirrors the Textual TUI feature set over an HTTP/JSON + SSE API so the
workflows (project management, speaker review, voiceprint review, corrections, and the
ASR intake pipeline) can run in a browser. The web layer is a thin adapter: it reuses
the existing presentation-neutral business functions and only adds HTTP marshalling,
per-project/per-store locking, and a small in-process job queue for long operations.

Web-only dependencies (fastapi, uvicorn, sse-starlette) live behind the optional
``web`` extra, so importing this package fails loudly with an install hint when the
extra is missing -- the root CLI never imports it eagerly.
"""

from __future__ import annotations
