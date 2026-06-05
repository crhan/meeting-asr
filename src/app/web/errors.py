"""Map domain exceptions to HTTP responses at the web boundary.

Business functions are written for the CLI and raise a mix of:

* ``typer.BadParameter`` for invalid user input. Typer 0.26 vendored Click into a private
  package and no longer exposes the Click exception classes (see CLAUDE.md), so we must
  NOT import them and must NOT ``isinstance``-check against them. We duck-type by shape
  (a usage error carries a ``format_message``/``message`` and an ``exit_code``).
* ``ValueError`` for invalid arguments resolved deeper in the stack.
* ``FileNotFoundError`` for missing projects/artifacts.

Everything else is a 500 with the message preserved; the traceback is logged server-side.
"""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.voiceprint_review_service import CaptureConflictError


def _looks_like_usage_error(exc: BaseException) -> bool:
    """Return whether ``exc`` quacks like a Typer/Click usage error.

    Shape-based detection deliberately avoids importing typer's private vendored Click
    exception classes, matching the duck-typing convention used elsewhere in the CLI.
    """
    return hasattr(exc, "format_message") or (
        exc.__class__.__name__ in {"BadParameter", "UsageError", "MissingParameter"}
    )


def _usage_error_message(exc: BaseException) -> str:
    """Extract a human-readable message from a usage-error-shaped exception."""
    formatter = getattr(exc, "format_message", None)
    if callable(formatter):
        try:
            return str(formatter())
        except Exception:  # noqa: BLE001
            pass
    message = getattr(exc, "message", None)
    return str(message) if message else str(exc)


def _problem(status_code: int, detail: str, *, kind: str) -> JSONResponse:
    """Build a small problem-detail JSON body."""
    return JSONResponse(
        status_code=status_code, content={"detail": detail, "error": kind}
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register web-boundary exception handlers on the app."""

    @app.exception_handler(CaptureConflictError)
    async def _conflict(_: Request, exc: CaptureConflictError) -> JSONResponse:
        return _problem(409, str(exc), kind="conflict")

    @app.exception_handler(FileNotFoundError)
    async def _not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return _problem(404, str(exc), kind="not_found")

    @app.exception_handler(FileExistsError)
    async def _exists(_: Request, exc: FileExistsError) -> JSONResponse:
        # e.g. a merge into a non-empty output dir without force -- a conflict, not a 500.
        return _problem(409, str(exc), kind="conflict")

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return _problem(400, str(exc), kind="bad_request")

    @app.exception_handler(Exception)
    async def _fallback(_: Request, exc: Exception) -> JSONResponse:
        if _looks_like_usage_error(exc):
            return _problem(400, _usage_error_message(exc), kind="bad_request")
        traceback.print_exc()
        return _problem(500, str(exc) or exc.__class__.__name__, kind="internal")
