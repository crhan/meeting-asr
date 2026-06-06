"""FastAPI application assembly and uvicorn entry point.

``create_app`` wires routers, exception handlers, shared singletons, and (when built)
the React SPA. ``run_server`` resolves settings, optionally generates an auth token for
non-loopback binds, and runs uvicorn with a single worker (the whole concurrency model
assumes one process; see ``locks`` and ``jobs``).
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import webbrowser
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from app.web.deps import require_auth
from app.web.errors import install_exception_handlers
from app.web.jobs import JobManager
from app.web.locks import LockRegistry
from app.web.routers import audio as audio_router
from app.web.routers import config as config_router
from app.web.routers import corrections as corrections_router
from app.web.routers import jobs as jobs_router
from app.web.routers import lexicon as lexicon_router
from app.web.routers import pipeline as pipeline_router
from app.web.routers import projects as projects_router
from app.web.routers import speakers as speakers_router
from app.web.routers import voiceprints as voiceprints_router
from app.web.settings import WebSettings

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: WebSettings) -> FastAPI:
    """Build the FastAPI app for one resolved settings object."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Capture the running loop so jobs can be spawned from sync-route worker threads.
        app.state.jobs.bind_loop(asyncio.get_running_loop())
        # Reclaim any capture backup dirs orphaned by a previous crash.
        from app.core.voiceprint_review_service import cleanup_orphan_backups

        cleanup_orphan_backups()
        yield

    app = FastAPI(title="meeting-asr web", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.locks = LockRegistry()
    app.state.jobs = JobManager(app.state.locks)

    # CORS is only for the Vite dev server (pinned to :5173), and only for the rare direct-call
    # setup -- the normal `npm run dev` proxies /api server-side, so the browser is same-origin
    # and needs no CORS. The production SPA is served same-origin from here. Allow ONLY the Vite
    # dev origins: a broad "any localhost port" rule would let any other local page (another dev
    # server, an XSS'd local app) read loopback-only secret-reveal responses (GET
    # /api/config?reveal=true) cross-origin and exfiltrate DashScope/OSS keys.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)
    app.include_router(projects_router.router)
    app.include_router(speakers_router.router)
    app.include_router(voiceprints_router.router)
    app.include_router(pipeline_router.router)
    app.include_router(corrections_router.router)
    app.include_router(lexicon_router.router)
    app.include_router(config_router.router)
    app.include_router(audio_router.router)
    app.include_router(jobs_router.router)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Liveness probe + bind metadata for the client (always unauthenticated).

        ``is_local`` lets the SPA hide loopback-only affordances (e.g. revealing secret
        config values) on a networked bind instead of offering a button that 403s.
        """
        return {
            "status": "ok",
            "auth_required": settings.token is not None,
            "is_local": settings.is_local,
        }

    @app.get("/api/auth/check", dependencies=[Depends(require_auth)])
    def auth_check() -> dict[str, bool]:
        """Token probe: 200 if the presented credential is valid, else 401.

        The SPA calls this to decide whether to render the app or prompt for a token,
        without depending on any particular project existing.
        """
        return {"ok": True}

    _mount_spa(app)
    return app


def _mount_spa(app: FastAPI) -> None:
    """Serve the built SPA at ``/`` with history-API fallback, if it exists.

    Before the frontend is built (early P0), there is no ``static/index.html``; we serve
    a short JSON hint at ``/`` instead so the API is still usable.
    """
    index_html = _STATIC_DIR / "index.html"
    if not index_html.is_file():

        @app.get("/")
        def _no_spa() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "Web UI assets not built. Run `npm --prefix web run build`.",
                    "api": "/api/projects",
                }
            )

        return

    assets_dir = _STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(index_html)

    @app.get("/{full_path:path}")
    def _spa_fallback(full_path: str) -> FileResponse:
        """Serve a real static file if present, else fall back to index for SPA routing."""
        # An unknown /api/... path must 404 as an API miss, not fall back to index.html:
        # otherwise a misspelled fetch or an external client gets 200 + HTML and only fails
        # later parsing it as JSON. Real API routes are registered before this catch-all, so
        # only genuinely-unmatched api paths reach here.
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"Not found: /{full_path}")
        candidate = (_STATIC_DIR / full_path).resolve()
        # Reject path traversal: only serve files that stay inside the static root.
        if candidate.is_file() and candidate.is_relative_to(_STATIC_DIR.resolve()):
            return FileResponse(candidate)
        return FileResponse(index_html)


def run_server(settings: WebSettings) -> None:
    """Run uvicorn for the given settings (blocking)."""
    import uvicorn

    app = create_app(settings)
    if settings.open_browser:
        _open_browser_when_ready(settings)
    uvicorn.run(
        app, host=settings.host, port=settings.port, workers=1, log_level="info"
    )


def resolve_token(host: str, explicit_token: str | None) -> str | None:
    """Return the auth token to enforce: explicit, auto-generated for LAN, or None.

    Loopback binds need no token; any other bind gets a generated token if none was
    supplied so the API is never silently exposed unauthenticated on a network.
    """
    probe = WebSettings(
        host=host,
        port=0,
        projects_dir=None,
        store_dir=None,
        open_browser=False,
        token=None,
    )
    if explicit_token:
        return explicit_token
    if probe.is_local:
        return None
    return secrets.token_urlsafe(24)


def base_url(settings: WebSettings) -> str:
    """Return the server's base URL, bracketing IPv6 literal hosts.

    IPv6 literals (``::1``, ``::``) must be wrapped in ``[...]`` in a URL authority, else
    ``http://::1:8765/`` is rejected by browsers.
    """
    host = f"[{settings.host}]" if ":" in settings.host else settings.host
    return f"http://{host}:{settings.port}/"


def authenticated_url(settings: WebSettings) -> str:
    """Return the entry URL, carrying ``?token=`` when the bind is token-protected.

    Opening (or printing) this URL is the token handoff: the SPA reads ``?token=`` on
    first load, stores it, and strips it from the address bar. Without this, a fresh
    browser on a non-loopback bind would 401 on every API call.
    """
    base = base_url(settings)
    if settings.token:
        # Percent-encode: an explicit --token may contain URL-reserved chars (& # +), which
        # would otherwise be parsed as a different token. The SPA reads it via URLSearchParams,
        # which decodes the percent-encoding back to the original.
        return f"{base}?token={quote(settings.token, safe='')}"
    return base


def _open_browser_when_ready(settings: WebSettings) -> None:
    """Open the default browser shortly after the server starts."""
    url = authenticated_url(settings)

    def _open() -> None:
        import time

        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
