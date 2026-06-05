"""FastAPI dependency providers.

Shared singletons (settings, lock registry, job manager) live on ``app.state`` and are
handed to routes via these dependencies. Authentication is enforced here: loopback binds
are token-free, non-loopback binds require a bearer token.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from app.web.jobs import JobManager
from app.web.locks import LockRegistry
from app.web.settings import WebSettings


def get_settings(request: Request) -> WebSettings:
    """Return the resolved web settings."""
    return request.app.state.settings


def get_locks(request: Request) -> LockRegistry:
    """Return the shared lock registry."""
    return request.app.state.locks


def get_jobs(request: Request) -> JobManager:
    """Return the shared job manager."""
    return request.app.state.jobs


def require_auth(
    settings: WebSettings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    """Enforce bearer-token auth on non-loopback binds.

    Loopback-only servers skip auth for zero-friction local use; any networked bind
    requires ``Authorization: Bearer <token>`` so a LAN peer cannot mutate state.
    """
    if settings.token is None:
        return
    expected = f"Bearer {settings.token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")
