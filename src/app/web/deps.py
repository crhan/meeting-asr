"""FastAPI dependency providers.

Shared singletons (settings, lock registry, job manager) live on ``app.state`` and are
handed to routes via these dependencies. Authentication is enforced here: loopback binds
are token-free, non-loopback binds require a bearer token.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Query, Request

from app.core.project_refs import resolve_project_ref
from app.web.jobs import JobManager
from app.web.locks import LockRegistry
from app.web.settings import WebSettings


def get_settings(request: Request) -> WebSettings:
    """Return the resolved web settings."""
    return request.app.state.settings


def resolve_web_project_ref(project_ref: Path | str, settings: WebSettings) -> Path:
    """Resolve a project ref for the web, refusing any that escapes the projects dir.

    The shared resolver accepts filesystem paths as a CLI convenience. Over HTTP that
    would be a path-traversal hole: a request could pass ``../../..`` or an absolute path
    to read/write project files anywhere on disk (and extract audio from arbitrary
    locations). The web only ever needs ids/titles of projects under ``projects_dir``, so
    every router resolves refs through this single chokepoint with the boundary enforced.
    """
    return resolve_project_ref(
        project_ref, settings.projects_dir, restrict_to_projects_dir=True
    )


def get_locks(request: Request) -> LockRegistry:
    """Return the shared lock registry."""
    return request.app.state.locks


def get_jobs(request: Request) -> JobManager:
    """Return the shared job manager."""
    return request.app.state.jobs


def require_auth(
    settings: WebSettings = Depends(get_settings),
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Enforce bearer-token auth on non-loopback binds.

    Loopback-only servers skip auth for zero-friction local use; any networked bind
    requires the token so a LAN peer cannot mutate state.

    The token is accepted from either ``Authorization: Bearer <token>`` (used by fetch)
    or a ``?token=<token>`` query parameter. The query path exists because browser-managed
    transports -- ``EventSource`` (SSE) and the ``<audio>`` element -- cannot set request
    headers, so they would otherwise be locked out of a token-protected bind. This is a
    single-user LAN tool; the token may surface in access logs, an accepted tradeoff here.
    """
    if settings.token is None:
        return
    presented = token
    if authorization and authorization.startswith("Bearer "):
        presented = authorization[len("Bearer ") :]
    if presented is not None and secrets.compare_digest(presented, settings.token):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")
