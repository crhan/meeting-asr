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


_LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_name(raw: str) -> str:
    """Return the lowercased hostname from a Host header, dropping any port.

    Handles IPv6 literals (``[::1]`` / ``[::1]:8765``) which embed colons, so a naive
    rsplit on ``:`` would mangle them.
    """
    raw = raw.strip()
    if raw.startswith("["):
        return raw[1:].split("]", 1)[0].lower()
    return (raw.rsplit(":", 1)[0] if ":" in raw else raw).lower()


def _is_trusted_loopback_host(host_header: str | None, settings: WebSettings) -> bool:
    """Whether a tokenless loopback request's Host header is a real loopback name.

    A DNS-rebinding page keeps its own hostname (e.g. ``evil.com``) in the Host header even
    after its DNS rebinds to 127.0.0.1, so the browser treats the request as same-origin
    (CORS does not help) while this server would otherwise serve it unauthenticated. Requiring
    a loopback Host blocks that: the attacker cannot forge a loopback Host from the victim's
    browser. A missing Host (anomalous for HTTP/1.1) is rejected.
    """
    if not host_header:
        return False
    name = _host_name(host_header)
    return name in _LOOPBACK_HOSTNAMES or name == settings.host.strip().lower()


def require_auth(
    settings: WebSettings = Depends(get_settings),
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
    host: str | None = Header(default=None),
) -> None:
    """Enforce bearer-token auth on non-loopback binds.

    Loopback-only servers skip auth for zero-friction local use; any networked bind
    requires the token so a LAN peer cannot mutate state.

    The token is accepted from either ``Authorization: Bearer <token>`` (used by fetch)
    or a ``?token=<token>`` query parameter. The query path exists because browser-managed
    transports -- ``EventSource`` (SSE) and the ``<audio>`` element -- cannot set request
    headers, so they would otherwise be locked out of a token-protected bind. This is a
    single-user LAN tool; the token may surface in access logs, an accepted tradeoff here.

    On a tokenless loopback bind, the request's Host must still be a loopback name. That
    closes DNS rebinding: without it a remote page could rebind to 127.0.0.1 and reach the
    unauthenticated secret-reveal / mutating routes as same-origin.
    """
    if settings.token is None:
        if not _is_trusted_loopback_host(host, settings):
            raise HTTPException(
                status_code=403,
                detail="Unexpected Host header for a loopback bind.",
            )
        return
    presented = token
    if authorization and authorization.startswith("Bearer "):
        presented = authorization[len("Bearer ") :]
    # Compare as bytes: secrets.compare_digest raises TypeError on non-ASCII str, so a
    # non-ASCII --token (or a client sending any non-ASCII token) would 500 instead of
    # returning 401. utf-8 bytes keep the compare constant-time and total.
    if presented is not None and secrets.compare_digest(
        presented.encode("utf-8"), settings.token.encode("utf-8")
    ):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")
