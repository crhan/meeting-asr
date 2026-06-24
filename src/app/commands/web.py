"""`meeting-asr web` -- launch the local web UI server.

Thin Typer adapter registered as a single root command (like ``doctor``/``paths``), so
it picks up the standard bilingual help handling. The web package and its
FastAPI/uvicorn dependencies are imported lazily so a broken or stale install surfaces
an actionable install hint instead of a stack trace.
"""

from __future__ import annotations

from pathlib import Path

import typer

_INSTALL_HINT = (
    "Web UI dependencies are not installed.\n"
    "Web server packages are default dependencies; this install is stale or incomplete.\n"
    "From a checkout, refresh the global editable tool:\n"
    "  scripts/install-tool.sh\n"
    "Then verify it with:\n"
    "  scripts/install-tool.sh --check\n"
    "For a published tool install, reinstall the default package:\n"
    "  uv tool install meeting-asr --python 3.14 --reinstall --refresh"
)


def _port_available(host: str, port: int) -> bool:
    """Return whether ``(host, port)`` can be bound right now.

    A pre-flight check gives a clean, actionable error instead of uvicorn's confusing
    "started ... address already in use ... shutdown" sequence when the port is taken.

    The socket family must match the host: an ``AF_INET`` socket cannot bind an IPv6
    address like ``::1``/``::``, so an IPv6 host would always (wrongly) look unavailable.
    """
    import ipaddress
    import socket

    try:
        family = (
            socket.AF_INET6
            if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address)
            else socket.AF_INET
        )
    except ValueError:
        # Hostname rather than a literal IP; default to IPv4 for the probe.
        family = socket.AF_INET

    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def command(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Bind host. Non-loopback binds require a token."
    ),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    projects_dir: Path | None = typer.Option(
        None,
        "--projects-dir",
        help="Projects parent directory (default: XDG data dir).",
    ),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Voiceprint/lexicon store directory (default: XDG data dir). "
        "Point this at a copy when experimenting to protect the real library.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Bearer token for non-loopback binds (auto-generated if omitted).",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the default browser on startup."
    ),
) -> None:
    """Start the web UI server and listen for connections."""
    try:
        # uvicorn is a default dependency but run_server imports it lazily, so probe it up
        # front here too. Otherwise a partial install (FastAPI present, uvicorn missing) would
        # pass this block, print the serving URL, and only then crash with a raw
        # ModuleNotFoundError instead of this actionable install hint.
        import uvicorn  # noqa: F401

        from app.web.server import (
            authenticated_url,
            base_url,
            resolve_token,
            run_server,
        )
        from app.web.settings import WebSettings
    except ImportError as exc:  # stale or incomplete install
        typer.echo(_INSTALL_HINT, err=True)
        raise typer.Exit(code=1) from exc

    if not _port_available(host, port):
        typer.echo(
            f"Port {port} on {host} is already in use by another process.\n"
            f"Pick a free port, e.g.:  meeting-asr web --port {port + 1}",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_token = resolve_token(host, token)
    settings = WebSettings(
        host=host,
        port=port,
        projects_dir=projects_dir.expanduser().resolve() if projects_dir else None,
        store_dir=store_dir.expanduser().resolve() if store_dir else None,
        open_browser=open_browser,
        token=resolved_token,
    )
    typer.echo(f"meeting-asr web serving at {base_url(settings)}")
    if resolved_token is not None:
        typer.echo(
            "This bind requires a token. Open this URL to authenticate automatically:\n"
            f"  {authenticated_url(settings)}\n"
            f"Token (for API clients / manual entry): {resolved_token}"
        )
    elif settings.is_local:
        # Loopback is shared by every local user on a multi-user host, and tokenless
        # loopback skips auth (reveal + mutating routes included). Nudge shared-host users
        # toward --token; single-user laptops can ignore it.
        typer.echo(
            "Warning: this loopback bind has no token; other local users on this host "
            "can reach it. On a shared machine, pass --token to require authentication.",
            err=True,
        )
    run_server(settings)
